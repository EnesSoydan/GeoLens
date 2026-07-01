# MİM-2a: AI Pipeline Tasarımı

> Plan/dokümantasyon. Uygulama kodu içermez.

## Online Akış (sorgu anı)
```mermaid
flowchart TD
    A[Görüntü yükleme<br/>multipart/form-data] --> B[Doğrulama<br/>MIME + boyut + EXIF sıyır]
    B --> C[Preprocessing<br/>resize 322x322 + ImageNet normalize]
    C --> D[Embedding<br/>DINOv2 ViT-B/14 + SALAD → 8448-dim, fp16]
    %% Not: 322x322 zorunlu - bkz. çözünürlük notu
    D --> E[L2 normalize]
    E --> F[FAISS arama<br/>IndexFlatIP top-K]
    F --> G[ID eşleme<br/>faiss_id → SQLite metadata]
    G --> H{is_confident<br/>&& include_heatmap?}
    H -->|evet| I[XAI: Eigen-CAM<br/>sorgu overlay<br/>status=included]
    H -->|hayır| J[XAI atla<br/>heatmap=null<br/>status=skipped_low_confidence / disabled]
    I --> K[Yanıt birleştirme<br/>top-K + heatmap? + status + sürüm]
    J --> K
    K --> L[JSON response]
```

> **Çözünürlük notu (322x322 zorunlu):** Resmi SALAD reposu (github.com/serizba/salad)
> eval script'i **322x322** kullanır ve MSLS-val R@1 %92.2 bu çözünürlükte alınmıştır.
> DINOv2 patch boyutu 14 → 322² **529 patch**, 224² yalnız 256 patch üretir; SALAD'ın
> 64-cluster optimal-transport aggregation'ı 529-patch dağılımıyla kalibre edilmiştir.
> Bu nedenle 224² sadece hafif doğruluk kaybı değil, **kalibrasyon uyumsuzluğu** yaratır.
> (Önceki 224² kararı düzeltildi.)

## Sequence (katmanlar arası)
```mermaid
sequenceDiagram
    participant U as İstemci (Gradio)
    participant API as FastAPI Router
    participant PS as PredictionService
    participant ES as EmbeddingService (GPU)
    participant RS as RetrievalService (FAISS)
    participant XS as XAIService (GPU)
    participant DB as MetadataRepo (SQLite)
    Note over ES,XS: Tek GLOBAL paylaşılan gpu_semaphore(1)<br/>her iki servis aynı örneği kullanır
    U->>API: POST /predict (image)
    API->>API: doğrula + EXIF sıyır
    API->>PS: predict(image, top_k)
    PS->>ES: embed(image)  [global gpu_sem]
    ES-->>PS: query_vector (8448-d)
    PS->>RS: search(vector, k)
    RS-->>PS: [(faiss_id, sim)...]
    PS->>DB: get_metadata(faiss_ids)
    DB-->>PS: [(lat,lon,seq)...]
    alt is_confident && include_heatmap
        PS->>XS: eigen_cam(image)  [global gpu_sem]
        XS-->>PS: heatmap_png
    else düşük güven veya kapalı
        Note over PS: XAI atlanır<br/>heatmap=null + status
    end
    PS-->>API: PredictionResult
    API-->>U: JSON (top-K + heatmap? + status)
```

## Offline Pipeline (referans DB → FAISS)
```mermaid
flowchart TD
    A[MSLS-val cph+sf indir<br/>scripts/download_msls] --> B[Doğrula/filtrele<br/>bozuk + eksik GPS ele]
    B --> C{image_role?}
    C -->|database| D[Preprocess 322x322 normalize]
    C -->|query| Q[INDEX'E GİRMEZ<br/>yalnız SQLite role=query<br/>Recall@N için ayrılır]
    D --> E[Batch embedding<br/>DINOv2+SALAD, fp16, checkpoint]
    E --> F[L2 normalize]
    F --> G[FAISS IndexIDMap over IndexFlatIP<br/>SADECE database + db_id]
    G --> H[Kaydet: faiss.index]
    D --> I[SQLite: role=database]
    Q --> I
    G --> M[Manifest: model_hash, dim, dataset,<br/>index_size = database görüntü sayısı]
    H --> M
```
**Adımlar:** indir → doğrula (GPS'siz/bozuk ele) → **role'e göre ayır** → *database*: 224² normalize → batch embed (fp16+checkpoint, risk #24) → L2 normalize → `IndexIDMap`(IndexFlatIP, kosinüs) SADECE database + db_id → kaydet → manifest. *query*: index'e **hiç girmez**, yalnız SQLite'a (`role=query`) yazılır ve `evaluate.py` için ayrılır.

> **Data leakage önlemi (kritik):** Yalnızca `image_role='database'` görüntüler FAISS index'ine yazılır. `query` görüntüleri index'te bulunmadığından Recall@N ölçümü suni yüksek skora (kendisiyle eşleşme) düşmez. Bu, MİM-4 değerlendirme protokolünün geçerliliğinin ön koşuludur.

## Model Versiyonlama (risk #15)
`ml/model_registry.py` manifest: backbone+sürüm, SALAD sürümü, embedding_dim, preprocessing config, dataset sürümü, index build hash. Index adı hash içerir. Startup'ta hash uyuşmazsa → `ModelNotReadyError`.

## Sync/Async GPU Kuyruğu (risk #28)
| Alternatif | Değerlendirme |
|---|---|
| Doğrudan blocking | Event loop kilitlenir ✗ |
| Servis başına ayrı Semaphore(1) | ✗ embed+eigen_cam eşzamanlı → 8GB OOM riski |
| **Tek global paylaşılan Semaphore(1) + run_in_executor** | ✓ Sistem genelinde tek GPU işlemi, loop duyarlı, Redis yok |
| Celery+Redis | v2; MVP overkill |

**Öneri:** `app.state` içinde **tek global `gpu_semaphore = asyncio.Semaphore(1)`**; DI ile hem EmbeddingService hem XAIService'e **aynı örnek** enjekte edilir. Her GPU çağrısı bu semaphore'u alır → herhangi bir anda yalnızca bir GPU işlemi (embed *veya* eigen_cam) çalışır; iki eşzamanlı isteğin çakışması engellenir. `run_in_executor` (tek-thread pool) ile event loop duyarlı kalır. fp16 + 322² (küçük batch ayarıyla) → 8GB VRAM içinde (risk #20/#21).

> **Kritik:** Servis başına ayrı semaphore, "tek-worker serileştirme" hedefini kırar (embed ve eigen_cam paralel çalışıp OOM). Bu nedenle tek paylaşılan örnek zorunludur.

### MİM-2a Özeti
Uçtan uca pipeline (3 Mermaid), offline index inşası, hash-tabanlı versiyonlama, Semaphore(1) GPU serileştirme.
