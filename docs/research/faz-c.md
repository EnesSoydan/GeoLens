# GeoReasoner — Araştırma Raporu
## FAZ C: Açıklanabilir YZ ve Teknoloji Yığını

> Araştırma-only çıktı. Kod/uygulama dosyası üretilmemiştir.

---

## Bölüm 6 — XAI Yöntemleri (Retrieval'a Uyarlama)

> **Temel zorluk:** Retrieval'da sınıflandırma başlığı (class logit) yok. Standart Grad-CAM logit gerektirir. Çözüm: hedefi **sorgu embedding'i ile eşleşen referans embedding'i arasındaki kosinüs benzerliği** olarak tanımla, bu skaleri geri-yay (`SimilarityToConceptTarget`).

| Yöntem | Ne açıklar | Retrieval'a uygunluk | Maliyet | Kütüphane |
|---|---|---|---|---|
| **Grad-CAM** (SimilarityToConceptTarget) | Eşleşmeyi süren bölgeler | Uyarlanır (benzerlik skaleri) | ~0.03-0.4s | pytorch-grad-cam |
| **Eigen-CAM** | Sınıf-agnostik bölge | **İdeal** (gradyan/logit gerekmez) | Yavaş ama embedding-dostu | pytorch-grad-cam |
| Grad-CAM++ / HiRes / XGrad | İyileştirilmiş ısı haritası | Uyarlanır | ~tek geri-yay | pytorch-grad-cam |
| Score-CAM | Bölge (gradyansız) | Uyar | ~3-4s | pytorch-grad-cam |
| **Attention Rollout** | ViT/DINOv2 dikkat | DINOv2 için temiz nesne-parça | Ucuz | pytorch-grad-cam |
| **En yakın referans görseli** | Kanıt | **Doğal açıklama** (görsel + GPS) | Bedava | FAISS sonucu |
| **VLM açıklama** (GeoReasoner tarzı) | İnsan-okunur coğrafi gerekçe | Mimari/tabela/bitki/yol işareti | Orta | HF VLM |
| Integrated Gradients / SmoothGrad | Piksel atıfı | Benzerlik skalerine uygulanır | Orta | Captum |

**Geolocation'a özel XAI:** Object-Level Explanations (arXiv:2605.00912) — atıf haritasını nesne ipuçlarına böler; Combi-CAM (arXiv:2603.24117) çok-katman füzyon.

### Önerilen XAI Yığını (RTX 4060)
1. **Bölge atfı (nereye baktı):** DINOv2-ViT için **Eigen-CAM / attention rollout** (logit gerekmez) → sorgu + eşleşen referans üzerine ısı haritası overlay.
2. **İnsan-okunur gerekçe (neden):** Hafif **VLM** ile coğrafi ipucu anlatımı (mimari, tabela dili, bitki örtüsü, yol işaretleri) — opsiyonel "wow" katmanı.
3. **Güven/kanıt:** En yakın referans görseli + GPS pin'i haritada.

> 8GB VRAM içinde ucuz bölge atfı + semantik gerekçe + retrieval kanıtını birleştirir. SALAD/DINOv2 sırtı kurulu olduğundan Eigen-CAM/rollout ek model istemez; VLM MVP'de kapatılabilir.

**Kaynaklar:** pytorch-grad-cam (github.com/jacobgil/pytorch-grad-cam) · Embedding Grad-CAM arXiv:2001.06538 · GeoReasoner VLM arXiv:2406.18572 · Object-level XAI arXiv:2605.00912 · Captum (captum.ai)

---

## Bölüm 7 — Teknoloji Yığını

| Katman | MVP (şimdi) | Ölçeklenme (sonra) | Gerekçe |
|---|---|---|---|
| **Backend/API** | **FastAPI** | FastAPI | ASGI/async; Pydantic; oto OpenAPI; ML standardı. Flask senkron darboğaz; Django aşırı. |
| **Frontend** | **Gradio** | **Next.js + React** | Gradio hızlı/paylaşılır demo; Next.js profesyonel portföy vitrini. |
| **Harita** | **Leaflet** (+leaflet.heat) | deck.gl | BSD, ücretsiz, API key yok. Mapbox non-OSS+ücretli; Google billing. |
| **Veritabanı** | **SQLite** | **PostgreSQL + PostGIS** | SQLite sıfır-config; PostGIS yalnızca mekânsal sorgu gerekince. |
| **Cache/Queue** | Yok | Redis + Celery | MVP'de overkill; event-loop bloklanırsa. |
| **ML çerçevesi** | **PyTorch** | PyTorch | Tüm VPR repoları + XAI araçları PyTorch-native. |
| **Vektör arama** | **FAISS in-process** | Qdrant | Statik galeri için FAISS ideal; Qdrant CRUD/filtre/ölçek. |
| **Deployment** | **Docker + compose** (yerel) | **HF Spaces** (bulut GPU) | Render/Railway GPU yok. HF Spaces ücretsiz CPU/ZeroGPU, T4 ~$0.40/saat. |

### Önerilen MVP Yığını
```
FastAPI + Gradio + Leaflet + SQLite + FAISS (in-process) + PyTorch + Docker
→ deploy: yerel (RTX 4060) veya HF Spaces
```
**Ölçeklenme:** Next.js + PostGIS + Qdrant + Redis/Celery + deck.gl.

> Önce Gradio demo → sonra Next.js. SQLite→PostGIS, FAISS→Qdrant geçişleri yalnızca gerçek ölçek ihtiyacında. MVP'yi gereksiz altyapıyla (Redis, vektör DB, PostGIS) şişirme.

**Kaynaklar:** FastAPI · Leaflet (leafletjs.com) · HF Spaces (huggingface.co/docs/hub/spaces-gpus) · VPR-methods-evaluation (github.com/gmberton) · Qdrant (qdrant.tech)

---

### Faz C Özeti
- **XAI:** 3 katman — Eigen-CAM/rollout ısı haritası + VLM gerekçe (ops.) + en yakın referans/harita pin. Retrieval'da logit yok → benzerlik-skaleri/gradyansız tercih.
- **Tech stack:** MVP = FastAPI + Gradio + Leaflet + SQLite + FAISS + PyTorch + Docker; ölçeklenme = Next.js + PostGIS + Qdrant + Redis.
