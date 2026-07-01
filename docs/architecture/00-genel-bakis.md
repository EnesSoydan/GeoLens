# GeoLens — Mimari Tasarım: Genel Bakış

> **Kapsam:** Bu klasör (`docs/architecture/`) yalnızca **mimari plan/dokümantasyon** içerir.
> Uygulama kodu (çalışan .py/.js, gerçek Dockerfile, gerçek requirements.txt) BURADA YOKTUR — Development fazına aittir.
> Tasarım kararlarının gerekçeleri için `docs/research/faz-a.md … faz-e.md` referans alınır.

## Proje Özeti (Araştırma Fazı Kararları)
- **Ad:** GeoLens (GeoReasoner akademik makaleyle çakıştığından değişti)
- **Problem tipi:** Retrieval-first (görüntü → embedding → FAISS kNN → konum)
- **Veri:** Resmi MSLS **validation** split = **cph + sf** (Kopenhag + San Francisco) — **resmi mapillary_sls MSLS-val protokolü** (~18.9k database + 740 sorgu). MVP **tam resmi val**'i kullanır (alt-küme yok). CC-BY-SA, açık GT. (DÜZELTME 2026-07: önceki "Amsterdam+Manila = val" faktüel hataydı; `default_cities['val']==['cph','sf']`, Amsterdam TRAIN şehri. Sayılar doğruydu, şehir adları yanlıştı. bkz. 08-degerlendirme.md)
- **Model:** DINOv2 ViT-B/14 + SALAD agregasyon (8448-dim)
- **Index:** FAISS Flat (~0.64GB; resmi val database ~18.9k vektör × 33KB)
- **XAI:** Eigen-CAM / attention rollout (gradyansız); VLM → v2
- **Stack:** FastAPI + Gradio (MVP) → Next.js (v2) + Leaflet + SQLite + PyTorch + Docker + HF Spaces
- **Donanım:** RTX 4060 8GB VRAM, 64GB RAM
- **Kapsam disiplini:** MVP yalnız resmi MSLS-val (cph+sf); ek şehir/real-time/VLM → v2
- **Değerlendirme:** Recall@1/5/10 + medyan km-hata (resmi MSLS-val cph+sf)

## Mimari Tasarımın Alt-Faz Haritası
10 başlık, mantıksal bağımlılığa göre 4 alt-faza gruplandı. Her alt-faz: analiz → alternatif karşılaştırma → gerekçeli karar → dosya + özet.

| Alt-faz | Başlıklar | Doküman(lar) | Durum |
|---|---|---|---|
| **MİM-0** | Genel bakış | `00-genel-bakis.md` | ⏳ Bu dosya |
| **MİM-1** | 1) Repo yapısı, 2) Backend, 10) Kod standartları | `01-repo-ve-backend.md`, `10-kod-standartlari.md` | Sırada |
| **MİM-2** | 3) AI Pipeline, 4) API sözleşmesi | `03-ai-pipeline.md`, `04-api-sozlesmesi.md` | Bekliyor |
| **MİM-3** | 5) Frontend, 6) Veri modeli/DB | `05-frontend.md`, `06-veri-modeli.md` | Bekliyor |
| **MİM-4** | 7) Deployment, 8) Değerlendirme, 9) Sprint | `07-deployment.md`, `08-degerlendirme.md`, `09-sprint-plani.md` | Bekliyor |

## Neden Bu Gruplama?
- **MİM-1 önce:** Repo iskeleti + backend katmanları + kod standartları, diğer her şeyin üzerine oturacağı temeldir.
- **MİM-2 sonra:** AI pipeline ve API, backend iskeleti netleşince anlamlı (endpoint'ler servis katmanına bağlı).
- **MİM-3:** Frontend ve veri modeli, API sözleşmesi belli olunca tasarlanır (tüketici + kalıcılık).
- **MİM-4:** Deployment/değerlendirme/sprint, tüm bileşenler tanımlıyken en sona.

## Çalışma Prensipleri
- Her alt-faz sonunda onay noktası; onaysız sonrakine geçilmez.
- Her önemli kararda ≥2-3 alternatif + eleme gerekçesi.
- Varsayım yok; belirsizlik → "Açık Sorular".
- Tüm çıktı Türkçe (teknik isimler İngilizce).
- **Kod yok** — yalnız plan/diyagram/tablo/sözleşme.

## Çıkış Kapısı
MİM-4 sonunda: "Development fazına (Prompt 3) geçmeye hazır mıyız?" sorusu teknik gerekçeyle yanıtlanır.
