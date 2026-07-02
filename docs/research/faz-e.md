# GeoReasoner — Araştırma Raporu
## FAZ E: Sentez, Karar ve Sonraki Aşama

> Araştırma-only çıktı. Kod/uygulama dosyası üretilmemiştir.

> **⚠️ ERRATA (2026-07):** Aşağıdaki "Hedef şehir = Amsterdam" kararı faktüel hataya dayanıyordu. Geçerli hedef: resmi MSLS-val = **cph+sf** (Kopenhag + San Francisco). Ayrıntı için `faz-b.md` errata'sına bakınız.

---

## 1) Proje İsim Önerileri
| İsim | Çağrışım | Değerlendirme |
|---|---|---|
| GeoReasoner (mevcut) | Coğrafi akıl yürütme; XAI | ⚠️ Aynı adda makale var (arXiv:2406.18572) → karışıklık |
| **GeoLens** | Coğrafi mercek; bakış/odak | ✓ Kısa, akılda kalıcı, XAI ile uyumlu, çakışma düşük |
| StreetSense | Sokak sezgisi | ✓ Güçlü, ürün çakışması olabilir |
| PlaceReasoner | Yer + akıl yürütme | Açık ama uzun |
| GeoXplain | Geo + Explain | XAI'yi öne çıkarır, yapay |
| WhereByVision | Görüntüyle nerede | Açıklayıcı, demo-dostu |

**Öneri: GeoLens** — mevcut "GeoReasoner" makale çakışmasını önler; mercek/odak metaforu XAI hikayesini taşır. (Mevcut isimde ısrar serbest; karar kullanıcının.)

---

## 2) Açık Sorular (Durum)
| Soru | Durum / Öneri |
|---|---|
| Hedef şehir? | ÇÖZÜLDÜ → **Amsterdam** (MSLS val, açık GT, CC-BY-SA) |
| Gerçek-zaman mı, demo mu? | Öneri: **demo**. DB embedding offline; yalnız sorgu canlı |
| Lokal mi bulut mu? | Öneri: **yerel geliştirme + HF Spaces demo** |
| Metrikler? | Öneri: **Recall@1/5/10 + medyan km-hata** |
| İsim? | GeoLens vs GeoReasoner — kullanıcı kararı |
| VLM açıklama MVP'de mi? | Öneri: **hayır, v2**; MVP'de Eigen-CAM yeter |

> Kalan sorular tercih niteliğinde; teknik blokör değil.

---

## 3) Nihai Karar Matrisi
| Boyut | Karar | Güven |
|---|---|---|
| Problem tipi | Retrieval-first | Yüksek |
| Veri | MSLS Amsterdam | Yüksek |
| Backbone+agg | DINOv2 ViT-B + SALAD (yedek EigenPlaces 512) | Yüksek |
| Index | Flat (MVP) → HNSW/PCA/IVFPQ | Yüksek |
| XAI | Eigen-CAM/rollout + (v2) VLM + referans/harita | Orta-Yüksek |
| Stack | FastAPI + Gradio + Leaflet + SQLite + Docker → HF Spaces | Yüksek |
| Fizibilite (8GB/64GB) | MVP fizibıl; çok-şehirde boyut indirgeme şart | Yüksek |

---

## 4) "Başlamanı öneriyor muyum?" — EVET
- **Fizibıl:** retrieval-first + pretrained + offline embedding → 8GB VRAM yeter; 64GB tek-şehir index taşır.
- **Farklılaşmış:** açık kaynakta retrieval + nesne-düzeyi XAI + harita + full-stack birleşimi yok.
- **Portföy değeri yüksek:** canlı demo + GIF + benchmark + tam-yığın/CV/XAI hikayesi.
- **Riskler yönetilebilir:** 35 riskin somut çözümü var; blokör yok (kritik: domain gap, GDPR).
- **Uyarı:** kapsamı tek-şehir MVP'de tut; scope creep en büyük tehdit.

---

## 5) Mimari Tasarım Aşamasına Hazırlık
**Hazır.** Net: problem tipi, veri, model, index, XAI, stack, deployment, riskler, MVP, metrikler.
Sonraki aşamada (ayrı görev, açık onayla) üretilebilir: Mermaid mimari diyagramı, veri akışı, modül/dizin planı, API sözleşmesi, MSLS Amsterdam değerlendirme protokolü.
> Kod yazımı ayrı ve açık onaya tabidir.

---

### Faz E Özeti
- İsim: GeoLens önerildi (çakışma nedeniyle).
- Açık sorular çözüldü/önerildi; teknik blokör yok.
- Nihai karar: **projeye başla** — fizibıl, farklılaşmış, yüksek portföy değeri.
- Mimari tasarım aşamasına hazırız.
