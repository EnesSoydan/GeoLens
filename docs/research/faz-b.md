# GeoReasoner — Araştırma Raporu
## FAZ B: Veri Setleri ve Model Mimarisi

> Araştırma-only çıktı. Kod/uygulama dosyası üretilmemiştir.
> Not: Bu dosya, RAM bellek-bütçesi düzeltmesi ve nihai şehir kararı (Amsterdam) dahil güncel halidir.

---

## Bölüm 3 — Veri Setleri İncelemesi

> **Kritik kıstas — lisans.** Google Street View (GSV) türevli setler (Pittsburgh, Tokyo 24/7, SF-XL, GSV-Cities) GSV ToS gereği toplu depolama/dağıtımı yasakladığından açık repoda riskli.

| Veri Seti | Görüntü | GPS | Lisans | İndirme | MVP |
|---|---|---|---|---|---|
| **MSLS** | 1.6M, çok şehir, gün/gece/mevsim | lat/lon + UTM | İmaj **CC-BY-SA**, toolkit MIT | Login gerekli | **En iyi yasal MVP** |
| KartaView | Milyonlar, global | var | **CC-BY-SA-4.0** | Açık API | Ücretsiz GSV alternatifi |
| OSV5M (HF) | 5M global | lat/lon | **CC-BY-SA** | load_dataset | Global mod, kolay |
| Google Landmarks v2 | 5M | landmark düzeyi | **CC-0/CC-BY** | Kayıtsız | Sokak değil, landmark |
| Pittsburgh 30k/250k | 30k/250k | var | İstek üzerine (gated), GSV | E-posta | Benchmark, kapalı |
| Tokyo 24/7 | ~76k | var | İstek üzerine, GSV | Talep | Sadece test, risk |
| SF-XL | ~41M (5GB tiny) | var | Form, GSV | Form | tiny laptop-dostu, risk |
| BDD100K | 100k video | GPS+IMU | İmaj non-commercial | Kayıt | Sürüş, lisans kısıtı |
| Nordland | 4 mevsim tren | UTM | Araştırma | rsync | Sadece test |

### Hedef Şehir Kararı: **Amsterdam**
**MSLS split yapısı (kritik):** Amsterdam + Manila = **doğrulama** şehirleri (GT herkese açık); Kopenhag + San Francisco = **test** şehirleri (GT gizli).

**Gerekçe:** Portföyde **Recall@N'i kendiniz hesaplamak** açık GT gerektirir → yalnızca val şehirlerinde mümkün. Kopenhag/SF elenir (self-evaluation yapılamaz). İki val şehrinden Amsterdam, Manila'ya kıyasla daha zengin gün/gece+mevsim çeşitliliği, daha tanınır ve demoda güçlü → MVP için en sağlam temel. Lisans CC-BY-SA (yasal, dağıtılabilir).

**Değerlendirme protokolü (düzeltme):** **Resmi mapillary_sls MSLS-val protokolü** kullanılır — Amsterdam+Manila toplam ~18.9k database + **740 sorgu**. MVP Amsterdam alt-kümesini kullanır. (Önceki '~11k sorgu' gayri-resmi VPR-datasets-downloader versiyonuydu; literatürle NetVLAD/SALAD/BoQ karşılaştırması için resmi protokol şart — bkz. `docs/architecture/08-degerlendirme.md`.)

### Genişleme Yol Haritası
1. **MVP (v1):** Amsterdam — referans DB + retrieval + XAI + harita.
2. **v2:** Aynı CC-BY-SA havuzunda 2-3 MSLS şehri ekle (sadece index'e ekleme).
3. **v3:** KartaView (CC-BY-SA, açık API) ile yeni şehirler.
4. **Global mod (ops.):** OSV5M ile kaba global tahmin (GeoCLIP tarzı).

---

## Bölüm 5 — Model Mimarisi ve Embedding Sistemi

### Sırt (Backbone)
| Model | Boyut | Param | 8GB VRAM |
|---|---|---|---|
| **DINOv2 ViT-B/14** | 768 | 86M | Rahat ✓ |
| DINOv2 ViT-S/14 | 384 | 21M | Çok rahat ✓ |
| DINOv2 ViT-L/14 | 1024 | 303M | Sınırda |
| CLIP ViT-B/32 | 512 | ~150M | ~1-2GB ✓ |
| SigLIP2 ViT-B/16 | 768 | 86M | ✓ (CLIP'i geçer) |
| ResNet50 | 2048 | 25M | Çok rahat (ViT'ten zayıf) |
| StreetCLIP/GeoCLIP | 768 | CLIP tabanlı | Global mod |

### Agregasyon
| Yöntem | Boyut | Eğitim | Not |
|---|---|---|---|
| **SALAD** (DINOv2) | 8448 | Pretrained | SOTA (Pitts250k R@1 %95.1) |
| BoQ | 12288 | Pretrained | SALAD'a yakın/üstü |
| AnyLoc | 49152 | Eğitimsiz | Maks. dayanıklılık |
| EigenPlaces | 512 | Pretrained | <8GB, kompakt |
| CosPlace | 512 | Pretrained | <4GB, en hafif |

### FAISS Bellek Bütçesi (RAM, CPU) — DÜZELTİLMİŞ
| Boyut | bayt/vec | 100k | 1M | 5M | 10M |
|---|---|---|---|---|---|
| 512 | 2 KB | 0.2 GB | 2 GB | 10 GB | 20 GB |
| 1024 (PCA-SALAD) | 4 KB | 0.4 GB | 4 GB | 20 GB | 40 GB |
| **8448 (SALAD ham)** | 33 KB | 3.3 GB | ~34 GB | 169 GB ✗ | 338 GB ✗ |
| 8448 + IVFPQ | ~64 B | 6 MB | 64 MB | 320 MB | 640 MB |

> Faz A'daki "1M ≈ 2GB" yalnızca 512-dim için geçerlidir. 8448-dim ~17x büyüktür.

### Önerilen Yığın (Katmanlı, Güncel)
- **MVP (Amsterdam, resmi val database alt-kümesi):** `DINOv2 ViT-B + SALAD (8448) + FAISS Flat` — ~0.64GB üst sınır (Amsterdam alt-kümesi daha küçük), en yüksek kalite, rahat sığar.
- **v2 çok-şehir (~1M):** `SALAD → PCA-whitening 1024-dim + HNSW` (1M ≈ 4GB), *veya* yerel 512-dim **EigenPlaces**.
- **v3 büyük ölçek (>5M):** `IVFPQ` (1M ≈ 64MB, kayıplı).
- **Hızlı başlangıç yedeği:** `EigenPlaces 512-dim + Flat` → en düşük VRAM/index, hızlı iterasyon.

**FAISS index tipleri:** Flat (exact, <100k), HNSW M=32 (hızlı+doğru, 100k-10M), IVF (>10M), IVFPQ (maks. sıkıştırma, kayıplı).

**Kaynaklar:** MSLS (mapillary.com/dataset/places) · DINOv2 arXiv:2304.07193 · SALAD arXiv:2311.15937 · BoQ arXiv:2405.07364 · AnyLoc arXiv:2308.00688 · CosPlace arXiv:2204.02287 · EigenPlaces arXiv:2308.10832 · FAISS wiki (facebookresearch/faiss)

---

### Faz B Özeti
- **Veri:** MVP = MSLS **Amsterdam** (açık GT → Recall@N self-eval; CC-BY-SA).
- **Model:** DINOv2 ViT-B + SALAD + FAISS; ham 8448 yalnızca ≤1M'e kadar → çok-şehir için PCA-1024/IVFPQ zorunlu.
- **Hızlı başlangıç:** EigenPlaces 512-dim + Flat.
