# GeoReasoner — Araştırma Raporu
## FAZ D: Risk, MVP ve Portföy Stratejisi

> Araştırma-only çıktı. Kod/uygulama dosyası üretilmemiştir.

---

## Bölüm 8 — Teknik Riskler ve Çözümleri (35 risk)

### Veri
| # | Risk | Çözüm |
|---|---|---|
| 1 | MSLS login/ToU arkasında | Erken indir, yerelde cache, özel depoya yedek |
| 2 | CC-BY-SA atıf + ShareAlike | Görüntü başına yazar/lisans kredisi; türev index uyumlu lisans |
| 3 | Mapillary ToU yeniden-kimliklendirme yasağı | Bulanıklaştırmayı geri alma; koruma süreci |
| 4 | GPS gürültüsü (~2m, kentsel kanyon kötü) | Etiket ±5m; km-hata raporla |
| 5 | Domain gap (dashcam vs telefon) | Telefon test seti; hafif fine-tune/augment |
| 6 | Gün/gece/mevsim varyasyonu | DINOv2 dayanıklı + augment; boşluğu belgele |
| 7 | Sınıf dengesizliği (seyrek banliyö) | Bölge-başı recall; yoğun alanı sınırla |

### Model / Retrieval
| # | Risk | Çözüm |
|---|---|---|
| 8 | Domain shift recall düşürür | DINOv2-SALAD seçildi; held-out benchmark |
| 9 | OOD sorgular | Güven/mesafe eşiği → "düşük güven" |
| 10 | SALAD 8448-dim RAM | ~33KB/vec; ≤1M yönetilebilir; çok-şehirde PCA/PQ |
| 11 | FAISS doğruluk-hız | <birkaç M Flat; hız için HNSW; agresif PQ'dan kaçın |
| 12 | Cold-start | Tam Amsterdam alt-kümesiyle başla |
| 13 | Algısal aliasing (MSLS zaafı) | Top-K + harita yayılımı döndür |
| 14 | Bakış-açısı değişimi | SALAD optimal-transport; R@1/5/10 |
| 15 | Index kayması | Model hash'iyle versiyonla |

### XAI
| # | Risk | Çözüm |
|---|---|---|
| 16 | Sınıf başlığı yok → Grad-CAM geçersiz | Eigen-CAM / attention rollout |
| 17 | Yanıltıcı ısı haritası | Insertion/deletion faithfulness doğrula |
| 18 | VLM halüsinasyonu | "Spekülatif" etiketle |
| 19 | Açıklama gecikmesi | Talep-üzerine, async, retrieval dışı |

### Donanım / Performans
| # | Risk | Çözüm |
|---|---|---|
| 20 | 8GB VRAM | ViT-B/14 @224² + fp16 + küçük batch |
| 21 | Model yükleme | Başlangıçta bir kez, sıcak tut |
| 22 | Çıkarım gecikmesi | DB embedding offline; yalnız sorgu canlı |
| 23 | FAISS RAM patlaması | Amsterdam'a sınırla; PQ |
| 24 | Toplu embedding işi | Parçala + checkpoint |

### Full-stack / Deployment
| # | Risk | Çözüm |
|---|---|---|
| 25 | Bulut GPU maliyeti | HF T4 ~$0.40/saat; geliştirme yerel |
| 26 | CORS | Origin kısıtla |
| 27 | Dosya yükleme güvenliği | MIME/boyut doğrula, EXIF sıyır, sandbox |
| 28 | Eşzamanlılık | Tek-worker GPU kuyruğu; rate-limit |
| 29 | Docker GPU passthrough | nvidia-container-toolkit; hedefte test |
| 30 | HF Spaces sınırları | Ücretsiz CPU 16GB/50GB, boşta uyur; index sığmalı |

### Değerlendirme / Ürün / Hukuki-Etik
| # | Risk | Çözüm |
|---|---|---|
| 31 | Recall@N ölçümü | MSLS R@1/5/10, GT mesafe eşiği |
| 32 | Kullanıcı fotoğrafında GT yok | Manuel etiket / EXIF-GPS alt-küme |
| 33 | Metrik seçimi | Recall + medyan km-hata birlikte |
| 34 | Scope creep / tek-şehir | Yalnız Amsterdam; sınırı belirt |
| 35 | İkili-kullanım + GDPR (Madde 9 özel kategori, DPIA) | Rıza banner, geçici işleme, depolama yok, suistimal politikası |

**Kaynaklar:** MSLS (github.com/mapillary/mapillary_sls) · CC-BY-SA 4.0 · SALAD arXiv:2311.15937 · CAM-FC arXiv:2506.01636 · FAISS guidelines · HF GPU (huggingface.co/docs/hub/spaces-gpus) · GeoShield arXiv:2508.03209 · GDPR EDPB 3/2019

---

## Bölüm 9 — MVP Yol Haritası

| Özellik | MVP (Şimdi) | Sonraya |
|---|---|---|
| Kapsam | Yalnız **Amsterdam** | Çok-şehir, global mod |
| Embedding | Önceden hesaplanmış **DINOv2-SALAD** | Fine-tune, PCA/PQ |
| Index | **FAISS Flat** (exact) | HNSW / IVFPQ |
| Giriş | Tek görüntü yükleme | Sıralı (sequence) retrieval |
| Çıktı | **Top-K Leaflet marker** | Yoğunluk/heatmap katmanı |
| XAI | Tek **Eigen-CAM** | VLM gerekçe |
| UI | **Gradio** (HF Spaces) | Next.js + React |
| Güvenlik | EXIF sıyır + boyut sınırı | Hesap, rate-limit |
| DB | SQLite | PostGIS, canlı re-index |

**İlke:** Çekirdek mantık (model+retrieval) UI'dan ayrı. Minimal-ama-etkileyici: tek-şehir, exact retrieval, bir XAI overlay, canlı demo.

---

## Bölüm 10 — GitHub Portföy Stratejisi

### Must-have
1. **Hero GIF** — ısı haritası + harita akışı (görünürlüğü en çok artıran).
2. **Canlı demo (HF Spaces)** — "tıkla-dene"; işe alımcı etkisi en yüksek tek faktör.
3. **Mermaid mimari diyagramı** (GitHub inline render).
4. **Tek-komutluk Docker quickstart**.
5. **Benchmark tablosu** — R@1/R@5/R@10 + medyan lokalizasyon hatası; NetVLAD baseline.
6. **Pinned bağımlılık + seed**; ağırlıklar HF Hub/Releases.
7. **MIT lisansı**, badge'ler, FastAPI `/docs` vurgusu.

### Nice-to-have
- GitHub Actions CI, pytest, Ruff, mypy, coverage.
- Eşlik eden blog/write-up.
- Görünürlük: Show HN + r/MachineLearning + LinkedIn + papers-with-code, koordineli.

**Kaynaklar:** papers-with-code VPR · HF Spaces · Mermaid (mermaid.js.org)

---

### Faz D Özeti
- 35 teknik risk + çözüm (8 kategori).
- MVP: Amsterdam + SALAD + FAISS Flat + Eigen-CAM + Gradio/HF Spaces.
- Portföy: hero GIF + canlı demo + benchmark + Mermaid = must-have.
