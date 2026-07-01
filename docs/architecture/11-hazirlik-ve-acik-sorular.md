# MİM-4d: Development Fazına Hazırlık + Açık Sorular

## "Development fazına (Prompt 3) geçmeye hazır mıyız?" — EVET

**Teknik gerekçe:**
- **Tüm 10 başlık** mimari olarak tanımlandı: repo yapısı, katmanlı backend, AI pipeline, API sözleşmesi, frontend, veri modeli, deployment, değerlendirme, sprint, kod standartları.
- **Tutarlılık doğrulandı:** üç kritik tutarsızlık giderildi — (1) tek global paylaşılan GPU semaphore (OOM önlemi), (2) düşük güvende heatmap üretilmez (pipeline↔API hizası), (3) `image_role` ile query görüntüleri index dışında (Recall@N data leakage önlemi).
- **Her karar gerekçeli**, ≥2-3 alternatifle karşılaştırıldı; donanım (8GB VRAM / 64GB RAM) kısıtına uygun.
- **Sprint yol haritası + kritik yol** net; S0 hemen başlatılabilir.
- **Kalan belirsizlikler blokör değil** (aşağıda); implementasyon detayı seviyesinde, ilk sprint'lerde kalibre edilir.

> **Uyarı:** Development fazı ayrı ve açık onaya tabidir. Kapsam disiplini korunmalı: MVP yalnız Amsterdam; çok-şehir/real-time/VLM/Next.js → v2.

## Açık Sorular
1. **MSLS GT pozitif eşiği:** ÇÖZÜLDÜ → resmi mapillary_sls eşiği **25 m VE ≤40° bakış-açısı** (birlikte); resmi MSLS-val protokolü (740 sorgu, Amsterdam alt-kümesi). Kalan: Amsterdam alt-kümesinin kesin database/query sayıları resmi split dosyalarından okunacak.
2. **Yükleme boyut limiti:** `/predict` için maks dosya boyutu (örn. 10 MB) — kesinleştirilecek.
3. **Güven eşiği (`is_confident`):** ampirik; S5 değerlendirmede kalibre edilecek (şimdilik placeholder).
4. **HF Spaces GPU katmanı:** ZeroGPU vs T4 — deploy anında maliyet/kota kararı.
5. **Referans görsel küçük resimleri:** popup'ta base64 (MVP tutarlı) vs statik — MVP base64 önerilir.
6. **v2 kapsamı:** Next.js + çok-şehir + VLM açıklama — MVP dışı, ayrı planlanacak.

## Sonraki Adım
Development fazı (Prompt 3): S0 sprint'inden başlayarak gerçek kod. Bu mimari dokümanlar (`docs/architecture/`) referans kaynak olur.
