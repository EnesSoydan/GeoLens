# Faz 2 — Ölçekleme Mimarisi Karar Matrisi (v2)

> **Görev tipi:** Araştırma + planlama (kod değil).
> **Amaç:** Faz 1'de seçilen kaynaktan (OSV5M birincil) küresel/kıtasal ölçeğe giderken
> hangi indeks/sıkıştırma mimarisini seçeceğimize karar vermek. Üç kademe tanımlanır
> (v2a / v2b / v3) ve her biri için **RAM/disk bütçesi** çıkarılır.
> **Donanım (değişmedi):** RTX 4060 8GB VRAM, 64GB RAM, Ryzen 7 7435HS.
> **Kaynak-doğruluk:** v1 kararları (retrieval-first, SALAD 8448-dim, FAISS) korunur;
> bu belge yalnız *ölçek* ekseninde ayrışır.

---

## 0. Problem: doğrusal ham ölçekleme duvarı

v1 indeksi Flat (IndexFlatIP) + ham float32. Bellek doğrusal:

```
N × 8448 × 4 bayt (float32)
  18.9k  →   0.60 GB   (v1, cph+sf — sığar)
   100k  →   3.38 GB
   500k  →  16.9 GB    (64GB'a sığar ama tek başına değil — model+OS+heatmap ile sıkışık)
     1M  →  33.8 GB    (üst pratik sınır, ham Flat)
     5M  → 169 GB      (v1 uyarısı — İMKÂNSIZ)
   5.1M  → 172 GB      (tam OSV5M — İMKÂNSIZ)
```

**İki kaldıraç var:** (1) **N'i düşür** (alt-örnekleme), (2) **vektör başına bayt'ı
düşür** (boyut indirgeme PCA + skalar/PQ nicemleme). Faz 2 bu ikisini kademelere böler;
doğruluk maliyeti Faz 3'ün konusu (burada yalnız bütçeyi hesaplarız).

---

## 1. Üç kademe

### v2a — Çok-şehir / bölgesel (İLK GERÇEKÇİ HEDEF)
- **Kapsam:** OSV5M'den 5-10 şehir veya bir ülke alt-kümesi, **~100k–500k** görüntü.
- **İndeks:** PCA ile **8448 → 1024** indir, sonra **HNSW** (graf-tabanlı, hızlı, RAM-içi).
- **Neden HNSW (Flat değil):** 500k'da Flat exact-search her sorguda 500k×1024 çarpım →
  CPU'da yavaş. HNSW yaklaşık ama ~ms; RTX4060 tek-kullanıcı demo için yeterli. Bellek
  Flat'e yakın (graf ~%20-40 ek).
- **Neden IVFPQ değil:** 500k'da PQ nicemleme doğruluk kaybı gereksiz; PCA-1024 float32
  zaten 64GB'a sığıyor. IVFPQ v2b'de devreye girer.

### v2b — Kıta / ülke ölçeği
- **Kapsam:** **~1M–5M** görüntü (OSV5M'in büyük kısmı veya tamamı).
- **İndeks:** PCA → 512 (veya 256) + **IVFPQ** (Inverted File + Product Quantization).
  Coarse quantizer (IVF, nlist hücreleri) + her vektör PQ-kodlu (örn. m=64 alt-vektör,
  8 bit → 64 bayt/vektör).
- **Neden IVFPQ:** ham 5.1M×8448 = 172GB'ı **64 bayt/vektör**'e sıkıştırır → 5.1M×64B =
  **0.33 GB** kod + IVF/PCA ek yapıları. Nicemleme kaybı var (Faz 3), nprobe ile
  recall/hız takası ayarlanır.

### v3 — Kaba küresel (coarse global)
- **Kapsam:** Tüm dünya, retrieval + kaba sınıflandırma karışımı.
- **İki alternatif (Faz 3/ileride karar):**
  1. **GeoCLIP-tarzı** — görüntüyü doğrudan GPS-gömülü uzaya eşleyen ayrı model (retrieval
     değil regresyon-benzeri). v1'de saf regresyon elenmişti; küresel kaba tahmin için
     yeniden değerlendirilir.
  2. **Hiyerarşik retrieval** — önce kaba (ülke/bölge, düşük-dim indeks), sonra o bölge
     içinde ince (yüksek-dim yerel indeks). İki-aşama → RAM'e tümünü birden yüklemeden.
- v3 MVP değil; yol haritasının ufku (Faz sonu roadmap).

---

## 2. RAM / disk bütçesi tablosu

Model yükü sabit: DINOv2 ViT-B + SALAD head ≈ **1.4 GB disk (ağırlık)**, ~**2.1 GB RAM
peak** (v1 ölçümü, CPU inference + 1 heatmap). Aşağıdaki indeks bütçeleri **buna ek**.

| Kademe | N | Boyut | Kodlama | Bayt/vektör | İndeks RAM | İndeks disk | 64GB'a? |
|---|---|---|---|---|---|---|---|
| v1 (mevcut) | 18.9k | 8448 | Flat f32 | 33.8 KB | 0.60 GB | 0.64 GB | ✅ bol |
| v2a-alt | 100k | 1024 | Flat f32 | 4.0 KB | 0.38 GB | 0.41 GB | ✅ bol |
| v2a | 500k | 1024 | HNSW f32 | ~5.5 KB* | ~2.6 GB | ~2.0 GB | ✅ rahat |
| v2b-orta | 1M | 512 | IVFPQ m64 | 64 B | ~0.13 GB | ~0.15 GB | ✅ bol |
| v2b | 5.1M | 512 | IVFPQ m64 | 64 B | ~0.40 GB | ~0.45 GB | ✅ bol |
| — | 5.1M | 8448 | Flat f32 (referans) | 33.8 KB | **172 GB** | 172 GB | ❌ İMKÂNSIZ |

\* HNSW graf ek maliyeti (M=16-32 komşu) ham float32 üstüne ~%20-40; PCA-1024 taban
4.0KB → ~5.5KB efektif.

**Kritik gözlem:** IVFPQ ile *bellek artık darboğaz değil* (5.1M → 0.4GB!). Darboğaz
kayar: (a) **disk** — 5.1M görüntüyü embed etmek için önce indirmek (259GB, HF free-tier
50GB'ı aşar → ev sunucusu/harici disk, Faz 4), (b) **embed süresi** — 5.1M × (GPU ~20-25
img/s) ≈ **57-71 saat** tek RTX4060'ta (Faz 4 kapasite planı), (c) **doğruluk** — PQ+PCA
recall kaybı (Faz 3).

---

## 3. Karar (Faz 2)

1. **v2a = ilk hedef, ŞİMDİ inşa edilebilir.** OSV5M'den ~100-500k alt-örnek +
   **PCA→1024 + HNSW**. RAM ~2.6GB (model dahil ~4.7GB) → 64GB'a çok rahat, disk ~2GB.
   Mevcut v1 pipeline'a en yakın (aynı SALAD, yalnız PCA katmanı + HNSW indeks eklenir).
2. **v2b = kıta ölçeği, IVFPQ ile.** Bellek sorunu çözülür (5.1M→0.4GB) ama disk+embed-süresi
   +doğruluk yeni kısıtlar → Faz 4 (kapasite) ve Faz 3 (doğruluk) bunları çözmeli.
3. **v3 = ufuk.** GeoCLIP-tarzı vs hiyerarşik retrieval kararı ertelenir (roadmap sonu).

**Öneri sırası:** v2a'yı gerçekten kur ve ölç (v1 → v2a en küçük sağlam adım), ardından
Faz 3 doğruluk maliyetiyle v2b'ye karar ver. v3'e sıçrama YAPMA (aşırı-mühendislik riski).

### Boyut-indirgeme uyarısı (Faz 3 önizleme, kararı etkiler)
v1 hafızasındaki "PCA→1024 zorunlu" ifadesi nüanslı: **SALAD üzerinde WPCA recall'ı
tutarlı biçimde DÜŞÜRÜYOR** (VLAD-BuFF, arXiv:2409.19293). Yani PCA "bedava" değil.
Bu, v2a'da PCA-1024'ün gerçek recall kaybını **ölçmeyi** (kendi val'imizde, cph+sf ya da
OSV5M-alt val) zorunlu kılar — varsayım değil, ölçüm. Faz 3 bunu ele alacak.

---

## 4. Faz 2 özeti (onay noktası)

- **v2a (100-500k, PCA→1024 + HNSW, ~2.6GB RAM):** ilk gerçekçi hedef, şimdi inşa edilebilir.
- **v2b (1-5M, PCA→512 + IVFPQ m64, ~0.4GB RAM):** bellek çözülür; darboğaz **disk (259GB)
  + embed süresi (~60 saat) + doğruluk (PQ/PCA kaybı)**'na kayar (Faz 3/4).
- **v3 (küresel):** GeoCLIP-tarzı vs hiyerarşik — ertelendi (roadmap ufku).
- **En büyük içgörü:** IVFPQ ile RAM darboğazı kalkar → v2 gerçek engelleri **disk kapasitesi,
  embed süresi, boyut-indirgeme doğruluk kaybı**. Sırasıyla Faz 4, Faz 4, Faz 3.

**Sonraki: Faz 3 — boyut indirgeme doğruluk maliyeti** (SALAD 8448 → PCA 1024/512, IVFPQ;
WPCA recall etkisi; kaynaklı sayılar). Onayınızı bekliyorum.

---

### Kaynaklar
- FAISS wiki — IVFPQ/HNSW bellek ve nprobe/nlist parametreleri.
- VLAD-BuFF, arXiv:2409.19293 (SALAD üzerinde WPCA recall etkisi).
- OSV5M, arXiv:2404.18873 (5.1M / 259GB — Faz 1).
- v1 ölçümleri (MEMORY): model ~2.1GB RAM peak, GPU ~20-25 img/s, cph+sf 0.60GB Flat.
