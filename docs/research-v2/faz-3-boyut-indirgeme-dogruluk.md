# Faz 3 — Boyut İndirgeme Doğruluk Maliyeti (v2)

> **Görev tipi:** Araştırma + planlama (kod değil).
> **Amaç:** Faz 2'de seçilen sıkıştırma araçlarının (PCA, IVFPQ) **recall'a maliyetini**
> kaynaklı sayılarla çıkarmak; "PCA/PQ bedava değil" varsayımını doğrulamak veya çürütmek;
> v2a/v2b için somut kodlama parametreleri önermek.
> **Kaynak-doğruluk:** v1 SALAD 8448-dim korunur; bu belge yalnız *sıkıştırma* eksenini işler.
> **İlke:** Sayılar kaynaklı; kaynaksız recall rakamı yazılmaz. Kendi val'imizde (cph+sf ya
> da OSV5M-alt) **ölçüm** gerektiren yerler açıkça "ölç, varsayma" diye işaretlenir.

---

## 0. İki bağımsız sıkıştırma kaldıracı

Faz 2'de indeks bütçesini iki kaldıraç düşürüyordu. Doğruluk maliyetleri **bağımsız** ve
**birikimli** (ikisi birden uygulanırsa kayıplar toplanır):

1. **Boyut indirgeme (PCA/WPCA):** 8448 → 1024/512/256. Vektörü kısaltır (bayt/vektör ↓).
2. **Nicemleme (PQ / IVFPQ):** float32 → PQ-kod (örn. 64 bayt). Her boyutu kaba kuantize eder.

Aşağıda her birinin recall etkisi ayrı ele alınır, sonra birleşik v2a/v2b önerisi verilir.

---

## 1. Boyut indirgeme (PCA) — SALAD'a özgü nüans

**Genel VPR sezgisi yanıltıcı.** Kaynaklı bulgular:

- **DÜZELTME (kullanıcı yakaladı) — "8448→8192 hafif budama kayıpsız" İDDİASI KAYNAKSIZDI,
  KALDIRILDI.** İlk taslakta bir web-özeti "SALAD post-hoc PCA→8192 anlamlı kayıpsız" diye
  aktarmıştı; **SALAD makalesinde (arXiv:2311.15937) böyle bir PCA iddiası YOK.** Makalede
  8192, VLAD kısmının *tasarım* boyutudur (64 küme × 128), 256 ise ayrı **global token**'dır:
  `8448 = (64×128) + 256`. Yani "8192'ye budamak" = global token'ı yapısal olarak ATMAK
  demektir — ve makalenin kendi bileşen ablasyonu (**Table 5, MSLS-val**) bunun kayıpsız
  OLMADIĞINI gösteriyor:
  - Tam SALAD: **R@1 92.2**
  - Global token'sız: **R@1 91.8** (−0.4 puan)
  - Dustbin'siz: **R@1 91.4** (−0.8 puan; "dustbin recall'da en etkili, sonra global token")
  → Global token'ı çıkarmak **−0.4 R@1** maliyetli (belgelenmiş). Ayrıca "en düşük varyanslı
  ~256 boyutu PCA ile atmak" *farklı* bir işlemdir (makale bunu test etmemiş) → **ÖLÇÜLECEK**,
  varsayılmayacak. Bu, Faz 3'ün kendi kuralına ("ölçümsüz/kaynaksız sayı yayınlanmaz") uygun.
- **Agresif WPCA recall'ı DÜŞÜRÜYOR (VLAD-BuFF, arXiv:2409.19293):** SALAD üzerinde
  **WPCA'nın recall'ı tutarlı biçimde düşürdüğü**, bunun düşük-boyutlu global descriptor'la
  verimli retrieval potansiyelini "belirgin biçimde kısıtladığı" raporlanıyor. Yani
  8448→1024/512 gibi *agresif* indirgeme SALAD'da bedava DEĞİL.
- **Karşı-örnek (EffoVPR, arXiv:2405.18065):** 128-dim ile Tokyo24/7'de SALAD-8448 ile
  *parite* bildiriyor (66× küçültme). AMA bu EffoVPR'ın **kendi** düşük-boyutlu
  descriptor'u — SALAD'a WPCA uygulamak değil. Yani "128-dim yeter" doğrudan bizim
  SALAD+PCA hattımıza taşınamaz; ancak düşük-boyutun *mümkün* olduğunu gösterir (doğru
  yöntemle).

**Sentez:** SALAD'a **agresif WPCA = ölçülebilir recall kaybı** (VLAD-BuFF). Bu, v1
hafızasındaki "PCA→1024 zorunlu" ifadesini nüanslar: PCA gerekli (bellek için) ama
*bedava değil*. Kaç puan? **Yayınlanmış SALAD+WPCA→1024 kesin recall tablosu bulunamadı**
→ bu **KENDİ VAL'İMİZDE ÖLÇÜLMELİ** (aşağı, §4). Varsayım yok.

### Ne yapmalı
- v2a'da **PCA-1024**'ü kendi cph+sf val'imizde ölç: R@1/5/10'u tam-8448 taban ile karşılaştır.
- Eğer kayıp kabul edilebilirse (örn. R@1 −1-2 puan) → v2a için onayla.
- **OPQ öncesi rotasyon** (PQ ile birlikte) veya **öğrenilmiş projeksiyon** (TLDR,
  arXiv:2110.09455 — DINO için PCA'yı geçtiği raporlanmış) alternatif; ölçüm kötüyse denenir.

---

## 2. Nicemleme (IVFPQ / PQ) — kaynaklı takaslar

v2b'nin bellek çözümü PQ'ya bağlı (5.1M → 0.4GB). Doğruluk maliyeti (FAISS/Pinecone
kaynakları):

- **PQ (8-bit, b=8) çoğu pratik ayarda "minimal recall kaybı"** (Pinecone/FAISS). Görüntü
  retrieval'ında **14× küçük indeks, Recall@5'e <%2 maliyet** raporlanmış.
- **AMA kötü konfigürasyonda recall ~%50'ye düşebilir** (FAISS issue #2096 senaryosu) →
  parametre (m, nbits, nlist, nprobe) **ölçülmeden bırakılamaz**.
- **OPQ (Optimized PQ) aynı kod boyutunda +2-6 recall puanı** kazandırır (vektörleri
  döndürüp alt-vektör dağılımını dengeler). SALAD descriptor'ları küme-yapılı (64×128)
  → alt-vektör dağılımı dengesiz olabilir → **OPQ muhtemelen değerli**.
- **nprobe = hız/recall takası:** IVF'de kaç hücre taranacağı. Düşük nprobe hızlı+düşük
  recall; yüksek nprobe yavaş+yüksek recall. Tek-kullanıcı demoda hız bol → **nprobe
  yüksek tutulabilir** (recall öncelikli). NOT: FAISS issue #2096 — bazı kurulumda nprobe
  artınca recall *düşebiliyor* (indeks patolojisi) → yine ölçüm şart.

### v2b öneri parametreleri (başlangıç, ölçümle ayarlanacak)
- **OPQ64 + PCA-512 + IVF(nlist≈√N) + PQ(m=64, nbits=8)** → 64 bayt/vektör.
- nprobe'u recall-doygunluğa kadar yükselt (demo hızı toleranslı).

---

## 3. Birikimli maliyet tablosu (kaynaklı + ölçülecek işaretli)

| Konfig | Boyut kaybı | Nicemleme kaybı | Beklenen R@1 etkisi | Kaynak durumu |
|---|---|---|---|---|
| Flat 8448 (v1) | — | — | taban (cph+sf 91.9%) | ölçüldü (v1) |
| Global token atma (→8192) | −0.4 R@1 | — | 92.2→91.8 | SALAD Table 5 (ablasyon) |
| PCA→8192 (düşük-varyans at) | **?** | — | **ÖLÇ** | kaynaksız — makale test etmemiş |
| PCA-1024 HNSW (v2a) | **?** (WPCA↓) | ~0 (HNSW exact-ish) | **ÖLÇ** | VLAD-BuFF: kayıp var, miktar ölçülecek |
| PCA-512 IVFPQ m64 (v2b) | ? (daha agresif) | <%2 (iyi ayar) / %50 (kötü) | **ÖLÇ** | FAISS/Pinecone + ölçüm |
| +OPQ | — | +2-6 puan geri kazanım | iyileştirir | FAISS/OPQ |

**Kilit mesaj:** Bellek (Faz 2) çözülmüş görünse de, **her sıkıştırma adımının recall
maliyeti kendi val'imizde ölçülmeden v2b onaylanamaz.** Nicemleme kaybı "minimal"den
"%50 çöküş"e kadar konfigürasyona bağlı — bu yüzden ölçüm-güdümlü.

---

## 4. Doğrulama protokolü (v2'de zorunlu adım)

v1'deki `scripts/evaluate.py` (Recall@1/5/10 + medyan km-hata, 25m radius GT) v2'de
**sıkıştırma-ablasyonu** için yeniden kullanılacak:

1. **Taban:** tam-8448 Flat, cph+sf val → R@1 91.9% (bilinen).
2. **v2a ablasyonu:** aynı val, PCA-1024 + HNSW → ΔR@1 ölç. Kabul eşiği önerisi:
   **R@1 kaybı ≤ 2 puan** → onayla; > ise TLDR/öğrenilmiş projeksiyon dene.
3. **v2b ablasyonu:** PCA-512 + (O)PQ + IVF → nprobe sweep, R@1 vs hız eğrisi. Kötü
   konfig (recall çöküşü) tespiti için nlist/m/nbits grid.
4. Her ablasyon sonucu `docs/research-v2/` veya metrik JSON'a (gitignored) yazılır;
   **kaynaksız/ölçümsüz sayı README'ye girmez** (v1 kuralı).

> **PCA fit verisi uyarısı:** PCA'yı cph+sf'in kendisinden fit edip cph+sf'te ölçmek
> iyimser (overfit). Doğrusu: PCA'yı **OSV5M-alt** (bağımsız) üzerinde fit et, cph+sf'te
> ölç — ya da v2a val'ini OSV5M-alt'tan ayır. Aksi hâlde recall şişer.

---

## 5. Faz 3 özeti (onay noktası)

- **PCA SALAD'da bedava değil:** "→8192 kayıpsız" iddiası **kaynaksızdı, kaldırıldı**
  (kullanıcı yakaladı); makale Table 5 aksine global token atmanın **−0.4 R@1** maliyetli
  olduğunu gösteriyor. Agresif WPCA (→1024/512) recall'ı düşürür (VLAD-BuFF) — miktar
  **ölçülecek**, varsayılmayacak.
- **PQ kaybı konfigürasyona bağlı:** iyi ayarda <%2 (Recall@5), kötü ayarda ~%50 çöküş;
  OPQ +2-6 puan geri kazandırır; nprobe hız/recall takası (demoda recall-öncelikli).
- **v2a önerisi:** PCA-1024 + HNSW, kabul eşiği R@1 kaybı ≤2 puan.
- **v2b önerisi:** OPQ64 + PCA-512 + IVFPQ(m64,nbits8), nprobe sweep ile ayar.
- **Zorunlu:** `evaluate.py` ile sıkıştırma-ablasyonu + bağımsız PCA-fit (overfit önlemi).
  Ölçümsüz sayı yayınlanmaz.

**Sonraki: Faz 4 — depolama/altyapı** (259GB disk + ~60 saat embed süresi; ev Ubuntu+Docker
sunucusunun rolü; HF Spaces free-tier limitleri). Onayınızı bekliyorum.

---

### Kaynaklar
- Izquierdo & Civera, *Optimal Transport Aggregation for VPR* (SALAD), CVPR 2024,
  arXiv:2311.15937 — **Table 5** bileşen ablasyonu: global token −0.4 R@1, dustbin −0.8
  (8448 = 64×128 VLAD + 256 global token). NOT: makale PCA→8192 kayıpsızlık iddiası İÇERMEZ
  (ilk taslaktaki o iddia kaynaksızdı, düzeltildi).
- VLAD-BuFF, arXiv:2409.19293 — SALAD üzerinde WPCA'nın recall'ı tutarlı düşürmesi.
- EffoVPR, arXiv:2405.18065 — 128-dim ile Tokyo24/7'de SALAD paritesi (kendi descriptor'u).
- TLDR, arXiv:2110.09455 — DINO için öğrenilmiş boyut indirgeme (PCA alternatifi).
- Pinecone/FAISS PQ eğitim serisi — 8-bit PQ minimal kayıp, 14× küçültme <%2 R@5 maliyet.
- FAISS OPQ — aynı kod boyutunda +2-6 recall puanı.
- FAISS issue #2096 — IVFPQ'da nprobe/recall patolojisi, kötü konfig ~%50 recall.
