# Faz 6 — Sentez & Kademeli v2 Yol Haritası

> **Görev tipi:** Araştırma + planlama sentezi (kod değil). Beş araştırma fazının
> (veri kaynağı, ölçekleme mimarisi, boyut indirgeme doğruluğu, depolama/altyapı, yasal)
> kararlarını tek bir uygulanabilir yol haritasına bağlar.
> **Ana öneri:** **v2a ile başla** (güvenli, laptop-yerel, kanıtlanmış pipeline'a en yakın);
> v2b'ye yalnız iki ön koşul doğrulandıktan sonra geç. v3'e sıçrama yok.
> **Donanım:** RTX 4060 8GB VRAM, 64GB RAM, laptop + ev sunucusu (Casper Nirvana,
> 466GB disk **434GB boş**, Ubuntu+Docker+Portainer).

---

## 0. Beş fazın tek-cümlelik özeti

| Faz | Karar |
|---|---|
| 1 Veri kaynağı | Birincil **OSV5M** (5.1M, küresel-dengeli, retrieval-hazır); yoğunlaştırma Mapillary API (SDK); boşluk KartaView. |
| 2 Ölçekleme mimarisi | **v2a** 100-500k PCA-1024+HNSW; **v2b** 1-5M PCA-512+IVFPQ; v3 ertelendi. IVFPQ RAM darboğazını kaldırır. |
| 3 Boyut indirgeme | PCA SALAD'da bedava DEĞİL (WPCA recall↓, global token −0.4 R@1); PQ kaybı konfige bağlı (+OPQ). **Hepsi ölçülecek.** |
| 4 Depolama/altyapı | Ölçüm: 434GB boş. Naif v2b (~430GB) güvensiz → **iki-geçişli streaming zorunlu** (tepe ~10-30GB). |
| 5 Yasal | Mapillary düşük-orta; **CC-BY-SA ShareAlike türev-indeks viralliği açık** → indeksi CC-BY-SA dağıt/private; BY atıf zorunlu. |

---

## 1. Ana strateji: v2a önce, v2b koşullu

v1 → küresel arasındaki **en küçük sağlam adım v2a**. Gerekçeler her fazdan geliyor:
- **Faz 2:** v2a mimarisi (PCA katmanı + HNSW) v1 SALAD pipeline'ına en yakın; en az yeni parça.
- **Faz 4:** v2a (~42GB) **laptop'a bile sığar** — ev sunucusu depolama için bile şart değil;
  embed ~6 saat (tek oturum, checkpoint gerekmez).
- **Faz 3:** v2a'da tek belirsizlik PCA-1024 recall kaybı → tek ablasyonla ölçülür (≤2 puan
  eşiği). v2b ise PCA+PQ+IVF+nprobe çok-değişkenli ablasyon gerektirir.
- **Faz 1:** v2a, OSV5M'in küçük bir küresel-dengeli alt-kümesiyle kurulabilir → tam 259GB
  indirmeye gerek yok (streaming-indirme ön koşulu v2a için de doğrulanmış olur).

**v2b'ye geçiş iki ön koşula bağlı (Faz 4):** (a) OSV5M **streaming-indirme** çalışır
(shard-shard, snapshot_download değil), (b) **§7.2 güvenlik payı** (50GB headroom, try/finally
temizlik, ayrı mount) kurulu. İkisi de yoksa v2a'da kal.

---

## 2. Kademeli yol haritası (uygulama sırası)

### Adım 0 — Streaming-indirme kanıtı (v2a/v2b ortak ön koşul)
- OSV5M'den `load_dataset(..., streaming=True)` veya dosya-dosya `hf_hub_download` ile
  **küçük bir shard** çek, embed et, sil → disk tepesinin şard-boyutunda kaldığını **ölç**.
- Çıktı: streaming-indirme çalışıyor mu? (v2b'nin (a) ön koşulu.)

### Adım 1 — v2a inşası (ilk gerçek hedef)
1. OSV5M'den **~100-500k** küresel-dengeli alt-küme seç (veya hedef kıta/ülke).
2. Embed (SALAD 8448, laptop GPU ~6h). **PCA'yı bu alt-kümede fit ET, cph+sf'te DEĞİL**
   (Faz 3 overfit önlemi).
3. **PCA→1024** uygula → **HNSW** indeks + SQLite metadata (v1 build_index deseni).
4. **Ablasyon:** v1 `evaluate.py` ile Recall@1/5/10 — 8448-Flat taban vs PCA-1024-HNSW.
   **Kabul: R@1 kaybı ≤2 puan.** Aşarsa TLDR/öğrenilmiş projeksiyon dene (Faz 3).
5. Servis: v1 FastAPI+Gradio+Docker; indeks ev sunucusu veya HF dataset (private/CC-BY-SA).
6. **Atıf bloğu** (Faz 5): UI+README'ye Mapillary logo+link + OSV5M + CC-BY-SA-4.0.

### Adım 2 — v2b'ye geçiş (KOŞULLU: Adım 0 + güvenlik payı OK ise)
1. **İki-geçişli streaming** (Faz 4 zorunlu yol): geçiş-1 alt-kümeden PCA-512 + IVFPQ eğit;
   geçiş-2 tüm 5.1M'i embed→PCA→PQ-kodla→indekse ekle, ham float32 YAZMA, ham shard sil.
2. Embed ~64h → **checkpoint'li dirençli job** (laptop GPU; CPU imkânsız).
3. İndeks: **OPQ64 + PCA-512 + IVFPQ(m64, nbits8)**; nprobe sweep (recall doygunluğa,
   demoda hız bol). Ablasyon: recall vs nprobe eğrisi; kötü-konfig (~%50 çöküş) taraması.
4. Nihai ~0.5GB IVFPQ indeks → HF dataset (CC-BY-SA/private) veya ev sunucusu 7/24 servis.

### Adım 3 — v3 (UFUK, bu yol haritasında YOK)
- GeoCLIP-tarzı vs hiyerarşik retrieval kararı ertelendi. v2b kanıtlanmadan başlanmaz.

---

## 3. Dürüst donanım-gerçekçiliği değerlendirmesi

| Boyut | v2a (~500k) | v2b (5.1M) |
|---|---|---|
| İndeks RAM | ~2.6GB | ~0.4GB (IVFPQ) |
| Disk tepe | ~42GB (laptop OK) | ~10-30GB **yalnız iki-geçişli streaming ile** |
| Embed süresi | ~6h tek oturum | ~64h checkpoint'li (CPU imkânsız) |
| Yeni parça sayısı | az (PCA+HNSW) | çok (PCA+PQ+IVF+OPQ+streaming+checkpoint) |
| Doğruluk riski | tek ablasyon (PCA-1024) | çok-değişkenli (PCA+PQ+nprobe) |
| Ön koşul | streaming-indirme kanıtı | + güvenlik payı + 64h dirençli job |
| **Verdikt** | **güvenli, şimdi yapılabilir** | **mümkün ama koşullu + emek-yoğun** |

**Açık gerçekler (abartısız):**
- Donanım küresel ölçek için **yeterli ama sınırda**: 64GB RAM + tek RTX4060 5.1M'i ancak
  IVFPQ + streaming + checkpoint ile taşır; naif yol her eksende (RAM/disk/süre) duvara çarpar.
- **En büyük risk teknik değil, doğruluk:** SALAD+agresif PCA/PQ recall kaybı ölçülmeden
  v2b'nin gerçek kalitesi bilinmiyor (Faz 3). "5M vektör" etkileyici ama R@1 çökerse değersiz.
- **v2a somut değer üretir:** cph+sf'ten çok-şehir/bölgeye geçiş, kanıtlanmış pipeline,
  ölçülebilir kalite — portföy için v2b'den daha az riskli, daha hızlı gösterilebilir kazanım.
- Yasal (Faz 5) teknik değil ama gerçek: dağıtılan indeks CC-BY-SA ShareAlike sorusu →
  ihmal edilmemeli.

---

## 4. Nihai öneri (karar)

**BAŞLA: v2a.** OSV5M küresel-dengeli ~100-500k alt-küme + PCA-1024 + HNSW; PCA'yı bağımsız
alt-kümede fit et; `evaluate.py` ablasyonuyla R@1 kaybını ölç (≤2 puan kabul); v1 servis
yığınıyla yayınla + atıf bloğu. **Adım 0 (streaming-indirme kanıtı) v2a'nın da ilk işi.**

**v2b'yi taahhüt ETME** — yalnız (a) streaming-indirme + (b) güvenlik payı doğrulanınca,
ve v2a ablasyonu PCA'nın kabul edilebilir olduğunu gösterince koşullu olarak geç. **v3'e
sıçrama yok** (v1 kuralı: aşırı-mühendislikten kaçın).

Bu, v1'in "retrieval-first + ölçüm-güdümlü + aşırıya kaçma" ilkelerinin küresel ölçeğe
sadık uzantısıdır.

---

### Kaynaklar
- Faz 1-5 belgeleri (docs/research-v2/faz-1..5) ve içlerindeki kaynaklar.
- v1 kararları & ölçümleri (MEMORY, docs/architecture/*).
