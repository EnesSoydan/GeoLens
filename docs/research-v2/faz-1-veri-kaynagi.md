# Faz 1 — Veri Kaynağı Stratejisi (v2)

> **Görev tipi:** Araştırma + planlama. Bu belge kod değil, karar dokümanıdır.
> **Amaç:** GeoLens'i iki-şehir (cph+sf, ~18.9k referans) ölçeğinden küresel/kıtasal
> ölçeğe taşımak için hangi veri kaynağının/kaynakların kullanılacağına karar vermek.
> **Kırmızı çizgi:** Google Street View kazıma (scraping) YASAK — ToS ihlali. Yalnız
> açık / CC-BY-SA kaynaklar.
> **Donanım (değişmedi):** RTX 4060 8GB VRAM, 64GB RAM, Ryzen 7 7435HS.

---

## 0. Bağlam ve kısıt (v1'den devralınan)

v1 mimari kararı **retrieval-first**: sorgu görüntüsü → SALAD (DINOv2 ViT-B/14) ile
**8448-dim** descriptor → FAISS Flat (kosinüs) → en yakın referansın GPS'i. Referans
indeksi = MSLS-val cph+sf, 18916 vektör. Ham FAISS Flat bellek maliyeti:

```
18916 × 8448 × 4 bayt (float32) ≈ 0.60 GB   → 64GB'da rahat.
```

Bu doğrusal ölçekleniyor. **Küresel ölçekte ham float32 FAISS Flat imkânsız** (Faz 2/3
konusu, burada yalnız kaynak seçimini kısıtladığı ölçüde anıyoruz):

```
5.100.000 × 8448 × 4 bayt ≈ 172 GB RAM   → 64GB'ı 2.7× aşıyor. İMKÂNSIZ.
```

Bu, v1 hafızasındaki "8448-dim ham yalnız ≤1M'e kadar (64GB)" uyarısını doğrular.
**Sonuç: veri kaynağı ne olursa olsun, küresel ölçek için (a) alt-örnekleme ve/veya
(b) boyut indirgeme + sıkıştırma (PCA/IVFPQ) ÖN KOŞULDUR.** Bu Faz 3'ün konusu; ama
kaynak seçimini de etkiler çünkü "5.1M görüntüyü indirsem bile tamamını ham indeksleyemem".

---

## 1. Aday veri kaynakları

Üç açık kaynak karşılaştırıldı. Hepsi **CC-BY-SA-4.0** (atıf + aynı lisansla paylaş).

### 1.1 OSV5M (OpenStreetView-5M)

- **Ne:** Küresel görsel geo-lokalizasyon için özel derlenmiş, hazır (assembled)
  veri seti. **4.894.685 eğitim + 210.122 test = ~5.1M görüntü**.
- **Kaynak makale:** *OpenStreetView-5M: The Many Roads to Global Visual Geolocation*,
  Astruc vd., CVPR 2024 (arXiv:2404.18873). HF: `osv5m/osv5m`.
- **Toplama metodolojisi (makaleden):** Mapillary'den çekilmiş, **100m×100m ızgara**
  üzerinden mekânsal olarak dengelenmiş örnekleme (yoğun bölgelerin baskın olmaması
  için). 70.000 şehir, 225 ülke, küresel neredeyse-uniform dağılım. %96.1'i
  lokalize-edilebilir (GPS'li).
- **Format:** 512px yükseklik (~792px genişlik) JPEG; her görüntüde lat/lon + ters-geocode
  metadata (ülke, şehir, bölge). Katı train/test ayrımı (mekânsal, sızıntısız).
- **Boyut (disk):** **~259 GB** (tam set). `snapshot_download` / `load_dataset` ile
  parça-parça çekilebilir.
- **Lisans:** CC-BY-SA-4.0 (Mapillary türevli → ShareAlike zinciri).
- **GeoLens için uygunluk:** ✅ **Retrieval-hazır.** GPS+görüntü çiftleri zaten temiz,
  küresel-dengeli, tek indirmede geliyor. API sayfalama/rate-limit/OAuth derdi yok.
  Mekânsal denge, retrieval indeksinin bir bölgeye çökmesini engeller (v2b/v3 için ideal).
- **Dezavantaj:** 259GB disk (tam). 64GB RAM'e tam ham indeks sığmaz → **alt-örnekleme
  ZORUNLU**. Ayrıca statik snapshot (2024) — güncellenmez.

### 1.2 Mapillary API (v4)

- **Ne:** Canlı, kitle-kaynaklı sokak-seviyesi görüntü platformu. v1'de MSLS
  (Mapillary Street-Level Sequences) türevi zaten kullanıldı.
- **Erişim:** OAuth2 token (`MLY|...`), REST API v4. `graph.mapillary.com/images`
  bbox sorgusu. **bbox limiti < 0.01 derece²** (16 Oca 2026 değişikliği), istek başına
  ~2000 feature. Büyük alanlar için resmi `mapillary-python-sdk` sayfalama yapar.
- **Görüntü:** çeşitli çözünürlükler (thumb 256/1024/2048/original). Her görüntüde
  `computed_geometry` (GPS), `compass_angle` (heading), `captured_at`, `sequence`.
- **Lisans:** görüntüler **CC-BY-SA-4.0**; atıf = Mapillary logosu + görüntü sayfasına link.
  Tüm kullanımlar (ticari dahil) ücretsiz.
- **Rate-limit:** resmi yayınlanmış sayısal limit YOK; SDK+sayfalama önerilir.
- **GeoLens için uygunluk:** ✅ **Hedefli yoğunlaştırma için.** Belirli bir şehri/bölgeyi
  taze ve yoğun kaplamak istersen (v2a: 5-10 şehir), API ile o bbox'ları tarayıp güncel
  görüntü çekebilirsin. Canlı → güncellenebilir indeks.
- **Dezavantaj:** Küresel toplu indirme için elverişsiz (bbox<0.01°² → milyonlarca
  istek, sayfalama, saatler-günler). "Sıfırdan 5M küresel" için OSV5M'i yeniden icat
  etmek olur. **Ham indirme + kendi indeksini dağıtma CC-BY-SA türev sorusunu açar** (§4).

### 1.3 KartaView (eski OpenStreetCam)

- **Ne:** Açık, topluluk-güdümlü sokak-seviyesi görüntü platformu (Mapillary alternatifi).
- **Erişim:** API ham + işlenmiş indirme yollarını expose ediyor (Mapillary v4 "original"
  erişimini kısıtladı → KartaView ham veriyi daha açık bırakıyor). **Toplu (batch)
  endpoint yok** (2020 durumu), auth belgelenmemiş.
- **Lisans:** CC-BY-SA-4.0.
- **Kapsama:** Güneydoğu Asya'da güçlü; Mapillary'nin zayıf olduğu bölgelerde
  **tamamlayıcı**.
- **GeoLens için uygunluk:** 🟡 **Tamamlayıcı, birincil değil.** Coğrafi boşluk doldurma
  (örn. SE Asya) için değerli ama olgunlaşmamış API + batch eksikliği → v2 başlangıcı için
  ana kaynak değil. v3'te kapsama genişletirken düşünülebilir.

---

## 2. Karşılaştırma tablosu

| Ölçüt | OSV5M | Mapillary API v4 | KartaView |
|---|---|---|---|
| Erişim şekli | HF snapshot (tek indirme) | REST API (OAuth, sayfalama) | REST API (belirsiz auth) |
| Hazır GPS+görüntü | ✅ temiz, retrieval-hazır | 🟡 API'den birleştir | 🟡 API'den birleştir |
| Küresel kapsama | ✅ 225 ülke, uniform | ✅ ama toplu çekim zor | 🟡 SE Asya güçlü |
| Mekânsal denge | ✅ 100m ızgara örnekleme | ❌ ham yoğunluk (kendin dengele) | ❌ |
| Toplu indirme | ✅ (259GB) | ❌ bbox<0.01°², milyon istek | ❌ batch yok |
| Güncellik | ❌ statik 2024 snapshot | ✅ canlı | ✅ canlı |
| Lisans | CC-BY-SA-4.0 | CC-BY-SA-4.0 | CC-BY-SA-4.0 |
| Türev-indeks riski | CC-BY-SA ShareAlike (§4) | CC-BY-SA ShareAlike (§4) | CC-BY-SA ShareAlike (§4) |
| v2 rolü | **Birincil çekirdek** | Hedefli yoğunlaştırma | Coğrafi boşluk (v3) |

---

## 3. Öneri (Faz 1 kararı)

**Katmanlı, tek-birincil-kaynak + iki-tamamlayıcı stratejisi:**

1. **Birincil çekirdek = OSV5M.**
   - *Neden:* Retrieval'a hazır (GPS+görüntü temiz), küresel mekânsal-dengeli, tek
     indirmede geliyor, API/rate-limit/OAuth külfeti yok. Bir retrieval indeksi kurmak
     için lazım olan tam olarak budur. Mekânsal denge, indeksin metropollere çökmesini
     engeller.
   - *Alternatif neden değil:* Mapillary API ile sıfırdan 5M küresel toplamak, OSV5M
     ekibinin zaten yaptığı işi (100m ızgara örnekleme) milyonlarca API isteğiyle yeniden
     icat etmek olur.

2. **Hedefli yoğunlaştırma = Mapillary API v4.**
   - *Neden:* v2a'da (5-10 seçili şehir) bir şehri güncel ve yoğun kaplamak istersen,
     o bbox'ları API ile tarayıp taze görüntü eklersin. OSV5M statik snapshot; canlı
     tazelik yalnız API'den gelir.
   - *Uygulama notu (kullanıcı, Faz 1 onayı):* Ham REST + manuel bbox parçalama YERİNE
     resmi **`mapillary-python-sdk`** kullan. Mapillary'nin kendi dokümantasyonu büyük
     alan sorguları için bunu öneriyor; SDK 0.01°² bbox limitini (16 Oca 2026) dahili
     sayfalama/parçalama ile soyutlar → limiti elle bölmekle uğraşmaktan kurtarır. v2
     yoğunlaştırma script'i bu SDK üzerine yazılacak.

3. **Coğrafi boşluk doldurma = KartaView (v3, opsiyonel).**
   - *Neden:* OSV5M/Mapillary'nin zayıf olduğu bölgelerde (SE Asya) tamamlayıcı. v2
     başlangıcında ZORUNLU değil.

### Kritik gerçekçilik notu (donanım)

**Tam OSV5M'i ham indeksleyemezsin:**

```
5.1M × 8448-dim × 4B = 172 GB RAM  ≫  64 GB.   → İMKÂNSIZ (ham Flat).
259 GB disk (tam görüntü seti)     →  50GB HF free-tier'ı da aşar.
```

Bu yüzden Faz 1 kararı **kaynak seçimi**ni belirler ama **ölçek** kararını Faz 2/3'e
bırakır. Somut v2 yolu (öneri, Faz 2'de kesinleşecek):

- **v2a (gerçekçi ilk hedef):** OSV5M'den **~100-500k görüntü alt-örnekle** (kıta/ülke
  odaklı veya küresel-seyrek), embed et. 500k × 8448 × 4B = **16.9 GB** → 64GB'a sığar
  ama rahat değil → **PCA→1024 sonrası 500k × 1024 × 4B = 2.0 GB** → HNSW ile hızlı.
- **v2b (kıta/ülke):** IVFPQ ile daha büyük alt-küme (milyonlar), sıkıştırılmış.
- Boyut-indirgeme doğruluk maliyeti (SALAD'da WPCA recall'ı DÜŞÜRÜYOR — arXiv:2409.19293)
  **Faz 3'ün** konusu; kaynak seçimini değiştirmez ama alt-örnekleme miktarını belirler.

---

## 4. Yasal / lisans (Faz 5 detayına köprü)

Üç kaynak da **CC-BY-SA-4.0**. İki nokta v2 için kritik ve şimdi işaretlenmeli:

1. **Atıf (BY):** Mapillary görüntüleri için atıf = Mapillary logosu + görüntü sayfasına
   link. OSV5M türevi de bu zincire tabi. Canlı demoda görünür atıf gerekir.

2. **ShareAlike (SA) — türev indeks sorusu (AÇIK, hukukçu gerekir):**
   CC-BY-SA'nın ShareAlike şartı, görüntülerden türetilen **embedding indeksinin** bir
   "türev eser" (adaptation) sayılıp sayılmayacağı sorusunu açar. Sayılırsa, dağıttığın
   FAISS indeksi de CC-BY-SA ile paylaşılmak zorunda kalabilir (viral etki). Bu, v1'deki
   SALAD GPL-3.0 kararına benzer bir "kamuya sunum = dağıtım olabilir" durumu. **Karar:**
   şimdilik işaretlendi; v2 gerçekten dağıtılırsa (HF Space + indirilebilir indeks) hukuk
   danışmanı önerilir. Faz 5'te derinleştirilecek.

---

## 5. Faz 1 özeti (onay noktası)

- **Birincil kaynak: OSV5M** (5.1M, küresel-dengeli, retrieval-hazır, CC-BY-SA).
- **Yoğunlaştırma: Mapillary API v4** (hedefli, canlı).
- **Boşluk: KartaView** (v3, opsiyonel).
- **En sert kısıt:** tam OSV5M ham indeks 172GB RAM / 259GB disk → **imkânsız**;
  alt-örnekleme + boyut indirgeme (Faz 3) ÖN KOŞUL. v2a gerçekçi hedef: ~100-500k
  alt-küme + PCA→1024 + HNSW (~2GB RAM).
- **Yasal:** CC-BY-SA atıf zorunlu; türev-indeks ShareAlike sorusu açık (Faz 5).

**Sonraki: Faz 2 — ölçekleme mimarisi karar matrisi** (v2a/v2b/v3, RAM/disk bütçeleri
faz-b.md tablo formatında). Onayınızı bekliyorum.

---

### Kaynaklar

- Astruc vd., *OpenStreetView-5M: The Many Roads to Global Visual Geolocation*,
  CVPR 2024, arXiv:2404.18873. HF: `osv5m/osv5m`.
- Mapillary API v4 dokümantasyonu (graph.mapillary.com, bbox<0.01°² değişikliği
  16 Oca 2026), `mapillary-python-sdk`.
- KartaView API (kartaview.org), CC-BY-SA-4.0.
- VLAD-BuFF, arXiv:2409.19293 (SALAD'da WPCA recall etkisi — Faz 3 önizleme).
- Creative Commons CC-BY-SA-4.0 lisans metni (ShareAlike/adaptation tanımı).
