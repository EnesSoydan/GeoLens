# GeoReasoner — Araştırma Raporu
## FAZ A: Temeller, Literatür ve Fizibilite

> Araştırma-only çıktı. Kod/uygulama dosyası üretilmemiştir.

---

## Bölüm 1 — Temel Kavramlar (Öğretici)

### 1.1 Görsel Coğrafi Konumlandırma (Visual Geolocation / Image Geolocalization)
**Kavram:** Bir görüntünün yalnızca *içeriğine* bakarak (EXIF/metadata olmadan) Dünya üzerinde nerede çekildiğini (GPS koordinatı) tahmin etme görevi.
**Neden:** AR, robotik, otonom araçlar, OSINT/adli analiz. GPS sinyali olmayan ortamlarda konum çıkarımı.
**Alternatif:** Metadata tabanlı (EXIF GPS) — çoğu görselde yok/silinmiş → görsel içerikten çıkarım gerekir.
**Tarihsel kök:** Google PlaNet (2016), koordinat regresyonu yerine "geocell sınıflandırma" yaklaşımını başlattı.

### 1.2 Görsel Yer Tanıma (Visual Place Recognition — VPR)
**Kavram:** Sorgu görüntüsünün GPS-etiketli bir *referans veritabanıyla* eşleştirilerek (metre düzeyi) konumunun tanınması.
**Neden:** V-SLAM döngü kapatma, GNSS'siz konumlandırma. Şehir/sokak ölçeğinde standart.
**Geolocalization'dan farkı:** VPR önceden kurulu veritabanına *eşleme* yapar (tanıma); geolocalization açık-dünya GPS *tahmini* yapar.

### 1.3 Görüntü Erişimi (Image Retrieval / CBIR)
**Kavram:** Görüntüleri içerikle getirme. Derin metrik öğrenmeyle (triplet loss) benzer görüntülerin kümelendiği embedding uzayı öğrenilir; kosinüs benzerliği / kNN ile arama.
**Neden:** VPR'ın altında yatan mekanizma. CBIR genel görev, VPR onun coğrafi özelleşmiş hali.

### 1.4 GeoGuessr AI
**Kavram:** GeoGuessr oynayan YZ. En güçlü örnek Stanford PIGEON: semantik geocell'ler + çok-görevli kontrastif ön-eğitim (StreetCLIP) + ProtoNets + retrieval. Bir profesyoneli 6-0 yendi; tahminlerin >%40'ı 25 km içinde.

### 1.5 Açıklanabilir YZ — XAI (Computer Vision)
**Kavram:** Modelin *neden* o tahmini yaptığını gösteren yöntemler. Grad-CAM, sınıf skorunun CNN öznitelik haritalarına göre gradyanıyla "modelin nereye baktığını" gösteren ısı haritası üretir.
**Neden kritik:** Projenin ayırt edici değeri — tahmin + "hangi görsel ipuçları" görselleştirmesi.

---

## Bölüm 2 — Literatür ve SOTA Taraması

| Yöntem | Yıl | Tip | Çekirdek fikir | Kod |
|---|---|---|---|---|
| IM2GPS (Hays & Efros) | 2008/2015 | Retrieval | Gezegen ölçekli ilk geoloc; el-yapımı öznitelik + kNN | site |
| PlaNet (Google) | 2016 | Sınıflandırma | Dünya'yı çok-ölçekli geocell'lere böl; CNN hücre sınıflar | resmi yok |
| NetVLAD | 2016 | Retrieval (tanımlayıcı) | Türevlenebilir VLAD havuzlama, uçtan uca | ✓ |
| CosPlace (Berton) | 2022 | Retrieval (sınıflandırma olarak eğitilir) | Eğitimi sınıflandırma kurar; %80 az GPU bellek | ✓ |
| EigenPlaces (Berton) | 2023 | Retrieval | Bakış-açısı dayanıklı; %60 az GPU | ✓ |
| GeoCLIP | 2023 | Hibrit | İlk GPS-kodlama; CLIP tarzı görüntü↔sürekli GPS hizalama | ✓ |
| PIGEON/PIGEOTTO | 2023/24 | Hibrit | Semantik geocell + CLIP + retrieval iyileştirme | kısmi |

**Güncel VPR SOTA (2023–2025):** AnyLoc (DINOv2 eğitimsiz), SALAD (CVPR'24, DINOv2 + Sinkhorn, Tokyo24/7 R@1 ~%94.6), BoQ (CVPR'24, R@1 ~%95.2), CricaVPR (DINOv2 + adaptör).

**Eğilimler:** (1) Şehir/sokak ölçeği = **retrieval-first** (global tanımlayıcı → kNN). (2) Gezegen ölçeği saf sınıflandırmadan **hibride** kaydı. (3) Sırtlar: DINOv2 VPR'da, CLIP gezegen ölçeğinde hâkim.

**Kaynaklar:** Survey arXiv:2112.15202 · PlaNet arXiv:1602.05314 · NetVLAD arXiv:1511.07247 · CosPlace arXiv:2204.02287 · EigenPlaces arXiv:2308.10832 · GeoCLIP arXiv:2309.16020 · PIGEON arXiv:2307.05845 · SALAD arXiv:2311.15937 · AnyLoc (github.com/AnyLoc/AnyLoc)

---

## Bölüm 4 — Problem Tipi Analizi (Karar Matrisi)

| Kriter | Sınıflandırma | Regresyon | **Retrieval** | Hibrit |
|---|---|---|---|---|
| Nasıl çalışır | Hücre olasılığı | Doğrudan koordinat | Embedding + FAISS kNN | Kombinasyon |
| Hassasiyet | Hücre sınırı | Düşük | **Metre düzeyi** | En yüksek |
| Belirsizlik | İyi | Zayıf | Orta | İyi |
| Eğitim ihtiyacı | Büyük veri + GPU | Az | **Pretrained, eğitim yok** | Yüksek |
| VRAM (8GB) | Orta | Düşük | **Sadece çıkarım** | Zorlayıcı |
| Açıklanabilirlik | Dolaylı | Yok | **Doğal (referans görseli)** | Orta |
| Yeni konum | Yeniden eğitim | Yeniden eğitim | **Index'e ekle** | Karışık |

**Öneri: Retrieval-first.** Gerekçe: embedding çıkarımı 8GB'e sığar; FAISS index 64GB RAM'de (CPU) yaşar; pretrained model → büyük GPU eğitimi gerekmez; doğal açıklanabilirlik. Regresyon zayıf; saf sınıflandırma 8GB'de büyük ölçekte pratik değil.

**Farklılaşma:** Liderler kapalı (GeoSpy, Picarta) ya da araştırma-amaçlı, UI/deployment/XAI yok (PIGEON, GeoCLIP, VPR repoları). Boşluk: retrieval + nesne düzeyi XAI + harita UI + full-stack'i birleştiren *açık* uygulama yok → GeoReasoner buraya oturuyor.

---

### Faz A Özeti
- 5 temel kavram öğretici formatta tanımlandı.
- 13+ SOTA yöntem tarandı; eğilim: şehir ölçeğinde retrieval-first, DINOv2/CLIP sırtları.
- Karar matrisi donanıma göre **retrieval-first** mimariyi önerdi.
