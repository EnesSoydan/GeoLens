# Faz 4 — Depolama / Altyapı (v2)

> **Görev tipi:** Araştırma + planlama (kod değil).
> **Amaç:** Faz 2/3'te bellek darboğazı IVFPQ ile çözülünce ortaya çıkan **gerçek**
> engelleri (disk 259GB, embed süresi ~60 saat, dağıtım limitleri) fiziksel/altyapı
> düzleminde planlamak. Ev Ubuntu+Docker sunucusunun rolü + HF Spaces free-tier limitleri.
> **Donanım:** RTX 4060 8GB VRAM, 64GB RAM, Ryzen 7 7435HS (laptop) + ev sunucusu (Ubuntu+Docker).

---

## 0. Bağlam: darboğaz nereye kaydı

Faz 2 gösterdi: IVFPQ ile 5.1M vektör **~0.4GB RAM**'e sığar. Yani bellek artık engel
değil. Faz 4'ün konusu, kalan üç fiziksel engel:

1. **Disk:** OSV5M tam görüntü seti **259GB** (embed etmek için önce indirmek gerek).
2. **Embed süresi:** 5.1M × (RTX4060 ~20-25 img/s) ≈ **57-71 saat** kesintisiz GPU.
3. **Dağıtım:** üretilen indeks + servis nereye konuşlanacak (HF free-tier sınırları).

---

## 1. Disk bütçesi (uçtan uca akış)

v2b (tam OSV5M) senaryosu için disk ihtiyacı, aşamalı:

| Aşama | Veri | Boyut | Kalıcı mı? |
|---|---|---|---|
| Ham görüntüler (indirilen) | OSV5M JPEG | **259 GB** | geçici (embed sonrası silinebilir) |
| Model ağırlıkları | DINOv2+SALAD | ~1.4 GB | kalıcı |
| Ham embedding (float32, ara) | 5.1M×8448×4B | **172 GB** | geçici (PCA/PQ sonrası silinir) |
| Nihai IVFPQ indeksi | 5.1M×64B + IVF/PCA | **~0.5 GB** | kalıcı (dağıtılan) |
| Metadata (SQLite lat/lon) | 5.1M satır | ~1-2 GB | kalıcı |

**Kritik gözlem:** Pipeline'ın *tepe* disk ihtiyacı **259GB (ham) + 172GB (ara embed) ≈
430GB** olabilir — ama bu **akış (streaming) ile azaltılabilir**: görüntüyü indir → embed
et → embedding'i diske yaz → ham görüntüyü SİL (batch batch). Böylece tepe, tek batch ham
(~GB'lar) + birikmiş embedding (172GB) ile sınırlanır. Yine de **172GB ara embedding + 259GB
akışlı ham** → laptop'un dahili diskine sığmaz (tipik 512GB-1TB, zaten dolu) → **harici disk
veya ev sunucusu diski ZORUNLU**.

> **Alt-örnekleme kaldıracı (Faz 1/2):** v2a (~500k) tercih edilirse disk 10× düşer:
> ~25GB ham (akışlı) + ~17GB ara embed → laptop diskine SIĞAR. Bu, v2a'nın "önce inşa
> edilebilir" olmasının bir başka nedeni.

---

## 2. Embed süresi (kapasite planı)

- **v2a (500k):** 500k / 22 img/s ≈ **6.3 saat** GPU. Tek gecede biter. Laptop RTX4060.
- **v2b (5.1M):** 5.1M / 22 img/s ≈ **64 saat** (~2.7 gün) kesintisiz GPU. Laptop bunu
  yaparken kullanılamaz + termal/kesinti riski → **checkpoint'li, dirençli batch job**
  gerektirir (her N batch'te embedding'i diske flush, kaldığı yerden devam).
- **Ev sunucusu rolü:** Eğer ev sunucusunda GPU varsa embed işini oraya devret (laptop
  serbest kalır). GPU yoksa CPU embed **~1.8 img/s** (v1 ölçümü) → 5.1M / 1.8 ≈ **787 saat
  (~33 gün)** → **CPU ile v2b imkânsız**; ev sunucusu GPU'suzsa yalnız *depolama + servis*
  rolü üstlenir, embed laptop GPU'sunda kalır.

**Karar:** v2b embed'i **laptop RTX4060 + checkpoint'li dirençli job** (ev sunucusu GPU'suz
varsayımıyla). v2a ise tek-oturum, checkpoint opsiyonel.

---

## 3. Ev Ubuntu + Docker sunucusunun rolü

Kullanıcının ev sunucusu (Ubuntu + Docker) için üç olası rol:

1. **Depolama düğümü (birincil rol):** 259GB ham + 172GB ara + nihai artefaktlar için
   disk. Laptop diski yetmez → ham veri + ara embedding burada durur. NFS/SMB veya doğrudan
   sunucuda çalıştır.
2. **Kalıcı servis (self-host):** Nihai indeks + FastAPI+Gradio container'ı 7/24 burada
   koşabilir (HF free-tier uyku/soğuk-başlangıç sınırı yok). Docker zaten kurulu → v1
   `deploy/Dockerfile` doğrudan çalışır. Ev IP'si + reverse proxy (Caddy/nginx) veya
   Cloudflare Tunnel ile dışa aç.
3. **Embed işçisi (yalnız GPU varsa):** §2 — GPU yoksa bu rol laptop'ta kalır.

**Öneri:** Ev sunucusu = **depolama + kalıcı servis** (rol 1+2); embed laptop GPU'sunda.
Bu, HF free-tier'ın kalıcı-disk/uyku sınırlarını (aşağıda) baypas eden en gerçekçi kurulum.

---

## 4. HF Spaces free-tier limitleri (dağıtım kısıtı)

v1 zaten HF Docker Space'te canlı. v2 için free-tier sınırları (2026 free CPU Basic):

- **RAM:** ~16 GB. v2a/v2b indeks (IVFPQ ~0.5GB) + model (~2.1GB) → **sığar** (v1 zaten 2.2GB).
- **Disk (ephemeral):** ~50 GB. Container imajı + weights (~1.4GB) + indeks. **v2a indeks
  (~2GB) + metadata sığar.** Ama **v2b'nin ham 259GB'ı ASLA HF'ye gitmez** — yalnız nihai
  **~0.5GB IVFPQ indeksi** gider (HF **dataset repo**'dan çekilir, v1 entrypoint deseni).
- **Kalıcı disk:** free-tier'da YOK (ücretli). → indeks her boot'ta HF dataset'ten indirilir
  (v1 `entrypoint.sh` + `hf_hub_download` deseni zaten bunu yapıyor).
- **Uyku:** free Space idle'da uyur → soğuk başlangıç. Kalıcı 7/24 için ev sunucusu (§3).
- **GPU:** free-tier GPU YOK → HF'de yalnız **CPU inference** (v1'de zaten böyle, thread-pin
  fix'li). CPU /predict ~2.6s (v1 ölçümü) → kabul edilebilir.

**Sonuç:** HF free-tier v2a/v2b **inference/servis** için yeterli (indeks küçük, IVFPQ);
ama **veri hazırlama (259GB indir + embed)** ASLA HF'de olmaz → laptop + ev sunucusunda
yapılır, yalnız nihai küçük indeks HF dataset'e yüklenir.

---

## 5. Uçtan uca v2b altyapı akışı (özet diyagram)

```
[OSV5M HF]  --snapshot_download-->  [ev sunucusu diski: 259GB ham]
                                          |
                          (batch: indir→embed→flush→sil ham)
                                          v
[laptop RTX4060 GPU]  --embed ~64h checkpoint'li-->  [172GB ara embedding]
                                          |
                              (PCA-512 fit [OSV5M-alt] + IVFPQ train)
                                          v
                              [nihai ~0.5GB IVFPQ indeks + SQLite]
                                    /                       \
                    (HF dataset repo'ya yükle)      (ev sunucusunda tut)
                            |                                 |
                  [HF Docker Space: CPU servis]     [ev sunucusu: 7/24 servis]
                   (uyku var, kalıcı disk yok)       (Docker, uyku yok)
```

---

## 6. Faz 4 özeti (onay noktası)

- **Disk:** v2b tepe ihtiyaç ~430GB (259 ham + 172 ara) → laptop'a sığmaz → **ev sunucusu
  diski zorunlu**. v2a (~500k) ~42GB → laptop'a sığar (v2a'nın bir avantajı daha).
- **Embed süresi:** v2a ~6 saat (tek oturum); v2b ~64 saat (checkpoint'li dirençli job,
  laptop GPU). **CPU ile v2b imkânsız (~33 gün)** → ev sunucusu GPU'suzsa embed laptop'ta.
- **Ev sunucusu rolü:** **depolama + kalıcı 7/24 servis** (Docker, v1 Dockerfile çalışır);
  embed GPU'lu değilse laptop'ta.
- **HF free-tier:** inference için yeterli (IVFPQ ~0.5GB indeks, CPU); ham veri/embed ASLA
  HF'de olmaz — yalnız nihai küçük indeks HF dataset'e. Kalıcı 7/24 isteniyorsa ev sunucusu.

---

## 7. DÜZELTME — disk gerçekliği (kullanıcı yakaladı)

§1/§6'daki **~430GB tepe** rakamı, ev sunucusu (**Casper Nirvana, 512GB NVMe toplam**)
kapasitesinin **%84'ü**. Üzerinde zaten Docker/Portainer koşan bir sistemde bu **çok dar
marj**. Üç netleştirme:

### (1) Gerçek boş disk ÖLÇÜLMELİ — varsayım YOK
Ben (asistan) sunucunun boş alanını **ölçemem** (erişimim yok). "512GB toplam" ≠ "430GB boş":
OS + Docker imajları/volume'ları + Portainer + mevcut container'lar zaten yer kaplıyor.
Tipik bir Ubuntu+Docker kurulumu 20-60GB+ tutabilir; imajlar birikirse çok daha fazla.
**Kullanıcının çalıştırması gereken (ölçüm, varsayım değil):**
```bash
df -h /               # kök/veri bölümünde GERÇEK boş alan
docker system df      # Docker'ın kapladığı imaj/volume/cache
```
Bu iki çıktı olmadan v2b'nin sunucuda mümkün olup olmadığı **bilinemez**. Faz 3'ün
"ölçümsüz sayı yayınlanmaz" kuralı burada da geçerli: **430GB "sığar/sığmaz" kararı ölçüme
bağlı, şimdilik AÇIK.**

### (2) Akış deseninin arıza modu + güvenlik payı
`indir → embed → flush → ham sil` deseninin tehlikesi: **bir adım hata verir/kesilirse
"ham sil" adımı atlanır → indirilen shard'lar birikir → disk dolar → sonraki yazma
başarısız → pipeline + potansiyel olarak Docker/OS bozulur** (tam dolu diskte servis
container'ları da yazamaz). Gerekli önlemler:
- **Sabit tampon (headroom) eşiği:** her batch öncesi `df` kontrolü; boş alan < örn. **50GB**
  ise DUR (yazmaya başlama), asla diski son bayta kadar doldurma.
- **İşlem-sonu temizlik garantisi:** ham shard silme `try/finally` (veya trap) içinde —
  embed hata verse bile shard silinir; "başarı" beklemez.
- **İdempotent/kaldığı-yerden:** her shard'ın embedding'i yazıldıktan SONRA ham silinir;
  kesinti sonrası zaten-işlenmiş shard'lar atlanır (çift indirme yok, disk şişmez).
- **Ayrı bölüm/kota:** mümkünse indirme dizinini Docker/OS bölümünden AYRI bir mount'a koy
  → pipeline diski doldursa bile sistem/servis bölümü korunur.

### (3) Tepe aslında DÜŞÜRÜLEBİLİR — 430GB pesimist
§1'deki 430GB, **259GB ham VE 172GB ara embedding'in aynı anda diskte durduğu** en kötü
hâli varsayar. Gerçekte **ikisi de gerekmez:**
- **172GB ara embedding materyalize edilmeyebilir.** İki-geçişli akış: (geçiş-1) küçük
  altküme embed'inden PCA + IVFPQ eğit; (geçiş-2) her batch'i embed→PCA-512→PQ-kodla→indekse
  ekle, **ham float32 embedding'i diske yazMA** (yalnız ~0.5GB nihai kod büyür). → 172GB
  ara **elenir**.
- **259GB ham akışlı indirilir**, shard shard silinir → ham tepe = tek shard (birkaç GB) +
  birikmiş indeks.
- Böylece **gerçekçi tepe: birkaç-GB ham shard + ~0.5GB büyüyen indeks + ~kısa süreli
  PCA/PQ eğitim altkümesi (~birkaç GB) ≈ 10-30GB civarı** olabilir (259GB'ı bir kez indirip
  saklamak yerine akışlarsak). **AMA:** bu, OSV5M'in shard-shard indirilip aradan
  silinebilmesine bağlı (snapshot_download tam-set çekme eğiliminde → şard-seçici indirme
  veya `load_dataset` streaming gerekir; doğrulanmalı).

### Koşullu sonuç (kullanıcının istediği netlik)
- **Eğer** ölçülen boş alan **≥ ~430GB** → naif akış (ham+ara birlikte) bile sığar (dar marj,
  §2 önlemleriyle).
- **Eğer** boş alan **~430GB'ın ALTINDA** (Casper 512GB toplamda çok olası) → **naif v2b
  İMKÂNSIZ.** İki seçenek kalır: **(a)** iki-geçişli akış (§7.3) ile tepeyi ~10-30GB'a indir
  → v2b yine mümkün olabilir (ama snapshot-streaming doğrulaması + §2 güvenlik payı şart);
  **(b)** v2b'den vazgeç, **yalnız v2a (~500k, ~42GB) gerçekçi** — ki zaten **laptop'a bile
  sığıyor**, ev sunucusu depolama için bile şart değil.
- **En dürüst özet:** v2b'nin bu sunucuda mümkünlüğü **ölçüme + streaming-indirme
  doğrulamasına bağlı, şu an KANITLANMAMIŞ**. **v2a her hâlükârda gerçekçi ve güvenli**
  (laptop-yerel). v2b'yi ancak `df -h` + `docker system df` + snapshot-streaming testi sonrası
  taahhüt et. Şüphede kalınırsa **v2a ile başla** (v1 kuralı: v3'e/aşırıya sıçrama yok).

**Sonraki: Faz 5 — yasal/lisans derinleştirme** (Mapillary ToS toplu indirme; CC-BY-SA
türev-indeks/embedding ShareAlike viralliği; atıf yükümlülükleri). Onayınızı bekliyorum.

---

### Kaynaklar
- OSV5M, arXiv:2404.18873 — 259GB / 5.1M (Faz 1).
- v1 ölçümleri (MEMORY): GPU ~20-25 img/s, CPU ~1.8 img/s embed; model 2.1-2.2GB RAM;
  CPU /predict ~2.6s; HF Docker Space entrypoint hf_hub_download deseni.
- Hugging Face Spaces dokümantasyonu — free CPU Basic ~16GB RAM / ~50GB ephemeral disk,
  kalıcı disk ücretli, free-tier GPU yok, idle uyku.
- FAISS — IVFPQ indeks boyutu (m=64 → 64B/vektör).
