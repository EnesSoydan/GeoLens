# Faz 5 — Yasal / Lisans Derinleştirme (v2)

> **Görev tipi:** Araştırma + planlama (kod değil). **Hukuki tavsiye DEĞİL** — açık
> kaynaklara dayalı risk haritası; ticari/resmi kullanımda hukuk danışmanı önerilir.
> **Amaç:** v2 veri ölçeklendirmesinin (OSV5M + Mapillary API + KartaView) lisans/ToS
> yükümlülüklerini netleştirmek; en kritik açık soruyu (CC-BY-SA türev-indeks viralliği)
> derinleştirmek; somut uyum eylemleri çıkarmak.
> **Kaynak-doğruluk:** v1 kararları (repo MIT, SALAD GPL-3.0 çalışma-zamanı beyanı) korunur.

---

## 0. Üç risk ekseni

1. **Mapillary ToS — toplu indirme/API kullanımı** (canlı yoğunlaştırma yolu).
2. **CC-BY-SA-4.0 — ShareAlike (SA):** türev embedding-indeksi "adaptation" mı? Dağıtımı
   viral mi? (v2'nin EN kritik açık sorusu.)
3. **CC-BY-SA-4.0 — Attribution (BY):** atıf yükümlülüğü (Mapillary logo + link).

---

## 1. Mapillary ToS + API kullanımı

- **Görüntü lisansı:** Mapillary görüntüleri **CC-BY-SA-4.0**. Tüm kullanımlar (ticari
  dahil) ücretsiz; atıf = Mapillary logosu + görüntü sayfasına link.
- **API erişimi:** OAuth2 token, `graph.mapillary.com`. bbox < 0.01°² (16 Oca 2026),
  ~2000 feature/istek. Resmi `mapillary-python-sdk` büyük alanlar için önerilir (Faz 1 notu).
- **Toplu indirme:** Mapillary ToS API'yi normal kullanım için açık bırakır; ancak
  **aşırı/otomatik kitlesel çekim** rate-limit/erişim kısıtına takılabilir (yayınlanmış
  sayısal limit yok → SDK'nın kendi hız yönetimine uy, agresif paralelleştirme yapma).
  v1'de MSLS (Mapillary türevi) zaten bu ekosistemden geldi — hedefli şehir yoğunlaştırması
  ToS sınırları içinde makul kullanımdır.
- **Risk:** düşük-orta. **Eylem:** SDK + makul hız + atıf. OSV5M birincil olduğu için
  Mapillary API yalnız *hedefli* yoğunlaştırmada → kitlesel çekim baskısı yok.

---

## 2. CC-BY-SA ShareAlike — türev indeks viralliği (EN KRİTİK, AÇIK)

**Soru:** OSV5M/Mapillary görüntülerinden ürettiğimiz **SALAD embedding'leri + FAISS
indeksi**, CC-BY-SA-4.0 anlamında bir **"adaptation" (türev eser)** midir? Öyleyse,
dağıttığımız indeks de **CC-BY-SA ile lisanslanmak** (ShareAlike viralliği) zorunda kalabilir.

### İki yorum (ikisi de savunulabilir — bu yüzden AÇIK)
- **"Türev DEĞİL" yorumu:** Embedding, görüntünün *ölçülmüş bir istatistiği/özniteliği*
  (feature vector) — telif korumalı ifadeyi yeniden üretmez, görüntüyü rekonstrükte etmez.
  Metin/veri madenciliği çıktısına benzer → SA tetiklenmeyebilir. (CC'nin kendi SSS'i
  "teknik olarak eseri kopyalamayan analiz" için esneklik ima eder ama **kesin hüküm vermez**.)
- **"Türev OLABİLİR" yorumu:** İndeks, kaynak görüntü kümesinden **doğrudan türetilmiş**
  ve onsuz var olamaz; "adaptation" tanımı geniş (CC-BY-SA-4.0 §1(a): "translated, altered,
  arranged, transformed, or otherwise modified"). Konservatif okuma → SA uygulanır → indeks
  de CC-BY-SA paylaşılmalı.

### Neden v1'deki GPL kararına benzer
v1'de "dağıtmıyoruz → risk düşük" argümanı kullanıcı tarafından **YANLIŞ** bulunmuştu:
HF Spaces'e deploy = **kamuya sunum = dağıtım sayılabilir**. Aynı mantık burada da geçerli:
indeks HF dataset repo'ya yüklenip Space'ten servis edilirse **kamuya dağıtım** olur → SA
tetikleyebilir. "İndeksi paylaşmıyoruz, yalnız API sonucu dönüyoruz" savı da zayıf çünkü
dataset repo'daki indeks dosyası **indirilebilir** olacak.

### Konservatif duruş (öneri)
- **İndeksi dağıtacaksan** (HF dataset public), onu da **CC-BY-SA-4.0 ile lisansla** +
  kaynak atfı ekle → SA'yı ihlal etmek yerine ona *uy*. Bu, GPL'deki "nötr dille beyan et"
  duruşunun veri karşılığı: riski kabul etmek yerine lisansa uyarak nötralize et.
- **Alternatif:** indeksi public dağıtma; yalnız ev sunucusunda/private tut, Space'e private
  çek (private dataset + HF_TOKEN — v1 entrypoint zaten destekliyor). Dağıtım olmazsa SA
  sorusu büyük ölçüde etkisiz. **Ama** Space'in kendisi çıktı ürettiği için görüntülerin BY
  (atıf) yükümlülüğü yine kalır (§3).
- **Ticari/resmi kullanımda hukuk danışmanı** (v1 ile aynı). Bu belge hukuki görüş değildir.

---

## 3. Attribution (BY) — her senaryoda zorunlu

SA sorusu nasıl çözülürse çözülsün, **atıf her hâlükârda gerekli**:
- **OSV5M:** CC-BY-SA-4.0 → veri seti + Mapillary kaynağına atıf (arXiv:2404.18873 + Mapillary).
- **Mapillary:** görüntü başına Mapillary logosu + görüntü sayfası linki (canlı yoğunlaştırma
  görüntüleri için).
- **KartaView:** CC-BY-SA-4.0 atıf.
- **Somut eylem:** Gradio UI'a **görünür bir "Veri kaynakları & lisans" bölümü** (Mapillary
  logosu + OSV5M/arXiv linki + CC-BY-SA-4.0 ibaresi); README'de aynı. v1'de zaten SALAD
  GPL beyanı deseni var → onun yanına veri atıf bloğu eklenir.

---

## 4. Uyum eylem listesi (v2)

| # | Eylem | Ne zaman | Kaynak |
|---|---|---|---|
| 1 | Mapillary API: SDK + makul hız, agresif paralel yok | yoğunlaştırma | Mapillary ToS |
| 2 | İndeks dağıtılacaksa CC-BY-SA-4.0 lisansla **veya** private tut | dağıtımdan önce | CC-BY-SA §1/SA |
| 3 | Gradio UI + README: görünür veri-kaynağı & atıf bloğu (Mapillary logo+link, OSV5M, CC-BY-SA) | UI/deploy | CC-BY BY |
| 4 | SALAD GPL-3.0 beyanı korunur (v1) | mevcut | v1 kararı |
| 5 | Ticari/resmi kullanım → hukuk danışmanı | ticarileşmeden önce | genel |

---

## 5. Faz 5 özeti (onay noktası)

- **Mapillary ToS:** düşük-orta risk; SDK + makul hız + atıf ile uyumlu (OSV5M birincil →
  kitlesel çekim baskısı yok).
- **CC-BY-SA ShareAlike (EN KRİTİK, AÇIK):** embedding-indeksinin "türev eser" olup olmadığı
  belirsiz; iki yorum da savunulabilir. **Konservatif duruş:** ya indeksi CC-BY-SA ile
  dağıt (SA'ya uy), ya da private tut (dağıtma). v1 GPL kararıyla aynı mantık — "deploy =
  dağıtım olabilir". Hukuki görüş değil; ticari kullanımda danışman.
- **Attribution (BY):** her senaryoda zorunlu → Gradio UI + README'ye görünür veri-kaynağı
  & lisans bloğu (Mapillary logo+link, OSV5M/arXiv, CC-BY-SA-4.0).
- **5 maddelik uyum eylem listesi** çıkarıldı.

**Bu, planlanan 5 araştırma fazının SONUNCUSU.** Sonraki adım: **tüm fazları sentezleyen
kademeli v2 yol haritası + dürüst donanım-gerçekçiliği değerlendirmesi** (final belge).
Onayınızla onu yazacağım.

---

### Kaynaklar
- Creative Commons CC-BY-SA-4.0 yasal metni — §1(a) adaptation tanımı, §3 ShareAlike.
- Creative Commons SSS — veri/analiz çıktılarının türev sayılıp sayılmaması (kesin hüküm yok).
- Mapillary ToS + Yardım — görüntü lisansı CC-BY-SA, atıf (logo+link), API kullanımı.
- OSV5M, arXiv:2404.18873 — CC-BY-SA-4.0, Mapillary kaynağı (Faz 1).
- v1 kararı (MEMORY): SALAD GPL-3.0 nötr beyan; "HF deploy = dağıtım olabilir" ilkesi.
