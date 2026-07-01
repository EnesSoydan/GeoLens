# MİM-4b: Değerlendirme Protokolü

> Plan/dokümantasyon. Kod değil — adım planı.

## Protokol Kararı: Resmi mapillary_sls MSLS-val
| Alternatif | Değerlendirme |
|---|---|
| **Resmi mapillary_sls MSLS-val (740 sorgu, ~18.9k database; Am+Manila)** | ✓ Literatürle (NetVLAD/SALAD/BoQ) karşılaştırılabilir GT+split; Amsterdam alt-kümesi daha küçük |
| Gayri-resmi VPR-datasets-downloader (~11k sorgu) | ✗ Daha büyük ama literatürle karşılaştırılamaz (protokol farkı skoru bozar) |

**Karar: Resmi protokol.** Benchmark'ın amacı NetVLAD baseline karşılaştırması → yalnız resmi GT/split literatürle kıyaslanabilir. Gayri-resmi 11k, model değil protokol farkını yansıtıp karşılaştırmayı geçersiz kılar.

**İncelik (Amsterdam-only kapsamı):** Resmi val Amsterdam+Manila'dır (740 sorgu). MVP yalnız Amsterdam → bunların **Amsterdam alt-kümesini** kullanır. Bu nedenle yayınlanmış tam-val sayılarını alıntılamak yerine **NetVLAD baseline'ı aynı Amsterdam alt-kümesinde kendimiz çalıştırırız** (apples-to-apples). Yayınlanmış tam-val = kaba doğrulama referansı.

## Recall@N + Medyan km-hata
**Adımlar (`scripts/evaluate.py`):**
1. `image_role='query'` görüntüler index DIŞINDA (MİM-3 leakage önlemi — ön koşul).
2. Her sorgu: embed → FAISS top-N (N=1,5,10), yalnız Amsterdam `database` index.
3. **GT (resmi MSLS eşiği):** bir database görüntüsü sorgu için pozitif sayılır ⇔ **≤25 m mesafe VE ≤40° bakış-açısı (heading) farkı — ikisi birlikte**. Top-N içinde ≥1 pozitif varsa "doğru@N".
4. **Recall@N** = en az bir doğru eşleşmeli sorgu oranı.
5. **Medyan km-hata:** top-1 database GPS ile sorgu GPS arası haversine → medyan (+ortalama).
6. N=1/5/10 raporla.

## README Benchmark Tablosu (şablon)
> Her iki satır da **aynı Amsterdam alt-kümesinde, resmi protokolle** hesaplanır.

| Model | R@1 | R@5 | R@10 | Medyan km-hata |
|---|---|---|---|---|
| DINOv2+SALAD (bizim) | – | – | – | – |
| NetVLAD (baseline, kendi çalıştırmamız) | – | – | – | – |

> `is_confident` güven eşiği bu adımda ampirik kalibre edilir. Yayınlanmış tam-val (Am+Manila) sayıları yalnız kaba referans; tabloya konmaz.

### MİM-4b Özeti
Resmi mapillary_sls MSLS-val protokolü (740 sorgu, Amsterdam alt-kümesi); GT eşiği **25 m VE ≤40°**; query index dışında (leakage yok); NetVLAD baseline aynı alt-kümede kendimiz çalıştırılır.
