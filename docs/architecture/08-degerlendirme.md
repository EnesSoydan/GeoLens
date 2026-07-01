# MİM-4b: Değerlendirme Protokolü

> Plan/dokümantasyon. Kod değil — adım planı.

## Protokol Kararı: Resmi mapillary_sls MSLS-val
| Alternatif | Değerlendirme |
|---|---|
| **Resmi mapillary_sls MSLS-val (740 sorgu, ~18.9k database; cph+sf)** | ✓ Literatürle (NetVLAD/SALAD/BoQ) DOĞRUDAN karşılaştırılabilir GT+split (tam resmi val) |
| Gayri-resmi VPR-datasets-downloader (~11k sorgu) | ✗ Daha büyük ama literatürle karşılaştırılamaz (protokol farkı skoru bozar) |

**Karar: Resmi protokol.** Benchmark'ın amacı NetVLAD baseline karşılaştırması → yalnız resmi GT/split literatürle kıyaslanabilir. Gayri-resmi 11k, model değil protokol farkını yansıtıp karşılaştırmayı geçersiz kılar.

**DÜZELTME (2026-07):** Resmi `mapillary_sls` `default_cities['val'] == ['cph','sf']` (Kopenhag + San Francisco) — Amsterdam/Manila TRAIN şehirleridir. Önceki "val = Amsterdam+Manila" ifadesi faktüel hataydı (740 sorgu / ~18.9k database sayıları doğruydu, yalnız şehir etiketleri yanlıştı). MVP artık **tam resmi MSLS-val'i (cph+sf)** kullanır → alt-küme hesabı gerekmez. Bu, sonuçlarımızı **yayınlanmış tam-val sayılarıyla doğrudan** kıyaslanabilir kılar; ek olarak **NetVLAD baseline'ı aynı resmi val'de kendimiz çalıştırırız** (apples-to-apples + literatür teyidi).

## Recall@N + Medyan km-hata
**Adımlar (`scripts/evaluate.py`):**
1. `image_role='query'` görüntüler index DIŞINDA (MİM-3 leakage önlemi — ön koşul).
2. Her sorgu: embed → FAISS top-N (N=1,5,10), yalnız `database` (cph+sf) index.
3. **GT (resmi MSLS eşiği):** bir database görüntüsü sorgu için pozitif sayılır ⇔ **≤25 m mesafe VE ≤40° bakış-açısı (heading) farkı — ikisi birlikte**. Top-N içinde ≥1 pozitif varsa "doğru@N".
4. **Recall@N** = en az bir doğru eşleşmeli sorgu oranı.
5. **Medyan km-hata:** top-1 database GPS ile sorgu GPS arası haversine → medyan (+ortalama).
6. N=1/5/10 raporla.

## README Benchmark Tablosu (şablon)
> Her iki satır da **aynı resmi MSLS-val'de (cph+sf), resmi protokolle** hesaplanır.

| Model | R@1 | R@5 | R@10 | Medyan km-hata |
|---|---|---|---|---|
| DINOv2+SALAD (bizim) | – | – | – | – |
| NetVLAD (baseline, kendi çalıştırmamız) | – | – | – | – |

> `is_confident` güven eşiği bu adımda ampirik kalibre edilir. Tam resmi val (cph+sf) kullanıldığından yayınlanmış tam-val sayıları meşru referanstır (istenirse tabloya "literatür" satırı olarak eklenebilir).

### MİM-4b Özeti
Resmi mapillary_sls MSLS-val protokolü (740 sorgu, cph+sf); GT eşiği **25 m VE ≤40°**; query index dışında (leakage yok); NetVLAD baseline aynı val'de kendimiz çalıştırılır.
