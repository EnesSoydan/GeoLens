# MİM-4b: Değerlendirme Protokolü

> Plan/dokümantasyon. Kod değil — adım planı.

## Protokol Kararı: Resmi mapillary_sls MSLS-val
| Alternatif | Değerlendirme |
|---|---|
| **Resmi mapillary_sls MSLS-val (750 sorgu, ~18.9k database; cph+sf)** | ✓ Literatürle (NetVLAD/SALAD/BoQ) DOĞRUDAN karşılaştırılabilir GT+split (tam resmi val) |
| Gayri-resmi VPR-datasets-downloader (~11k sorgu) | ✗ Daha büyük ama literatürle karşılaştırılamaz (protokol farkı skoru bozar) |

**Karar: Resmi protokol.** Benchmark'ın amacı NetVLAD baseline karşılaştırması → yalnız resmi GT/split literatürle kıyaslanabilir. Gayri-resmi 11k, model değil protokol farkını yansıtıp karşılaştırmayı geçersiz kılar.

**DÜZELTME (2026-07):** Resmi `mapillary_sls` `default_cities['val'] == ['cph','sf']` (Kopenhag + San Francisco) — Amsterdam/Manila TRAIN şehirleridir. Önceki "val = Amsterdam+Manila" ifadesi faktüel hataydı. MVP artık **tam resmi MSLS-val'i (cph+sf)** kullanır → alt-küme hesabı gerekmez. Bu, sonuçlarımızı **yayınlanmış tam-val sayılarıyla doğrudan** kıyaslanabilir kılar.

**DÜZELTME-2 (2026-07, S5): sorgu sayısı ve GT eşiği kaynak-doğrulandı.**
- **Sorgu sayısı = 750** (740 değil). Resmi im2im val yalnız `query/subtask_index.csv`'de `all == True` işaretli sorguları kullanır: cph **502** + sf **248** = **750**. Kalan sorgu görüntüleri (cph 6595, sf 4525 toplamdan) resmi val'e dahil değildir. **Database tarafında** `all` sütunu TÜM satırlarda True (cph 12601, sf 6315) → `build_index.py`'nin ürettiği tam FAISS index'i zaten resmi val database'idir, ek filtre gerekmez.
- **GT eşiği = yalnız ≤25 m mesafe (bakış-açısı KOŞULU YOK).** Önceki "≤25 m VE ≤40° bakış-açısı" ifadesi **faktüel hataydı**. Resmi kaynak doğrulaması: `mapillary_sls/evaluate.py` `--threshold` varsayılanı **25** olup `MSLS(mode='val', posDistThr=25)`'a geçer; pozitifler `NearestNeighbors.radius_neighbors(utmQ, posDistThr)` ile UTM mesafesinden türer — **hiçbir yön/heading terimi yoktur**. `mapillary_sls/utils/eval.py::recall` de top-k sıralamasını yalnız bu mesafe-tabanlı pozitiflerle eşleştirir. `raw.csv`'deki `ca` (compass/heading) sütunu VAR ve doludur, ama resmi protokol onu **yalnız eğitimdeki `sideways` alt-görev ağırlıklandırmasında** kullanır — GT tanımında değil. Bu yüzden 40°'yi **sessizce mesafeye indirgemiyoruz; resmi protokolün heading kullanmadığını açıkça belgeliyoruz** ve heading filtresi uygulamıyoruz (uygulasak literatürle kıyaslanamaz hale gelirdi).
- **k değerleri = {1, 5, 10, 20}** (resmi `evaluate.py`: `ks = [1, 5, 10, 20]`).

## Recall@N + Medyan km-hata
**Adımlar (`scripts/evaluate.py`):**
1. `image_role='query'` görüntüler index DIŞINDA (MİM-3 leakage önlemi — ön koşul).
2. Sorgu kümesi: `subtask='all'` (`subtask_index.csv all==True`) → 750 sorgu (cph+sf).
3. Her sorgu: embed → FAISS top-K (K=20), yalnız `database` (cph+sf) index.
4. **GT (resmi MSLS eşiği):** bir database görüntüsü sorgu için pozitif sayılır ⇔ **≤25 m mesafe** (yalnız GPS; heading yok). Top-k içinde ≥1 pozitif varsa "doğru@k".
5. **Recall@k** = en az bir doğru eşleşmeli sorgu oranı (k=1/5/10/20). Pozitifi olmayan sorgular paydadan düşer (resmi davranış).
6. **Medyan km-hata:** top-1 database GPS ile sorgu GPS arası haversine → medyan (+ortalama). Bu bizim ek teşhis metriğimizdir, resmi metriğin parçası değildir.

## README Benchmark Tablosu (şablon)
> "Bizim" satır(lar)ımız resmi MSLS-val'de (cph+sf, 750 sorgu, resmi protokol, GT=25 m) `scripts/evaluate.py` ile hesaplanır. NetVLAD satırı **literatürden alıntıdır** (aşağıya bakınız).

| Model | R@1 | R@5 | R@10 | R@20 | Medyan km-hata |
|---|---|---|---|---|---|
| DINOv2+SALAD (bizim, `evaluate.py`) | – | – | – | – | – |
| NetVLAD (baseline, literatür) | – | – | – | – | — |

**NetVLAD baseline kaynak KARARI (S5):** Kendi NetVLAD'ımızı MSLS-train üzerinde eğitip değerlendirmek MVP kapsamı dışında ve hata-eğilimli (ayrı eğitim + eval altyapısı). Bunun yerine **literatürden alıntı** yapılır; kaynak README'de NET belirtilir: kaynaksız "NetVLAD ~X" yasak.
- **Atıf kaynağı:** Sergio Izquierdo, Javier Civera — *"Optimal Transport Aggregation for Visual Place Recognition" (SALAD)*, CVPR 2024. Kullandığımız modelin (DINOv2+SALAD) makalesi olduğu için aynı tablo hem baseline'ı hem hedef modeli aynı MSLS-val protokolünde raporlar → en tutarlı atıf. README'de tam tablo/satır numarası ile belirtilecek (NetVLAD MSLS-val R@1/R@5/R@10 satırı). Sayılar makale değerleridir, bizim ölçümümüz değildir; tabloda "(literatür)" ile işaretlenir.
- Alternatif kabul edilebilir kaynak: NetVLAD'ın MSLS-val sayılarını raporlayan başka bir hakemli VPR makalesi (ör. MSLS/BoQ tabloları) — hangisi seçilirse README'de makale + tablo açıkça yazılır.

> `is_confident` güven eşiği bu adımda ampirik kalibre edilir.

### MİM-4b Özeti
Resmi mapillary_sls MSLS-val protokolü (**750** sorgu, cph+sf, `subtask='all'`); GT eşiği **yalnız ≤25 m mesafe** (heading YOK — resmi protokol kullanmıyor); Recall@{1,5,10,20} + medyan km-hata (ek teşhis); query index dışında (leakage yok); NetVLAD baseline **literatürden** (SALAD/CVPR2024 tablosu), kaynak README'de net.
