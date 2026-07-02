# MİM-2b: API Sözleşmesi

> Plan/dokümantasyon. Gerçek Pydantic kodu değil — alan/tip/açıklama taslağı.

## Endpoint Listesi
| Method | Path | Amaç |
|---|---|---|
| POST | `/api/v1/predict` | Görüntü → top-K konum + XAI heatmap |
| GET | `/api/v1/health` | Liveness/readiness |
| GET | `/api/v1/model-info` | Model + index sürümü |

## POST /predict — Request (multipart/form-data)
| Alan | Tip | Açıklama |
|---|---|---|
| image | file | Sorgu görüntüsü (JPEG/PNG, ≤ N MB) |
| top_k | int? | Eşleşme sayısı (varsayılan 5) |
| include_heatmap | bool? | XAI overlay (varsayılan true) |

## POST /predict — Response (application/json)
| Alan | Tip | Açıklama |
|---|---|---|
| query_id | str(uuid) | İzleme kimliği |
| top_prediction | obj{lat,lon,similarity} | En iyi tahmin |
| predictions | array | {rank, lat, lon, similarity, reference_image_id, sequence_id} |
| confidence | float | Top-1 kosinüs benzerliği |
| is_confident | bool | Eşik üstü mü (risk #9) |
| heatmap_png_base64 | str? | Eigen-CAM overlay, **base64 gömülü PNG**. Yalnızca `include_heatmap=true` **VE** `is_confident=true` iken dolu; aksi halde `null` |
| heatmap_status | str | `"included"` / `"skipped_low_confidence"` / `"disabled"` — null nedenini belgeler |
| model_version | str | Manifest hash/sürüm |
| processing_ms | int | İşleme süresi |

> **Heatmap kararı:** base64 gömülü PNG (MVP). Gerekçe: Gradio tek request-response; statik dosya sunumu/temizleme gereksiz karmaşıklık. v2 Next.js'te `heatmap_url` alternatifi eklenir.
>
> **Düşük güvende heatmap üretilmez:** `is_confident=false` iken `include_heatmap=true` olsa bile Eigen-CAM atlanır (`heatmap_png_base64=null`, `heatmap_status="skipped_low_confidence"`). Gerekçe: güvenilmez eşleşmenin XAI açıklaması yanıltıcı olur (risk #17) ve boşa GPU harcamaz (risk #19). Bu, `03-ai-pipeline.md` flowchart/sequence ile tutarlıdır.
>
> **GPU semaphore:** heatmap üretimi (Eigen-CAM) ve embedding, `03-ai-pipeline.md`'de tanımlı **tek global paylaşılan `gpu_semaphore(1)`** üzerinden serileştirilir.

## GET /model-info — Response
| Alan | Tip | Örnek |
|---|---|---|
| backbone | str | "DINOv2 ViT-B/14" |
| aggregator | str | "SALAD" |
| embedding_dim | int | 8448 |
| index_type | str | "FAISS IndexFlatIP" |
| index_size | int | Resmi MSLS-val database sayısı (cph+sf; ~18.9k) |
| dataset | str | "MSLS-val (cph+sf)" |
| model_hash | str | manifest hash |
| cities | list[str] | ["cph", "sf"] |

## GET /health — Response
| Alan | Tip |
|---|---|
| status | str |
| model_ready | bool |
| index_ready | bool |
| uptime_s | int |

## Hata Kodları (standart format)
Gövde: `{ "error": { "code": str, "message": str, "detail": obj? } }`

| HTTP | Kod | Ne zaman |
|---|---|---|
| 400 | INVALID_IMAGE | Bozuk/desteklenmeyen dosya |
| 413 | FILE_TOO_LARGE | Boyut sınırı (risk #27) |
| 422 | VALIDATION_ERROR | Pydantic hatası |
| 503 | MODEL_NOT_READY | Startup/hash uyuşmazlığı |
| 500 | INTERNAL_ERROR | Beklenmeyen |

> Düşük güven **hata değildir**: `is_confident=false` ile 200 döner (OOD'da bile en iyi tahmin; risk #9/#13).

## OpenAPI / Swagger
FastAPI otomatik `/docs` (Swagger) + `/redoc`. Şemalar Pydantic'ten; router `tags`; her endpoint `summary`+`description`+response `examples` (portföy vurgusu).

### MİM-2b Özeti
3 endpoint, base64 heatmap'li predict sözleşmesi, standart hata zarfı, düşük-güven=200, otomatik OpenAPI.
