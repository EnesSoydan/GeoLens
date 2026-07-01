# MİM-3a: Frontend Mimarisi

> Plan/dokümantasyon. Uygulama kodu içermez.

## MVP: Gradio Bileşen Ağacı
```mermaid
graph TD
    App["Gradio Blocks: GeoLens"] --> Header["Header: başlık + açıklama"]
    App --> Body["Row"]
    Body --> L["Column: Girdi"]
    Body --> R["Column: Sonuç"]
    L --> IMG["gr.Image (upload)"]
    L --> TK["gr.Slider top_k"]
    L --> HM["gr.Checkbox include_heatmap"]
    L --> BTN["gr.Button 'Tahmin Et'"]
    R --> MAP["gr.HTML: folium/Leaflet harita"]
    R --> HEAT["gr.Image: Eigen-CAM overlay"]
    R --> TBL["gr.Dataframe: rank/lat/lon/similarity"]
    R --> CONF["gr.Label: güven + is_confident uyarı"]
    BTN -.fn.-> HTTP["POST /api/v1/predict"]
```

## Gradio ↔ Backend Bağlantısı
| Yaklaşım | Değerlendirme |
|---|---|
| Service in-process çağrı | Hızlı; UI'yı iç API'ye bağlar, sözleşme test edilmez |
| **Gradio → FastAPI HTTP (localhost)** | ✓ Gerçek sözleşme; v2 Next.js drop-in; overhead ihmal |
| Ayrı process/port | Deploy karmaşıklığı |

**Öneri:** Gradio, FastAPI'ye HTTP ile bağlanır (aynı container; API `/api/v1`, UI `/ui` mount). Aynı sözleşme MVP+v2; full-stack sergisi; `/docs` vurgusu.

> Gradio'da yerleşik harita yok → **folium** (Leaflet) → `gr.HTML`. Alternatif: plotly scattermapbox / gradio_leaflet.

## v2: Next.js Sayfa/Komponent Haritası
```mermaid
graph TD
    Home["/ (ana sayfa)"] --> Uploader["ImageUploader"]
    Home --> Result["ResultPanel"]
    Result --> Map["MapView (react-leaflet)"]
    Result --> Heat["HeatmapOverlay"]
    Result --> List["PredictionList"]
    Result --> Badge["ConfidenceBadge"]
    About["/about (metodoloji)"]
    Uploader -.POST /api/v1/predict.-> API[("FastAPI")]
```

## Leaflet Entegrasyonu (veri akışı)
`predictions[]` (lat/lon) → top-K marker, top-1 vurgu, bounds fit. Popup: referans küçük resim + similarity. Aliasing'de (risk #13) top-K yayılımı görünür. `is_confident=false` → "düşük güven" mesajı, heatmap yok (MİM-2 ile tutarlı).

### MİM-3a Özeti
Gradio Blocks (folium/Leaflet) → FastAPI HTTP; v2 Next.js react-leaflet drop-in; aynı API sözleşmesi.
