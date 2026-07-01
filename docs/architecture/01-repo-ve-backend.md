# MİM-1a: Repo Yapısı + Backend Mimarisi

> Plan/dokümantasyon. Uygulama kodu içermez.

## Başlık 1 — Repo / Proje Yapısı

### Monorepo vs Ayrı Repo
| Kriter | Monorepo | Ayrı repo |
|---|---|---|
| Solo geliştirici | ✓ Tek kaynak, atomik commit | Koordinasyon yükü |
| Portföy (tek link) | ✓ Tek vitrin | Dağınık |
| CI/dokümantasyon | ✓ Tek pipeline | Tekrar |
| MVP (Gradio=Python) | ✓ Frontend minimal | Gereksiz ayrım |
| Bağımsız deploy | Orta | ✓ |

**Öneri: Monorepo.** Solo dev + portföy (tek etkileyici link) + MVP'de Python frontend. v2 Next.js `frontend/` alt-klasörü olarak eklenir.

### Klasör Ağacı (plan)
```
geolens/
├── README.md · LICENSE (MIT) · .gitignore · .env.example
├── pyproject.toml · docker-compose.yml · Dockerfile · Makefile
├── docs/{research/, architecture/, assets/}
├── backend/app/
│   ├── main.py                    # FastAPI + lifespan
│   ├── api/{routes_predict, routes_health, deps}
│   ├── core/{config, logging, exceptions}
│   ├── services/{embedding, retrieval, xai, prediction}
│   ├── repositories/metadata_repo
│   ├── schemas/{prediction, common}
│   └── ml/{backbone(DINOv2), aggregator(SALAD), model_registry}
├── frontend/gradio_app/           # MVP; v2 → Next.js
├── data/{raw/, processed/, index/}
├── models/weights/
├── scripts/{download_msls, build_index, evaluate}
├── tests/{unit/, integration/, conftest}
└── .github/workflows/ci.yml
```

### Config / Secrets
- `.env.example` commit; `.env` gitignore.
- `core/config.py` → **Pydantic Settings** (env→tiplenmiş config).
- HF Spaces Secrets UI; yerelde `.env`.
- Gerekçe: tek kaynak, tip güvenli, test override kolay.

---

## Başlık 2 — Backend Mimarisi (FastAPI)

### Katmanlı vs Tek-katman
| | Tek-katman | Katmanlı |
|---|---|---|
| Test | Zor | ✓ İzole |
| ML state (GPU/FAISS) | Dağınık | ✓ Kapsüllü |
| Boilerplate | Az | Orta |
| 3-6 ay bakım | Riskli | ✓ |

**Öneri: Hafif katmanlı** (router→service→repository). Router=HTTP/validasyon; Service=iş mantığı; Repository=yalnız SQLite. FAISS/model service'e gömülür (repository'ye değil — tek tüketici).

### Router (`/api/v1`)
`POST /predict` · `GET /health` · `GET /model-info`

### Model + FAISS Yükleme
| Yaklaşım | Değerlendirme |
|---|---|
| `@app.on_event("startup")` | Deprecated ✗ |
| **lifespan + app.state** | ✓ Modern, tek yükleme |
| Global singleton | Lifecycle/test zayıf |

**Öneri: lifespan + app.state.** Model+FAISS başlangıçta bir kez yüklenir, sıcak tutulur (risk #21), DI ile service'lere geçer.

### Hata Yönetimi
`core/exceptions.py`: `GeoLensError`(base) → `InvalidImageError`, `LowConfidenceError`, `ModelNotReadyError`. FastAPI `exception_handler` → standart JSON. Boundary'de Pydantic + MIME/boyut (risk #27).

### Config
Pydantic Settings — tiplenmiş/doğrulanmış env, `.env`+override, test mock kolay.

---

### MİM-1a Özeti
- Monorepo + katmanlı FastAPI (router/service/repository).
- lifespan ile model/FAISS sıcak yükleme; Pydantic Settings config; hiyerarşik exception.
