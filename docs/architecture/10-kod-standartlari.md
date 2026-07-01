# MİM-1b: Kod Standartları ve Konvansiyonlar

> Plan/dokümantasyon. Uygulama kodu/config içeriği içermez (Development fazına ait).

| Konu | Karar | Gerekçe |
|---|---|---|
| Lint + Format | **Ruff** (Black+isort+flake8 yerine tek araç) | En hızlı, tek config kaynağı |
| Tip denetimi | **mypy** (public API zorunlu) | Erken hata yakalama |
| Docstring / satır | Google style / 100 karakter | Okunabilirlik |
| Commit | **Conventional Commits** (feat/fix/docs/refactor/test/chore) | Otomatik changelog, okunur geçmiş |
| Branch | Trunk-based hafif: korumalı `main` + kısa `feat/*`,`fix/*` + PR | Solo dev'e GitFlow aşırı |
| İsimlendirme | snake_case dosya/fonksiyon, PascalCase sınıf, tekil modül | PEP8 |
| Pre-commit | ruff + mypy + whitespace hook | CI öncesi kalite kapısı |
| CI | GitHub Actions: lint → type → test | Otomatik doğrulama |
| Test | pytest (unit/integration ayrımı), coverage hedefi | Güvenilirlik + portföy sinyali |

## Alternatifler (kısa)
- **Ruff vs Black+isort+flake8:** Ruff hepsini tek hızlı araçta birleştirir → seçildi.
- **Trunk-based vs GitFlow:** GitFlow çok dallı, solo/MVP'ye ağır → trunk-based hafif seçildi.
- **mypy vs pyright:** mypy Python-yerel, yaygın; pyright hızlı ama Node bağımlı → mypy seçildi (basitlik).

### MİM-1b Özeti
Ruff + mypy + Conventional Commits + trunk-based + GitHub Actions. Development fazına net konvansiyon temeli.
