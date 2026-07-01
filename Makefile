.PHONY: install lint format type test check download-msls run

install:  ## Install project + dev tooling (editable).
	pip install -e ".[dev]"

lint:  ## Ruff lint.
	ruff check .

format:  ## Ruff auto-format.
	ruff format .

type:  ## mypy static type check.
	mypy

test:  ## Run the test suite.
	pytest

check: lint type test  ## Full local quality gate (mirrors CI).

download-msls:  ## Filter the MSLS-val (cph+sf) subset (set MSLS_RAW_DIR).
	python scripts/download_msls.py --source "$(MSLS_RAW_DIR)"

run:  ## Start the FastAPI dev server.
	uvicorn app.main:app --reload --app-dir backend
