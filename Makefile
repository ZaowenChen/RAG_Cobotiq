SHELL := /bin/bash
PYTHON := python3

install:
	$(PYTHON) -m pip install -r requirements.txt

up:
	docker compose up -d

down:
	docker compose down

init-meili:
	bash scripts/init_meili.sh

init-qdrant:
	bash scripts/init_qdrant.sh

ingest:
	$(PYTHON) ingest.py --raw-dir data/raw --processed-dir data/processed
