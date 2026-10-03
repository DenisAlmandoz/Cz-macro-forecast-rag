PYTHON ?= python3
VENV   ?= .venv
BIN     = $(VENV)/bin
HOST   ?= 127.0.0.1
PORT   ?= 8000

.DEFAULT_GOAL := help
.PHONY: help venv install dev run index reset health clean docker-build docker-run docker-stop

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv: ## Create the virtualenv
	test -d $(VENV) || $(PYTHON) -m venv $(VENV)

install: ## Install dependencies
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -r requirements.txt

dev: ## Run with autoreload
	$(BIN)/python -m uvicorn app.main:app --host $(HOST) --port $(PORT) --reload

run: ## Run in production mode
	$(BIN)/python -m uvicorn app.main:app --host 0.0.0.0 --port $(PORT)

index: ## Wipe the store and re-index every PDF in DOCUMENTS_DIR
	$(BIN)/python scripts/reingest.py

reset: ## Re-index without wiping (dedup applies)
	$(BIN)/python scripts/reingest.py --keep

health: ## Query the running app
	@curl -fsS http://$(HOST):$(PORT)/health && echo

clean: ## Remove build and cache artifacts
	rm -rf build dist *.egg-info .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

docker-build: ## Build the image
	docker build -t czech-forecast-rag .

docker-run: ## Run the image on :8000
	docker run --rm -p $(PORT):8000 --env-file .env -v "$(PWD)/data:/app/data" \
	  --name czech-forecast-rag czech-forecast-rag

docker-stop: ## Stop and remove the container
	-docker rm -f czech-forecast-rag