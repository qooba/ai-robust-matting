MAKE_HELP_LEFT_COLUMN_WIDTH:=14
.PHONY: help build-docker
help: ## This help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-$(MAKE_HELP_LEFT_COLUMN_WIDTH)s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST) | sort

build-docker-dev: ## builds dev docker image.
	cd docker && \
	cp -r ../src/app app && \
	docker build -t qooba/aimatting:robust-dev -f Dockerfile.dev . && \
	rm -rf app

build-docker: ## builds docker image.
	cd docker && \
	docker build --no-cache -t qooba/aimatting:robust -f Dockerfile .

run-dev: ## run dev mode
	docker run -it --gpus all -p 8000:8000 --rm --name aimatting -v $(pwd)/src/app:/app qooba/aimatting:dev /bin/bash

format: ## Format all the code using isort and black
	isort src/
	black --target-version py37 src

lint: ## Run mypy, isort, flake8, and black
	mypy src/
	isort src/ --check-only
	flake8 src/
	black --check src

test:
	pytest tests


