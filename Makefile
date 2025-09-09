SHELL := /bin/bash

.PHONY: up down logs rebuild precommit

up:
	docker compose up -d --build

down:
	docker compose down -v

logs:
	docker compose logs -f --tail=200

precommit:
	pre-commit install

pipeline-shell:
	docker compose run --rm pipeline bash

