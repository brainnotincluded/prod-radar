.PHONY: up down up-infra migrate seed logs logs-ml logs-enricher logs-api test-ml test-health test-api build rebuild clean

up:
	docker compose up -d

down:
	docker compose down

up-infra:
	docker compose up -d postgres redis nats

migrate:
	docker compose run --rm migrate

seed:
	cd backend && go run tools/seed-brands/.

logs:
	docker compose logs -f

logs-ml:
	docker compose logs -f ml

logs-enricher:
	docker compose logs -f enricher

logs-api:
	docker compose logs -f api

test-ml:
	curl -s -X POST http://localhost:8000/analyze -H "Content-Type: application/json" -d '{"text": "Тестовое сообщение"}' | python3 -m json.tool

test-health:
	curl -s http://localhost:8000/health | python3 -m json.tool

test-api:
	curl -s http://localhost:8080/health | python3 -m json.tool

build:
	docker compose build

rebuild:
	docker compose build --no-cache

clean:
	docker compose down -v
