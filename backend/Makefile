.PHONY: up down up-infra migrate migrate-down build-all seed-brands tidy test vet lint docker-build swagger

DB_URL ?= postgres://brandradar:brandradar@localhost:5432/brandradar?sslmode=disable

SERVICES := api collector processor enricher spike alerter

up:
	docker compose up -d

down:
	docker compose down

up-infra:
	docker compose up -d postgres redis nats

migrate:
	migrate -path migrations -database "$(DB_URL)" up

migrate-down:
	migrate -path migrations -database "$(DB_URL)" down

seed-brands:
	cd tools/seed-brands && go run .

tidy:
	cd pkg && go mod tidy
	@for svc in $(SERVICES); do cd services/$$svc && go mod tidy && cd ../..; done
	cd tools/seed-brands && go mod tidy

build-all:
	@mkdir -p bin
	@for svc in $(SERVICES); do \
		echo "building $$svc..."; \
		cd services/$$svc && go build -o ../../bin/$$svc ./cmd/ && cd ../..; \
	done

test:
	cd pkg && go test ./...
	@for svc in $(SERVICES); do cd services/$$svc && go test ./... && cd ../..; done

vet:
	cd pkg && go vet ./...
	@for svc in $(SERVICES); do cd services/$$svc && go vet ./... && cd ../..; done

lint: vet
	@echo "vet passed for all modules"

docker-build:
	@for svc in $(SERVICES); do \
		echo "docker build $$svc..."; \
		docker build -t brandradar-$$svc -f services/$$svc/Dockerfile .; \
	done

swagger:
	cd services/api && swag init -g cmd/main.go -o docs --parseDependency --parseInternal
