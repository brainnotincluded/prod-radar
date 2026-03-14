# Prod Radar

Brand monitoring platform with ML-powered sentiment analysis for Russian social media.

## Architecture
- **backend/** — Go microservices (API, collector, processor, enricher, spike detector, alerter)
- **ml-service/** — Python FastAPI ML service (sentiment, embeddings, risk detection)
- **frontend/** — React + TypeScript SPA
- **docs/** — Integration documentation

## Quick Start
```bash
# Start infrastructure + all services
make up

# Run migrations
make migrate

# Test ML service
make test-ml

# Test API
make test-api

# View logs
make logs
```

## ML Service Endpoints
- POST /analyze — Combined sentiment + embedding (used by enricher)
- POST /sentiment — Single text sentiment
- POST /sentiment/batch — Batch sentiment
- POST /embedding — 768-dim embedding
- POST /classify-risk — Risk detection
- GET /health — Healthcheck

## Environment
Copy `backend/.env.example` to `backend/.env.docker` and configure.
