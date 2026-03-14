# BrandRadar

Real-time brand monitoring platform. Collects mentions from multiple sources, processes and enriches them with sentiment analysis, detects spikes, and delivers alerts.

## Architecture

```
┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐    ┌───────┐    ┌─────────┐
│ Collector │───>│ Processor │───>│ Enricher │───>│  Spike   │───>│Alerter│    │   API   │
│  (fetch)  │    │ (filter)  │    │(ML+dedup)│    │(detector)│    │(notify)│    │ (REST)  │
└──────────┘    └───────────┘    └──────────┘    └──────────┘    └───────┘    └─────────┘
      │               │               │               │               │            │
      └───────────────┴───────────────┴───────────────┴───────────────┘            │
                              NATS JetStream                                       │
      ┌───────────────┬───────────────┬───────────────┐                            │
      │           PostgreSQL          │     Redis     │                            │
      └───────────────────────────────┴───────────────┘────────────────────────────┘
```

### Services

| Service       | Description                                                        |
|---------------|--------------------------------------------------------------------|
| **API**       | REST API: auth, projects, sources, brands, feed, analytics, alerts |
| **Collector** | Fetches mentions from configured sources on a schedule             |
| **Processor** | Keyword/risk-word filtering, external_id dedup, mention insertion  |
| **Enricher**  | ML sentiment/embedding, pgvector dedup, cluster management         |
| **Spike**     | Statistical spike detection with Redis counters and cooldown       |
| **Alerter**   | Alert persistence, notification dispatch with retry                |

### NATS Subjects

```
mentions.raw       → Collector publishes raw items
mentions.filtered  → Processor publishes filtered mentions
mentions.ready     → Enricher publishes enriched mentions
alerts.trigger     → Spike publishes spike alerts
```

## Prerequisites

- Go 1.22+
- Docker & Docker Compose
- [golang-migrate](https://github.com/golang-migrate/migrate) CLI

## Quick Start

```bash
# 1. Copy environment config
cp .env.example .env

# 2. Start infrastructure (Postgres, Redis, NATS)
make up-infra

# 3. Run migrations
make migrate

# 4. Seed brand catalog (optional)
make seed-brands

# 5. Build all services
make build-all

# 6. Run individual services
./bin/api
./bin/collector
./bin/processor
./bin/enricher
./bin/spike
./bin/alerter
```

### Docker Compose (all services)

```bash
make up        # start everything
make down      # stop everything
```

## Environment Variables

| Variable          | Default                              | Description              |
|-------------------|--------------------------------------|--------------------------|
| `DATABASE_URL`    | `postgres://...localhost:5432/...`    | PostgreSQL connection    |
| `NATS_URL`        | `nats://localhost:4222`              | NATS server              |
| `REDIS_URL`       | `redis://localhost:6379`             | Redis server             |
| `JWT_SECRET`      | `brandradar-dev-secret-change-me`    | JWT signing secret       |
| `API_PORT`        | `8080`                               | API listen port          |
| `ML_SERVICE_URL`  | `http://localhost:8000`              | ML service for enricher  |

## API Endpoints

### Auth
- `POST /api/v1/auth/register` — register
- `POST /api/v1/auth/login` — login
- `POST /api/v1/auth/refresh` — refresh token
- `GET  /api/v1/auth/me` — current user

### Projects
- `POST /api/v1/projects` — create project
- `GET  /api/v1/projects` — list projects
- `GET  /api/v1/projects/{id}` — get project
- `PUT  /api/v1/projects/{id}` — update project
- `DELETE /api/v1/projects/{id}` — delete project

### Sources
- `POST /api/v1/projects/{id}/sources` — add source
- `GET  /api/v1/projects/{id}/sources` — list sources
- `PUT  /api/v1/projects/{id}/sources/{sid}` — update source
- `DELETE /api/v1/projects/{id}/sources/{sid}` — delete source
- `POST /api/v1/projects/{id}/sources/{sid}/toggle` — enable/disable

### Brand
- `POST /api/v1/projects/{id}/brand` — create brand config
- `GET  /api/v1/projects/{id}/brand` — get brand config
- `PUT  /api/v1/projects/{id}/brand` — update brand config
- `DELETE /api/v1/projects/{id}/brand` — delete brand config

### Feed
- `GET  /api/v1/projects/{id}/feed` — list mentions (filters: sentiment, status, brand_id, source_id, since, until, search, has_risk)
- `GET  /api/v1/projects/{id}/feed/{mid}` — mention detail
- `POST /api/v1/projects/{id}/feed/{mid}/dismiss` — dismiss mention

### Analytics
- `GET /api/v1/projects/{id}/analytics` — summary (total, sentiment counts, top sources)
- `GET /api/v1/projects/{id}/analytics/timeline` — time series (params: since, until, interval)

### Alerts
- `GET /api/v1/projects/{id}/alerts` — list alerts
- `GET /api/v1/projects/{id}/alert-config` — get alert config
- `PUT /api/v1/projects/{id}/alert-config` — upsert alert config

### Health
- `GET /health` — health check (database, redis, nats)

## Makefile Targets

```
make up           # docker compose up -d
make down         # docker compose down
make up-infra     # start only postgres, redis, nats
make migrate      # run migrations
make migrate-down # rollback migrations
make seed-brands  # seed brand catalog
make build-all    # build all service binaries to ./bin/
make test         # run all tests
make vet          # go vet all modules
make lint         # vet + lint check
make tidy         # go mod tidy all modules
make docker-build # build all Docker images
```

## Project Structure

```
├── migrations/          # SQL migrations (golang-migrate)
├── pkg/                 # Shared library (config, auth, domain, messaging, httputil, events)
├── services/
│   ├── api/             # REST API service
│   ├── collector/       # Source fetcher service
│   ├── processor/       # Keyword filter + mention inserter
│   ├── enricher/        # ML enrichment + pgvector dedup
│   ├── spike/           # Spike detection service
│   └── alerter/         # Alert notification service
├── tools/
│   └── seed-brands/     # Brand catalog seeder
├── docker-compose.yml
├── Makefile
└── go.work              # Go workspace definition
```
