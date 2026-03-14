# ML Service — Backend Integration Guide

## Connection

```
Host: 84.252.140.233
Port: 8000
Protocol: HTTP (no TLS)
```

---

## Endpoints

### 1. `GET /health`

Docker Compose liveness probe.

```json
// Response
{
  "status": "ok",
  "models_loaded": true,
  "device": "cuda",
  "latency_ms": 3.9
}
```

---

### 2. `POST /sentiment`

Single text sentiment analysis.

```bash
curl -X POST http://84.252.140.233:8000/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "Ужасный сервис", "lang": "ru"}'
```

```json
// Request
{ "text": "string", "lang": "ru" }

// Response
{ "label": "positive|negative|neutral|mixed", "score": 0.97 }
```

| Field | Type | Description |
|---|---|---|
| `label` | string | `positive`, `negative`, `neutral`, or `mixed` |
| `score` | float | Confidence 0.0–1.0 |

---

### 3. `POST /sentiment/batch`

Batch sentiment (max 128 items).

```json
// Request
{
  "items": [
    { "id": "uuid-1", "text": "Отличный сервис!" },
    { "id": "uuid-2", "text": "Всё сломалось" }
  ]
}

// Response
{
  "results": [
    { "id": "uuid-1", "label": "positive", "score": 0.92 },
    { "id": "uuid-2", "label": "negative", "score": 0.88 }
  ]
}
```

---

### 4. `POST /embedding`

Generate 768-dim vector for dedup/clustering.

```json
// Request
{ "text": "Лукойл повысил дивиденды", "lang": "ru" }

// Response
{ "embedding": [0.012, -0.034, ..., 0.051] }  // 768 floats, L2-normalized
```

---

### 5. `POST /embedding/batch`

Batch embeddings (max 128 items).

```json
// Request
{
  "items": [
    { "id": "uuid-1", "text": "..." },
    { "id": "uuid-2", "text": "..." }
  ]
}

// Response
{
  "results": [
    { "id": "uuid-1", "embedding": [0.012, ...] },
    { "id": "uuid-2", "embedding": [-0.005, ...] }
  ]
}
```

---

### 6. `POST /classify-risk`

Hybrid risk detection (keyword matching + sentiment signal).

```json
// Request
{
  "text": "Против компании подан судебный иск",
  "risk_words": ["судебный иск", "штраф", "скандал"]
}

// Response
{
  "is_risk": true,
  "matched": ["судебный иск"],
  "confidence": 0.92
}
```

| Confidence logic | Condition | Range |
|---|---|---|
| Keywords + negative sentiment | Strongest signal | 0.6–0.95 |
| Keywords only | Moderate | 0.4–0.85 |
| Negative sentiment only (>0.8) | Weak signal | ~0.5 |
| Nothing matched | No risk | ~0.1 |

`is_risk = true` when `confidence >= 0.4`

---

## Error Handling

| HTTP Code | Meaning |
|---|---|
| 200 | Success |
| 400 | Bad request (batch > 128 items) |
| 422 | Validation error (missing fields) |
| 503 | Models not loaded yet (startup) |

---

## Performance

| Metric | Value |
|---|---|
| Sentiment single | **6ms** p50 |
| Sentiment batch (10) | **8ms** p50 |
| Embedding single | **10ms** p50 |
| Health check | **4ms** |
| Max batch size | 128 items |
| Startup time | ~8s (model loading) |

---

## Models

| Task | Model | Params | Output |
|---|---|---|---|
| Sentiment | `rubert-tiny2` fine-tuned on 188K+10K samples | ~29M | 3-class + confidence |
| Embeddings | `cointegrated/LaBSE-en-ru` | ~175M | 768-dim L2-normalized vectors |
| Risk | Keyword matching + sentiment fusion | — | boolean + matched words + confidence |

**Sentiment metrics** (on 9.3K test set):
- F1-macro: 0.739
- F1-negative: 0.708
- Accuracy: ~75%

---

## DB Schema Mapping

```
mentions.sentiment_label    <- response.label      ("positive"/"negative"/"neutral"/"mixed")
mentions.sentiment_score    <- response.score       (0.0-1.0)
mentions.embedding          <- response.embedding   (float[768])
mentions.matched_risk_words <- response.matched     (text[])
```

---

## Server Details

```
IP:      84.252.140.233
SSH:     ssh -i ~/.ssh/prod-radar.pem ubuntu@84.252.140.233
GPU:     NVIDIA L4 (23GB VRAM, using ~900MB)
RAM:     16GB
Disk:    97GB (72GB free)
OS:      Ubuntu 22.04
Python:  3.10
Process: uvicorn on port 8000 (single worker)
```

---

## Enricher Integration Example (TypeScript)

```typescript
const ML_SERVICE = 'http://84.252.140.233:8000';

// Single mention
async function classifySentiment(text: string): Promise<{ label: string; score: number }> {
  const res = await fetch(`${ML_SERVICE}/sentiment`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, lang: 'ru' }),
  });
  return res.json();
}

// Batch (for enricher pipeline)
async function classifySentimentBatch(items: { id: string; text: string }[]) {
  const res = await fetch(`${ML_SERVICE}/sentiment/batch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ items }),
  });
  return res.json();
}

// Embedding for dedup
async function getEmbedding(text: string): Promise<number[]> {
  const res = await fetch(`${ML_SERVICE}/embedding`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, lang: 'ru' }),
  });
  const data = await res.json();
  return data.embedding;
}

// Risk classification
async function classifyRisk(text: string, riskWords: string[]) {
  const res = await fetch(`${ML_SERVICE}/classify-risk`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, risk_words: riskWords }),
  });
  return res.json();
}
```

---

---

## Combined Endpoint (for BrandRadar Enricher)

### `POST /analyze`

Single call that returns sentiment + embedding together. This is what the enricher service calls.

```bash
curl -X POST http://84.252.140.233:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Против компании подан судебный иск"}'
```

```json
// Request
{ "text": "string" }

// Response
{
  "sentiment_label": "negative",
  "sentiment_score": 0.91,
  "embedding": [0.012, -0.034, ..., 0.051]  // 768 floats
}
```

**Enricher env var:** `ML_SERVICE_URL=http://84.252.140.233:8000`

---

## Docker Compose Health Check

```yaml
ml-service:
  # external service at 84.252.140.233:8000
  # healthcheck from enricher:
  healthcheck:
    test: ["CMD", "curl", "-f", "http://84.252.140.233:8000/health"]
    interval: 30s
    timeout: 10s
    start_period: 60s
    retries: 3
```
