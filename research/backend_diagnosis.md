# BrandRadar Backend Diagnosis

**Date:** 2026-03-17
**Server:** ubuntu@84.252.140.233
**Code path:** ~/BrandRadar/backend/

---

## Executive Summary

Six bugs found. The most critical is **Bug #1** (batch endpoint drops `relevance_score`) which is the root cause of "relevance always 0 or 1" on the frontend. Bug #2 (collector `JOIN user_brand_filters`) is a **design flaw** that makes the collector silently stop collecting for any brand without user-level filters. Bug #3 is a **scan column mismatch** in `ListFeed` that will cause runtime panics. The DB is currently empty due to cascading deletes from project deletion (Bug #5).

---

## Bug #1: Batch Sentiment Endpoint Drops `relevance_score` and `similarity_score`

**Severity:** HIGH -- root cause of "relevance always 0 or 1"

### The Problem

The batch ML path (`/sentiment/batch`) returns `relevance_score` and `similarity_score` in its response, but the Go struct that deserializes the batch results **does not have those fields**. They are silently discarded.

**File:** `backend/services/enricher/consumer/batch.go`, lines ~30-35

```go
type batchSentimentResult struct {
    ID    string  `json:"id"`
    Label string  `json:"label"`
    Score float64 `json:"score"`
    // MISSING: RelevanceScore  float64 `json:"relevance_score"`
    // MISSING: SimilarityScore float64 `json:"similarity_score"`
}
```

The ML `/sentiment/batch` endpoint returns:
```json
{
  "results": [
    {
      "id": "test1",
      "label": "positive",
      "score": 0.7842,
      "relevance_score": 0.941,
      "similarity_score": 0.0
    }
  ]
}
```

But only `id`, `label`, and `score` are deserialized. `relevance_score` and `similarity_score` are dropped.

Then in `flush()` (line ~115-125), the `mlResponse` is constructed with only sentiment data:

```go
resp := &mlResponse{
    SentimentLabel: "neutral",
    SentimentScore: 0,
    // RelevanceScore is zero-valued (0.0)
    // SimilarityScore is zero-valued (0.0)
}

if sr, ok := sentMap[item.ID.String()]; ok {
    resp.SentimentLabel = sr.Label
    resp.SentimentScore = sr.Score
    // relevance_score is NEVER set from batch result
}
```

### The Fix

```go
type batchSentimentResult struct {
    ID              string  `json:"id"`
    Label           string  `json:"label"`
    Score           float64 `json:"score"`
    RelevanceScore  float64 `json:"relevance_score"`
    SimilarityScore float64 `json:"similarity_score"`
}
```

And in `flush()`:

```go
if sr, ok := sentMap[item.ID.String()]; ok {
    resp.SentimentLabel = sr.Label
    resp.SentimentScore = sr.Score
    resp.RelevanceScore = sr.RelevanceScore
    resp.SimilarityScore = sr.SimilarityScore
}
```

### Impact

Every mention processed via the batch path (the primary path) gets `relevance_score = 0` and `similarity_score = 0` written to the database. The fallback `/analyze` path correctly parses all fields (see `mlResponse` struct in `consumer.go`), but it only fires when the batch fails.

---

## Bug #2: Collector `loadSubscribedBrands` Requires `user_brand_filters` (INNER JOIN)

**Severity:** HIGH -- causes "no subscribed brands" when user_brand_filters is empty

### The Problem

**File:** `backend/services/collector/scheduler/fetcher_builder.go`, lines ~79-110

```go
func (s *Scheduler) loadSubscribedBrands(ctx context.Context) ([]domain.Brand, error) {
    rows, err := s.pool.Query(ctx,
        `SELECT b.id, b.project_id, b.name, ...
         FROM brands b
         JOIN user_brand_filters ubf ON ubf.brand_id = b.id   -- INNER JOIN!
         LEFT JOIN LATERAL ( ... ) agg ON true
         GROUP BY ...`)
```

This uses `JOIN user_brand_filters ubf ON ubf.brand_id = b.id` -- an **INNER JOIN**. If a brand has no entry in `user_brand_filters`, it is excluded from the result entirely.

Currently the `user_brand_filters` table has **0 rows**. This means `loadSubscribedBrands` returns an empty list, and the collector logs:

```
WARN tick: no subscribed brands found, skipping
```

### The Fix

Change `JOIN` to `LEFT JOIN`:

```sql
FROM brands b
LEFT JOIN user_brand_filters ubf ON ubf.brand_id = b.id
```

Or better yet, remove the dependency on `user_brand_filters` entirely for the collector. The collector should collect for **all brands**, not just those with user filters. The user_brand_filters table is for per-user UI filtering, not for controlling which brands get monitored.

Alternatively, a simpler query:

```sql
SELECT b.id, b.project_id, b.name, b.keywords, b.exclusions, b.risk_words
FROM brands b
```

---

## Bug #3: `ListFeed` Duplicate Scan Column Mismatch

**Severity:** HIGH -- will cause runtime panics/errors when loading duplicates

### The Problem

**File:** `backend/services/api/repo/mention.go`, in `ListFeed()`, the duplicate-loading query at the bottom:

```go
dupRows, err := r.pool.Query(ctx,
    `SELECT `+mentionColumns+`, cluster_id
     FROM mentions
     WHERE cluster_id = ANY($1) AND id != ALL($2)
     ORDER BY published_at DESC`,
    clusterIDs, excludeIDs,
)
```

The `mentionColumns` constant selects **19 columns**:
```
id, project_id, source_id, brand_id, external_id, title, text, url, author,
published_at, matched_keywords, matched_risk_words,
sentiment_label, sentiment_score, relevance_score, similarity_score,
is_duplicate, status, created_at
```

Plus `cluster_id` = **20 columns** total.

But the scan for duplicate rows only scans **18 columns** (missing `relevance_score`, `similarity_score`):

```go
err := dupRows.Scan(
    &dup.ID, &dup.ProjectID, &dup.SourceID, &dup.BrandID, &dup.ExternalID,
    &dup.Title, &dup.Text, &dup.URL, &dup.Author, &dup.PublishedAt,
    &dup.MatchedKeywords, &dup.MatchedRiskWords,
    &dup.SentimentLabel, &dup.SentimentScore,
    // MISSING: &dup.RelevanceScore, &dup.SimilarityScore,
    &dup.IsDuplicate, &dup.Status, &dup.CreatedAt,
    &cid,
)
```

The query returns 20 columns but the scan expects 18 destinations. This will cause a `pgx` scan error: "number of field descriptions must equal number of destinations".

### The Fix

Add the missing scan fields:

```go
err := dupRows.Scan(
    &dup.ID, &dup.ProjectID, &dup.SourceID, &dup.BrandID, &dup.ExternalID,
    &dup.Title, &dup.Text, &dup.URL, &dup.Author, &dup.PublishedAt,
    &dup.MatchedKeywords, &dup.MatchedRiskWords,
    &dup.SentimentLabel, &dup.SentimentScore, &dup.RelevanceScore, &dup.SimilarityScore,
    &dup.IsDuplicate, &dup.Status, &dup.CreatedAt,
    &cid,
)
```

---

## Bug #4: `mention_clusters` Foreign Key Violation (Spam in Logs)

**Severity:** MEDIUM -- every enriched mention produces a warning

### The Problem

Enricher logs show continuous warnings:

```
WARN assign cluster error="ERROR: insert or update on table \"mention_clusters\"
violates foreign key constraint \"mention_clusters_project_id_fkey\" (SQLSTATE 23503)"
```

**File:** `backend/services/enricher/internal/repository/mention.go`, `AssignCluster()`:

```go
_, err = r.pool.Exec(ctx,
    `INSERT INTO mention_clusters (id, project_id, centroid, mention_count)
     VALUES ($1, $2, $3::vector, 1)`,
    newID, projectID, string(embJSON),
)
```

The `projectID` being inserted references a project that no longer exists in the `projects` table (because all projects were deleted -- see Bug #5). The `mention_clusters_project_id_fkey` constraint prevents the insert.

Even when projects exist, this could happen if a mention's `project_id` is NULL or stale. The `AssignCluster` function receives `projectID` from `fm.ProjectID` which is a `*uuid.UUID` -- it could be dereferenced even when the project no longer exists.

### The Fix

1. The immediate fix is to handle the FK violation gracefully: if cluster creation fails with an FK error, skip clustering rather than logging a warning for every single mention.
2. The root cause is that mentions arrive in the NATS queue with `project_id` values that reference deleted projects.

---

## Bug #5: Database Emptied by CASCADE Deletes

**Severity:** CRITICAL -- all data lost

### The Problem

The `projects` table has `ON DELETE CASCADE` to:
- `brands`
- `mentions`
- `sources`
- `mention_clusters`
- `alerts`
- `alert_configs`
- `events`
- `excluded_default_sources`

When a user deletes their project (via API), **everything** is cascade-deleted: all brands, all mentions, all sources, all events. Currently the database is completely empty:

```
projects:           0 rows
brands:             0 rows
mentions:           0 rows
sources:            0 rows
user_brand_filters: 0 rows
```

But users still exist (5 users), and `brand_catalog` has 8432 entries.

The init container seeded 4 demo projects at 00:26. The collector was running fine at 00:44 (19 projects, 27 brands, fetching thousands of articles). At 00:46 containers restarted and by that point all projects were gone.

### The Fix

1. **Soft-delete projects** instead of hard-delete: add a `deleted_at` column, filter by `deleted_at IS NULL`.
2. Or at minimum, require confirmation for project deletion and warn that all data will be lost.
3. The init container should not blindly re-run seed if data already exists -- but that's secondary since the seed uses `ON CONFLICT DO NOTHING`.

---

## Bug #6: Frontend `MentionCardContent` Displays Raw Float Instead of Percentage

**Severity:** LOW -- display issue only

### The Problem

**File:** `frontend/src/modules/mention-feed/ui/mention-feed/MentionCardContent.tsx`, lines 25-26:

```tsx
const rawRelevance = 'relevance_score' in mention ? mention.relevance_score : undefined
const relevanceScore = rawRelevance != null ? Math.round(rawRelevance) : undefined
```

The ML service returns `relevance_score` as a float 0-1 (e.g., 0.941). The schema says:
```
@description ML-based relevance score (0-1) from multi-task model
```

But `Math.round(0.941)` = `1`, and `Math.round(0.0)` = `0`. So the displayed value is always 0 or 1.

The display shows:
```tsx
<span className="block truncate">Релевантность: {relevanceScore ?? '---'}</span>
```

This shows "Релевантность: 0" or "Релевантность: 1" instead of a meaningful percentage.

### The Fix

Multiply by 100 before rounding:

```tsx
const relevanceScore = rawRelevance != null ? Math.round(rawRelevance * 100) : undefined
```

This would display "Релевантность: 94" (percent implied by context).

---

## Additional Findings

### ML Connectivity: Working

- `ML_SERVICE_URL=http://172.17.0.1:8000` in `.env`
- ML service is reachable from inside containers: `{"status":"ok","models_loaded":true,"device":"cuda","latency_ms":13.2}`
- `/analyze` endpoint returns correct `relevance_score` (0.941 for test input)
- `/sentiment/batch` also returns `relevance_score` in the response -- but the Go struct drops it (Bug #1)

### Circuit Breaker: Properly Configured

**File:** `backend/pkg/httputil/retry.go`

```go
DefaultRetryConfig() RetryConfig {
    MaxAttempts:      3,
    InitialDelay:     200ms,
    MaxDelay:         2s,
    FailureThreshold: 5,    // 5 consecutive failures to open
    ResetTimeout:     30s,  // half-open after 30s
}
```

Circuit breaker logic is correct. After 5 consecutive failures it opens, then after 30s allows a half-open probe. On success, it resets to closed. No bugs found here.

### API Layer: Correctly Passes `relevance_score` Through

The full pipeline from DB to JSON is correct in the API layer:

1. **`repo/mention.go`** -- `mentionColumns` includes `relevance_score, similarity_score`
2. **`repo/mention.go`** -- `scanMentionRow()` scans `&m.RelevanceScore, &m.SimilarityScore`
3. **`repo/mention.go`** -- `MentionRow` struct has `RelevanceScore *float64 json:"relevance_score"`
4. **`repo/mention_news.go`** -- `ListNewsWithTotal()` query selects `m.relevance_score, m.similarity_score` and scans them correctly
5. **`handler/feed.go`** -- Returns `result.Data` directly (which includes the `MentionRow` with all fields)

The API layer is **not** the problem. The data simply arrives as 0 because of Bug #1.

---

## Summary of Fixes Required

| # | Bug | File | Fix |
|---|-----|------|-----|
| 1 | Batch drops relevance_score | `enricher/consumer/batch.go` | Add `RelevanceScore`, `SimilarityScore` to `batchSentimentResult` struct; set them in `flush()` |
| 2 | Collector INNER JOIN on user_brand_filters | `collector/scheduler/fetcher_builder.go` | Change `JOIN` to `LEFT JOIN` (or remove the join entirely) |
| 3 | Duplicate scan column mismatch | `api/repo/mention.go` ListFeed() | Add `&dup.RelevanceScore, &dup.SimilarityScore` to dupRows.Scan() |
| 4 | mention_clusters FK violation | `enricher/internal/repository/mention.go` | Handle FK error gracefully; check project existence |
| 5 | CASCADE deletes wipe all data | DB schema / API delete handler | Soft-delete projects or add safeguards |
| 6 | Frontend rounds 0-1 float to 0 or 1 | `frontend/.../MentionCardContent.tsx` | `Math.round(rawRelevance * 100)` |
