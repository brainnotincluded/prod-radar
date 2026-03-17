# Frontend Diagnosis: Sentiment & Relevance Display

Date: 2026-03-17
Codebase: `/Users/mac/projects/brand-radar/frontend/`

---

## 1. How Sentiment Is Displayed on Mention Cards

### MentionCardContent.tsx (the actual card renderer)

**File:** `src/modules/mention-feed/ui/mention-feed/MentionCardContent.tsx`

Sentiment is displayed as a **Badge** with color-coded background/text:

```tsx
const sentimentLabel = mention.sentiment_label
  ? (SENTIMENT_LABELS[mention.sentiment_label] ?? mention.sentiment_label)
  : 'Не определена'

// ...

<Badge
  variant="secondary"
  className={cn(
    'inline-flex h-6 items-center whitespace-nowrap px-2.5 text-12 leading-none font-medium',
    getSentimentToneClass(mention.sentiment_label)
  )}
>
  {sentimentLabel}
</Badge>
```

**Color mapping** (`getSentimentToneClass`):
| sentiment_label | CSS classes |
|---|---|
| `positive` | `bg-success-bg text-success` (green) |
| `negative` | `bg-error-bg text-error` (red) |
| `neutral` | `bg-(--color-info-bg) text-(--color-info)` (blue/info) |
| fallback | `border border-(--color-border) bg-(--color-bg-subtle) text-(--color-text-60)` (gray) |

**Label mapping** (from `constants.ts`):
| Key | Display |
|---|---|
| `positive` | "Позитивные" |
| `negative` | "Негативные" |
| `neutral` | "Нейтральные" |
| `unknown` | "Не определено" |

**Field used:** `mention.sentiment_label` -- this comes directly from the API response `MentionRow.sentiment_label` (typed as `string | null`).

**Verdict:** Correct. The card reads `sentiment_label` from the API and maps it to a localized label with color coding. No issues.

### NewsModal.tsx (detail modal on click)

**File:** `src/modules/mention-feed/ui/NewsModal.tsx`

The modal also shows sentiment as a Badge, but uses its own local label map:

```tsx
const SENTIMENT_BADGE_LABELS: Record<string, string> = {
  positive: 'Позитив',
  negative: 'Негатив',
  neutral: 'Нейтрально',
}
```

Note the slightly different wording from the card (card says "Позитивные"/"Негативные"/"Нейтральные", modal says "Позитив"/"Негатив"/"Нейтрально"). This is a minor UX inconsistency.

The modal additionally calls `POST /ml/analyze/detailed` to get sentence-level analysis and displays each sentence with colored left-border and dot indicators (green for positive, red for negative, gray for neutral).

---

## 2. How Relevance Is Displayed

### MentionCardContent.tsx

**File:** `src/modules/mention-feed/ui/mention-feed/MentionCardContent.tsx`

```tsx
const rawRelevance = 'relevance_score' in mention ? mention.relevance_score : undefined
const relevanceScore = rawRelevance != null ? Math.round(rawRelevance) : undefined

// ...

<Badge
  variant="secondary"
  className="inline-flex h-6 max-w-[210px] items-center overflow-hidden whitespace-nowrap border-(--color-border) bg-(--color-bg-subtle) px-2.5 text-12 leading-none font-medium text-(--color-text-60)"
  title={`Релевантность: ${relevanceScore ?? '—'}`}
>
  <span className="block truncate">Релевантность: {relevanceScore ?? '—'}</span>
</Badge>
```

### BUG: `Math.round()` on a 0.0-1.0 float

The ML model returns `relevance_score` as a float between 0.0 and 1.0 (schema says `Format: double, @description ML-based relevance score (0-1) from multi-task model`).

`Math.round(0.73)` = **1**
`Math.round(0.49)` = **0**
`Math.round(0.85)` = **1**

So the user sees either **"Релевантность: 0"** or **"Релевантность: 1"**. This is the bug -- the display shows a binary 0/1 instead of a meaningful score.

### Fix options:
1. **Show as percentage:** `Math.round(rawRelevance * 100)` -> "Релевантность: 73"
2. **Show as decimal:** `rawRelevance.toFixed(2)` -> "Релевантность: 0.73"
3. **Show as percentage with symbol:** `${Math.round(rawRelevance * 100)}%` -> "Релевантность: 73%"

### Schema definition for `relevance_score`

**File:** `src/common/schema.d.ts` (line ~944)

```typescript
MentionRow: {
    // ...
    /**
     * Format: double
     * @description ML-based relevance score (0-1) from multi-task model
     */
    relevance_score?: number | null;
    // ...
};
```

The type is `number | null` -- correct. The field IS present on `MentionRow` and therefore on `NewsItem` (which extends `MentionRow`).

### Historical note from API description

The `listNews` endpoint description says:
> "sorting by date, sentiment score, relevance (**matched keywords count**), or popularity"

This is a stale description from before ML was added. The backend sort_by=relevance may still sort by `matched_keywords` count rather than `relevance_score`. This needs backend verification.

---

## 3. Analytics / Sentiment Chart

### SentimentChart.tsx

**File:** `src/modules/brand-charts/ui/SentimentChart.tsx`

Renders a **donut/pie chart** (Recharts `PieChart` with `innerRadius`) showing sentiment distribution.

**Two data sources depending on context:**
1. **Project-level:** `useSentimentData(projectId)` -> calls `GET /projects/{projectID}/analytics/sentiment`
2. **Brand-level:** `useCompareBrandsQuery(projectId, { brand_ids: brandId })` -> calls `GET /projects/{projectID}/analytics/compare`

**Data flow for project-level:**
- `useSentimentQuery` -> `GET /projects/{projectID}/analytics/sentiment`
- Response: `{ [key: string]: number }` (e.g., `{ positive: 42, negative: 15, neutral: 28, unknown: 3 }`)
- `useSentimentData` transforms this into `[{ name: "Позитивные", value: 42, raw: "positive" }, ...]`
- Chart uses `CHART_COLORS` mapping: positive=green, negative=red, neutral=gray, unknown=muted

**Data flow for brand-level:**
- `useCompareBrandsQuery` -> `GET /projects/{projectID}/analytics/compare`
- Response: `BrandComparison[]` with `sentiment_counts: { [key: string]: number }`
- Same transformation pattern

**Label display on chart:** Percentage labels like "Позитивные 48%"

**Verdict:** The sentiment chart is correct. It uses dedicated analytics endpoints that return pre-aggregated counts, not raw per-mention data.

---

## 4. API Data Flow

### GET /projects/{projectID}/news

**Schema operation:** `listNews`

**Response type:** `NewsListResponse`
```typescript
{
  data?: NewsItem[];       // Array of news items
  next_cursor?: string;    // Pagination cursor
  total?: number;          // Total count
}
```

**`NewsItem` extends `MentionRow` and adds:**
```typescript
{
  brand_name?: string;
  duplicate_count?: number;
}
```

**Fields on MentionRow (and therefore NewsItem) include:**
- `relevance_score?: number | null` -- ML relevance (0-1 float)
- `sentiment_label?: string | null` -- ML sentiment label
- `sentiment_score?: number | null` -- ML sentiment confidence
- `similarity_score?: number | null` -- ML similarity score (0-1 float)
- `matched_keywords?: string[]`
- `matched_risk_words?: string[]`
- `status?: "pending_ml" | "enriched" | "ready" | "dismissed"`

### Data fetching on the feed page

**File:** `src/modules/mention-feed/model/useNewsStream.ts`

The main feed page uses `useNewsStream`, which:
1. Makes initial REST call: `GET /projects/{projectID}/news` with filter params
2. Opens SSE connection to `/projects/{projectID}/news/stream`
3. When SSE events arrive, debounces (800ms) and refetches the REST query

**File:** `src/modules/mention-feed/model/useFeedQuery.ts`

A simpler alternative that just does the REST call. Both use `normalizeList(data)` to extract the array from `{ data: [...], total: N }`.

**Filter params passed through:**
- `sentiment`, `since`, `until`, `brand_id`, `search`
- `sort_by` (default: 'date'), `sort_order` (default: 'desc')
- `limit`, `offset`

### Does the response include the ML fields?

Yes. The schema confirms `relevance_score`, `sentiment_label`, `sentiment_score`, and `similarity_score` are all on `MentionRow`. The frontend type `MentionRow = components['schemas']['MentionRow']` or `NewsItem = components['schemas']['NewsItem']` gives typed access to all of them.

---

## 5. All Files Referencing "relevance"

### File inventory (8 files total):

| # | File | What it does with "relevance" |
|---|---|---|
| 1 | `src/common/schema.d.ts` | Defines `relevance_score: number \| null` on `MentionRow` and `ProcessedMention`/`ProcessedMentionUpdate`. Also defines `sort_by: "relevance"` as a valid sort option for `listNews`, `listBrandNews`, `listFeed`. API description mentions "relevance (matched keywords count)" -- stale doc. |
| 2 | `src/modules/mention-feed/ui/mention-feed/MentionCardContent.tsx` | **Reads** `mention.relevance_score`, applies `Math.round()` (BUG: rounds 0-1 float to 0 or 1), displays as Badge text "Релевантность: {score}". |
| 3 | `src/modules/mention-feed/model/useFeedQuery.ts` | Defines `FeedFilters.sort_by` with `'relevance'` as an option. Passes `sort_by` to the API query params. |
| 4 | `src/modules/mention-feed/ui/NewsFilters.tsx` | Defines `RelevanceFilterValue = 'all' \| 'high' \| 'low'`. Renders a `<Select>` dropdown with options: "Релевантность: все", "Сначала высокая", "Сначала низкая". This controls sort order, not filtering. |
| 5 | `src/modules/mention-feed/ui/MentionFeed.tsx` | Uses `relevance` filter state. When relevance='high', sets `sort_by='relevance', sort_order='desc'`. When 'low', sets `sort_by='relevance', sort_order='asc'`. When 'all', falls back to date sort. |
| 6 | `src/modules/mention-feed/ui/mention-feed/MentionFeedFiltersPopover.tsx` | Passes `relevance` filter value through to `NewsFilters` component. Pure plumbing. |
| 7 | `src/modules/brands/model/useAllNewsQuery.ts` | Client-side sort: when `sort_by === 'relevance'`, sorts by `a.relevance_score ?? 0` vs `b.relevance_score ?? 0`. This is the public brands page aggregator. |
| 8 | `app/(public)/brands/page.tsx` | Same relevance filter UI pattern as `MentionFeed.tsx`. Sets `sort_by: 'relevance'` when relevance filter is not 'all'. |

---

## Summary of Issues Found

### Critical Bug
- **"Релевантность: 1" display bug** in `MentionCardContent.tsx`: `Math.round()` on a 0.0-1.0 float always produces 0 or 1. Should multiply by 100 first or use `toFixed(2)`.

### Minor Issues
- **Inconsistent sentiment label wording:** Card shows "Позитивные" (from shared constants), modal shows "Позитив" (from local map). Pick one.
- **Stale API description:** The OpenAPI spec for `listNews` describes relevance sort as "matched keywords count". If the backend truly sorts by keyword count rather than ML `relevance_score`, the "sort by relevance" filter is misleading since users expect ML-based relevance. Needs backend verification.

### Things Working Correctly
- Sentiment badge colors on cards (green/red/blue/gray)
- Sentiment pie chart on analytics page (uses dedicated aggregation endpoint)
- SSE streaming with debounced refetch
- Relevance sort filter (sends correct `sort_by`/`sort_order` to API)
- Schema types correctly define `relevance_score` as `number | null` with `Format: double`
- NewsModal sentence-level analysis with `/ml/analyze/detailed`
