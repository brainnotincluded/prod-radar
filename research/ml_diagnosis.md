# ML Service Diagnosis Report

**Date:** 2026-03-17
**Server:** 84.252.140.233:8000
**Code:** `/Users/mac/projects/prod-radar/ml-service/app.py`
**GitLab:** `/Users/mac/projects/brand-radar/ml/app.py`

---

## 1. Server Status

**HEALTHY.** `/health` returns:
```json
{"status":"ok","models_loaded":true,"device":"cuda","latency_ms":13.6}
```
- Running on CUDA (GPU)
- Inference latency: ~14ms (fast)
- All models loaded

---

## 2. `/analyze` Endpoint (5 fields)

All 5 fields present: `sentiment_label`, `sentiment_score`, `relevance_score`, `similarity_score`, `embedding`.

| Test text | label | score | relevance | similarity |
|---|---|---|---|---|
| "Сбербанк повысил ставки по вкладам" | neutral | 0.5619 | 0.30 | 0.0 |
| "Отличный банк, все нравится!" | positive | 0.8063 | 0.9676 | 0.0 |
| "Ужасный сервис, ухожу к конкурентам!" | negative | 0.7757 | 0.9308 | 0.0 |
| "Банк выпустил новую карту" | neutral | 0.6768 | 0.3384 | 0.0 |

**No issues with field presence.** All 5 fields returned correctly.

---

## 3. `/sentiment/batch` Endpoint

**Works correctly.** Returns `relevance_score` and `similarity_score` for each item:
```json
{"results":[
  {"id":"1","label":"positive","score":0.7461,"relevance_score":0.8953,"similarity_score":0.0},
  {"id":"2","label":"negative","score":0.8065,"relevance_score":0.9678,"similarity_score":0.0}
]}
```

---

## 4. Code Analysis

### 4a. Does `/sentiment/batch` return `relevance_score` and `similarity_score`?

**YES.** Lines 440-450 in app.py compute `relevance_score` using the same formula as `/analyze`, and hardcode `similarity_score: 0.0`. Both endpoints use identical relevance logic.

### 4b. Sentence Aggregation -- Is It Working Correctly?

**Mostly correct, but has issues:**

**ISSUE 1: "Офис рядом с домом" classified as negative (score 0.4).** The model misclassifies some neutral/positive statements as negative. This is a model-level issue, not a code bug, but it cascades into aggregation. In the mixed text test, 4 sentences were negative vs 3 positive, but one of the "negative" sentences ("Офис рядом с домом") is clearly neutral/positive. The aggregation cannot compensate for per-sentence model errors.

**ISSUE 2: The `pos_total == neg_total` edge case defaults to neutral with score 0.5 (line 173-174).** This is a reasonable design choice for truly balanced texts, but the "mixed" label is declared in the Pydantic model (line 49) and never actually produced by the code. Mixed-sentiment texts get classified as either positive, negative, or neutral -- never "mixed". This is a dead label.

**ISSUE 3: Post-processing is NOT applied to sentence-aggregated results.** Look at the flow:
- For short texts: `_classify_single()` -> `_postprocess_sentiment()` (sarcasm/churn applied)
- For long texts: sentence aggregation -> `_postprocess_sentiment()` (applied to the aggregated result)

This means sarcasm detection on the full text still runs even after aggregation, which is correct. However, individual sentences are classified via `_map_sentiment()` without post-processing -- this is by design (sarcasm usually spans the whole message, not individual sentences).

### 4c. Is the 15% Emotional Ratio Threshold Appropriate?

**Mostly, but with an edge case concern.** With 15%, even 1 emotional sentence out of 7 (14.3%) would fall below the threshold and the whole text defaults to neutral with score 0.8. For 2+ sentences, 1/7 = 14.3% which misses the cutoff. In practice:
- 2 sentences: 1 emotional = 50% (passes)
- 3 sentences: 1 emotional = 33% (passes)
- 7 sentences: 1 emotional = 14.3% (FAILS -- forced neutral)
- 10 sentences: 1 emotional = 10% (FAILS -- forced neutral)

**Verdict:** The 15% threshold is reasonable for filtering noise. Texts with only 1 emotional sentence out of 7+ are probably genuinely neutral/informational. The threshold protects against a single outlier sentence coloring a long neutral article.

### 4d. Incorrect Neutral Defaults

**ISSUE 4: "Нормальный банк, ничего особенного" is classified as POSITIVE (score 0.5598).** This is clearly a neutral/lukewarm statement but the model returns positive. The post-processing rules have no mechanism to demote weak positives to neutral. This is a model quality issue.

**ISSUE 5: Empty text returns negative (score 0.5043, relevance 0.6052).** Sending `{"text":""}` to `/analyze` returns negative sentiment with non-trivial relevance. There is no input validation for empty/whitespace-only text. This should return neutral with low relevance or an error.

**ISSUE 6: Off-topic text "Погода сегодня хорошая" returns positive (0.7571).** The model has no relevance filter at the sentiment level -- it classifies any Russian text by sentiment, regardless of whether it mentions the product/brand. The relevance_score partially compensates (positive off-topic text would get high relevance when it should get low), but there is no topic-awareness in the relevance formula.

**Other neutral default paths:**
- `_map_sentiment()` with empty results: returns neutral/0.5 (line 250) -- correct
- Unknown labels in `_LABEL_MAP`: default to "neutral" (line 255) -- correct
- Aggregation with <15% emotional: returns neutral/0.8 (line 176-177) -- correct but score 0.8 is surprisingly high for "no clear emotion detected"

### 4e. Model Loading Path

**Correct.** Line 102 checks `os.path.isdir("models/phase1-ruroberta/final")` and falls back to `Daniil125/prodradar-sentiment-ru` on HuggingFace. Since the server shows CUDA device and 14ms latency, the local model is likely loaded (HF fallback would be slower on first load). The path is relative to the working directory where uvicorn starts, so it depends on the deployment setup.

---

## 5. Local vs GitLab Comparison

**IDENTICAL.** `diff` between `/Users/mac/projects/prod-radar/ml-service/app.py` and `/Users/mac/projects/brand-radar/ml/app.py` produces no output. The files are byte-for-byte identical.

---

## 6. Relevance Score Analysis

Current formula:
```python
if sentiment.label in ("positive", "negative"):
    relevance = min(sentiment.score * 1.2, 1.0)
else:
    relevance = max(0.3, sentiment.score * 0.5)
```

**Observed ranges from testing:**

| Sentiment | Score range | Relevance range | Comment |
|---|---|---|---|
| positive | 0.56-0.81 | 0.67-0.97 | OK for strong positive; too high for weak "Нормальный банк" |
| negative | 0.49-0.81 | 0.59-0.97 | Reasonable |
| neutral | 0.44-0.68 | 0.30-0.34 | Always clustered around 0.30-0.34 |

**Problems:**

1. **Neutral relevance is too flat.** `max(0.3, score * 0.5)` means: with typical neutral scores (0.5-0.8), relevance = max(0.3, 0.25-0.40) = 0.30-0.40. The dynamic range is only 0.10. A confidently neutral text (score 0.8) gets relevance 0.40, while a barely neutral text (score 0.5) gets 0.30. There is almost no differentiation.

2. **Weak positives get inflated relevance.** "Нормальный банк" (positive 0.56) gets relevance 0.67. This is high for what is essentially a lukewarm non-opinion. The problem compounds: the model misclassifies neutral as weak positive, then the formula boosts it further with the 1.2x multiplier.

3. **No topic relevance.** The formula only considers sentiment confidence, not whether the text is actually about the tracked brand/product. "Погода хорошая" (positive 0.76) would get relevance 0.91, same as a genuine product review.

4. **similarity_score is always 0.0.** Hardcoded placeholder everywhere. If the enricher depends on this field, it is getting no value. If it handles dedup via embeddings independently, this field is wasted bandwidth (768 floats per request).

**Improvement suggestions:**

- Add a minimum score threshold for positive/negative: if score < 0.65, treat as neutral for relevance purposes
- Or use a smoother formula: `relevance = 0.3 + 0.7 * max(pos_score, neg_score)` regardless of label
- Consider embedding-based topic relevance: compare text embedding to a reference brand embedding, use cosine similarity as a topic-relevance multiplier
- If similarity_score is unused downstream, consider removing it to reduce payload size

---

## Summary of Issues

### Bugs / Must-Fix

| # | Severity | Issue | Location |
|---|---|---|---|
| 1 | **HIGH** | Empty text input returns negative sentiment instead of error/neutral | `/analyze`, no input validation |
| 2 | **MEDIUM** | "mixed" label declared in schema but never produced | `SentimentResponse` model, `predict_sentiment()` |
| 3 | **MEDIUM** | Neutral aggregation defaults to score=0.8, which is confusingly high confidence | `predict_sentiment()` line 177 |

### Model Quality Issues (Not Code Bugs)

| # | Severity | Issue | Example |
|---|---|---|---|
| 4 | **MEDIUM** | Lukewarm/neutral text classified as weak positive | "Нормальный банк, ничего особенного" -> positive 0.56 |
| 5 | **LOW** | Factual neutral sentences sometimes classified as negative | "Офис рядом с домом" -> negative 0.40 |
| 6 | **LOW** | Off-topic text gets normal sentiment classification | "Погода хорошая" -> positive 0.76 |

### Design Limitations

| # | Severity | Issue | Impact |
|---|---|---|---|
| 7 | **MEDIUM** | Relevance formula has no topic awareness | Off-topic mentions get high relevance |
| 8 | **LOW** | Neutral relevance range too narrow (0.30-0.40) | Cannot differentiate confidently vs. weakly neutral |
| 9 | **LOW** | similarity_score always 0.0 | Wasted response payload |
| 10 | **LOW** | Sarcasm detection only flips positive->negative, not neutral->negative | Sarcastic texts already classified as neutral are not caught |

### What Works Well

- Server is stable, fast (14ms on GPU), all endpoints operational
- Sarcasm detection fires correctly: "Спасибо за прекрасный сервис 🙃 часов" -> negative 0.49
- Churn detection works: "Достали, закрываю счет" -> negative 0.76
- Sentence aggregation produces reasonable results for genuinely mixed texts
- Risk classification is solid: matched all 3 keywords + negative sentiment -> confidence 0.95
- Local and GitLab code are in sync
- `/analyze/detailed` provides excellent per-sentence transparency
