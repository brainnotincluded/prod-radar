# Phase 2 Post-Processing Analysis: False Negative Investigation

**Date:** 2026-03-15
**File:** `ml-service/app.py`, function `_postprocess_sentiment` (line 254)

## Test Text

```
Сбербанк повысил процентные ставки на вклады. Причины роста доходности
депозитов в банках объяснены. Специалисты предупредили о скрытых механизмах,
связанных с высокими ставками по накопительным счетам. В систему страхования
вкладов АСВ был исключен ещё один банк.
```

Expected label: **neutral** (factual financial news)
Observed label: **negative**

---

## Rule-by-Rule Trace

### 1. Sarcasm Detection (positive word + negative context)

| Check | Result | Details |
|---|---|---|
| `_POSITIVE_WORDS` match | **NONE** | No word from the set appears in text |
| `_NEGATIVE_CONTEXT` match | **NONE** | No phrase from the set appears in text |
| Sarcasm emoji | **NONE** | No emojis in text |
| Moderate sarcasm emoji | **NONE** | No emojis in text |
| `_IRONIC_PATTERNS` regex match | **NONE** | No ironic pattern fires |
| **sarcasm_score** | **0.0** | Threshold is 0.35; not reached |

**Verdict:** Sarcasm rule does NOT fire.

### 2. Churn Detection

| Pattern | Match |
|---|---|
| `ухожу (в\|к\|на)` | No |
| `перейду (в\|к\|на)` | No |
| `перехожу (в\|к\|на)` | No |
| `закрываю (счёт\|счет\|карту)` | No |
| `достали` | No |
| `надоело` | No |
| `отключаюсь` | No |
| `расторгаю` | No |

**Verdict:** Churn rule does NOT fire.

### 3. Emoji Correction (short text)

- `len(text_no_space) = 230` -- threshold is `< 20`
- Rule does NOT apply (text is far too long).

**Verdict:** Emoji rule does NOT fire.

### 4. Partial Match Analysis

Specific checks requested:

| Question | Answer |
|---|---|
| Does "комиссию" match "комиссии"? | No. Python `in` checks exact substring. "комиссию" != "комиссии". Neither word appears in the text anyway. |
| Does "предупредили" match any rule? | No. It shares no prefix >= 4 chars with any `_NEGATIVE_CONTEXT` entry. It is absent from all rule sets. |
| Do `_POSITIVE_WORDS` trigger false sarcasm? | No. Zero positive words match, so `has_pos_word = False`, and the sarcasm score stays at 0.0 regardless of negative context. |

### 5. Summary of All Rules

```
Rules that FIRE:  NONE
Post-processing output: PASS-THROUGH (returns ML model result unchanged)
```

---

## Root Cause

The negative classification comes from the **ML model itself** (Phase 1 ruRoBERTa-large), not from the post-processing layer.

### Why the model misclassifies

1. **Sigmoid outputs, not softmax.** The model uses multi-label sigmoid, so scores for negative, neutral, and positive are independent and do not sum to 1. For borderline text, negative and neutral scores can be very close (e.g., negative=0.68 vs neutral=0.65). The `_map_sentiment` function picks whichever label has the highest raw score.

2. **Negatively-connotated vocabulary in neutral context.** The text contains words that carry negative connotation in isolation but are used factually in financial reporting:
   - "предупредили" (warned)
   - "скрытых механизмах" (hidden mechanisms)
   - "исключен" (excluded)
   - "страхования" (insurance -- associated with risk)

   The model likely learned these tokens as negative signals from social media training data where they appear in complaints.

3. **Domain mismatch.** The model was trained on social media mentions (complaints, praise, churn signals). Neutral financial/news text is underrepresented, so the model has a negative bias on text that discusses risks or warnings, even when the text is informational.

---

## Structural Problem in `_postprocess_sentiment`

The post-processing function is **one-directional**: it can only push sentiment toward negative. There are three code paths that change the label, and all three produce `label="negative"`:

1. Line 279: sarcasm flip -- `positive -> negative`
2. Line 284: churn override -- `any -> negative`
3. Line 292: negative emoji -- `any -> negative`

There is one path that pushes toward positive (line 294: positive emoji on short text), but **no path exists to correct a false negative back to neutral.** This means:

- If the ML model gets it right, post-processing can only make it worse (for positive/neutral texts).
- If the ML model gets it wrong (false negative), post-processing cannot fix it.
- The function has no "neutral domain" safeguards or confidence-gating.

---

## Recommendations

### Short-term (fix the immediate false negative problem)

1. **Add a confidence margin check in `_map_sentiment`.** If the gap between the top-two labels is below a threshold (e.g., 0.05), default to neutral rather than picking the marginally-winning label:
   ```python
   if best_label == "negative" and scores["neutral"] > 0 and (best_score - scores["neutral"]) < 0.05:
       return SentimentResponse(label="neutral", score=round(scores["neutral"], 4))
   ```

2. **Add a neutral-domain word list to `_postprocess_sentiment`.** If the text contains financial/news vocabulary ("ставки", "вклады", "депозитов", "доходности", "страхования") AND no strong negative signals from `_NEGATIVE_CONTEXT`, downgrade a marginal negative to neutral.

### Medium-term (fix the training data gap)

3. **Add neutral financial news samples to the training set.** The Phase 1 model was trained on 318K samples that likely skew toward opinionated social media text. Adding 5-10K neutral news articles about banking, finance, and regulations would reduce false negatives on informational text.

4. **Switch `_map_sentiment` from max-score to calibrated thresholds.** Instead of "pick the highest sigmoid score," use per-label thresholds calibrated on a validation set. E.g., require negative score > 0.70 before assigning negative, otherwise default to neutral.

### Long-term (architectural)

5. **Make post-processing bidirectional.** Add rules that can correct false negatives (negative -> neutral), not just false positives (positive -> negative). The current design assumes the model only errs by being too positive, which is incorrect for news/financial text.
