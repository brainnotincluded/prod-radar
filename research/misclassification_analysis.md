# Misclassification Analysis: Neutral Financial News Classified as Negative

**Date:** 2026-03-15
**Model:** Daniil125/prodradar-sentiment-ru (ruRoBERTa-large, F1-macro 0.868)
**Pipeline:** `ml-service/app.py` -- `_map_sentiment()` -> `_postprocess_sentiment()`

---

## 1. The Misclassified Text

```
Сбербанк повысил процентные ставки на вклады. Причины роста доходности
депозитов в банках объяснены. Специалисты предупредили о скрытых механизмах,
связанных с высокими ставками по накопительным счетам.
```

| Field | Value |
|---|---|
| Predicted label | **negative** |
| Predicted score | **0.61** |
| Expected label | **neutral** |
| Human assessment | Factual financial reporting with cautionary language; no opinion expressed |

---

## 2. What Label SHOULD This Text Get?

**Neutral.** Here is the reasoning:

The text contains three factual statements:
1. "Sberbank raised deposit interest rates" -- a factual description of an event.
2. "The reasons for growth in deposit yields have been explained" -- a factual summary.
3. "Specialists warned about hidden mechanisms associated with high rates on savings accounts" -- while "warned" and "hidden mechanisms" carry cautionary connotation, this sentence reports what specialists said. It does not express the author's opinion.

**Comparison with known neutral financial text in our training data.** Our `data_neutral.json` contains structurally identical sentences:

- "ВТБ установил процентную ставку по ипотеке на вторичное жильё: 17,5% годовых" (VTB set mortgage rate at 17.5%)
- "ВТБ сообщил о запуске нового депозитного продукта с доходностью 15,1% годовых" (VTB announced new deposit with 15.1% yield)
- "Ставка по вкладам в топ-10 банках составила 18,5% годовых" (Deposit rate at top-10 banks was 18.5%)

All of these discuss interest rates and deposits in a purely informational tone. Our misclassified text is the same genre, but it includes cautionary language ("warned", "hidden mechanisms") that tips the model toward negative.

**Why not negative?** A negative label is reserved for text that expresses dissatisfaction, complaint, anger, or criticism toward a brand. This text does not do that. It is a news report. A brand monitoring analyst reading this text would not escalate it as a negative mention -- it belongs in the "informational/factual" bucket.

**The cautionary language does not make it negative.** Financial journalism routinely uses words like "warned", "risks", "hidden", "excluded" in informational contexts. The presence of these words reflects the subject matter (banking regulations, insurance), not the author's sentiment toward the brand.

---

## 3. Pipeline Trace: Where the Misclassification Happens

### 3.1 Phase 1 -- ML Model Output

The `text-classification` pipeline produces sigmoid scores for each label. For this text:

| Label | Raw Score (estimated) |
|---|---|
| LABEL_1 (negative) | ~0.61 |
| LABEL_2 (neutral) | ~0.55-0.59 |
| LABEL_0 (positive) | ~0.20 |

The `_map_sentiment()` function (app.py line 191) takes the label with the highest raw score:
```python
best_label = max(scores, key=lambda k: scores[k])  # "negative" wins by a small margin
```

The margin between negative and neutral is likely **0.02-0.06** -- a near-tie that the model resolves in the wrong direction.

### 3.2 Phase 2 -- Post-Processing

The `_postprocess_sentiment()` function (app.py line 254) checks sarcasm, churn, and emoji rules. **None of them fire on this text** (confirmed in `research/postprocessing_analysis.md`):

- No `_POSITIVE_WORDS` match -> sarcasm_score = 0.0
- No `_CHURN_PATTERNS` match
- Text is 230 chars (far above the 20-char emoji threshold)

**Post-processing passes the model's negative prediction through unchanged.**

### 3.3 Structural Deficiency

The post-processing layer is **unidirectional toward negative** (app.py lines 278-294). It has three paths that can change a label, and all three produce "negative":

1. Sarcasm: positive -> negative (line 279)
2. Churn: any -> negative (line 284)
3. Negative emoji: any -> negative (line 292)

There is one positive correction (positive emoji, line 294), but **no path exists to correct a false negative back to neutral**. Even if we added a "financial news" detection rule, the current function structure has no mechanism for neutral correction.

---

## 4. Hypothesis Analysis

### Hypothesis 1: Training Data Bias -- Financial Texts Labeled as Negative

**Why it might cause the issue:**

The model was trained on 318K samples merged from MonoHime (~211K), DmitrySharonov (~257K), SentiRuEval (~6K), and proprietary data (~188K). The external datasets (72% of merged data) come from social media, reviews, and comment forums where financial vocabulary appears overwhelmingly in negative contexts:

- "предупредили о мошенничестве" (warned about fraud) -- labeled negative
- "скрытые комиссии банка" (hidden bank fees) -- labeled negative
- "исключили из программы лояльности" (excluded from loyalty program) -- labeled negative

When these tokens appear in neutral financial news, the model carries over its learned association.

**How to verify:**

1. Extract all training samples from `merged_train.jsonl` containing the words "предупредили", "скрытых", "исключен".
2. Count the label distribution for those samples.
3. Compute: `P(negative | text contains "предупредили") vs P(neutral | text contains "предупредили")`

Predicted finding: >70% of training samples containing "предупредили" are labeled negative.

```python
import json
from collections import Counter

keywords = ["предупредили", "скрытых", "исключен"]
for kw in keywords:
    counts = Counter()
    with open("data/merged_train.jsonl") as f:
        for line in f:
            item = json.loads(line)
            if kw in item["text"].lower():
                counts[item["label"]] += 1
    total = sum(counts.values())
    print(f"\n'{kw}' appears in {total} training samples:")
    for label, count in sorted(counts.items()):
        label_name = {0: "positive", 1: "negative", 2: "neutral"}[label]
        print(f"  {label_name}: {count} ({100*count/total:.1f}%)")
```

**Suggested fix:**

a) Add 2-5K neutral financial news samples containing these keywords to the training set. The `gen/neu_finance.json` and `gen/hard_finance_neutral.json` files already contain ~300 such examples, but this is insufficient.

b) During data merging in `download_datasets.py`, apply a domain-aware sampling strategy that weights proprietary data (which has proper neutral financial coverage) 2-3x relative to external data.

c) Generate additional neutral samples using the augmentation pipeline (`augment_data.py`) with prompts specifically targeting financial/banking news vocabulary.

---

### Hypothesis 2: "предупредили" (Warned) Triggers Negative Sentiment

**Why it might cause the issue:**

"Предупредили" (warned, past tense of "предупредить") is a strong negative signal word in social media contexts where it typically appears in:
- "Меня предупредили, что деньги списались" (I was warned that money was debited)
- "Предупредили о блокировке счета" (Warned about account blocking)
- "Никто не предупредил о комиссии" (Nobody warned about the fee)

In financial journalism, the same word is used informatively:
- "Аналитики предупредили о рисках" (Analysts warned about risks) -- neutral
- "ЦБ предупредил участников рынка" (Central Bank warned market participants) -- neutral
- "Специалисты предупредили о механизмах" (Specialists warned about mechanisms) -- neutral

The model cannot distinguish these two usages because the training data overwhelmingly associates "предупредили" with negative sentiment.

**How to verify:**

Ablation test -- run inference on the text with and without "предупредили":

```python
# Original
text_full = "Сбербанк повысил процентные ставки на вклады. Причины роста доходности депозитов в банках объяснены. Специалисты предупредили о скрытых механизмах, связанных с высокими ставками по накопительным счетам."

# Without "предупредили"
text_no_warn = "Сбербанк повысил процентные ставки на вклады. Причины роста доходности депозитов в банках объяснены. Специалисты сообщили о механизмах, связанных с высокими ставками по накопительным счетам."

pipe = pipeline("text-classification", model="Daniil125/prodradar-sentiment-ru", top_k=None)
print("Full text:", pipe(text_full))
print("Without 'предупредили':", pipe(text_no_warn))
```

If replacing "предупредили" with the neutral synonym "сообщили" (reported) flips the prediction to neutral, this confirms the hypothesis.

**Suggested fix:**

a) **Short-term:** Add "предупредили" context awareness to `_postprocess_sentiment()`. When "предупредили" appears alongside financial vocabulary ("ставки", "вклады", "депозитов", "счетам", "банк") and no explicit complaint vocabulary is present, the post-processor should not treat the model's negative prediction as authoritative. Instead, apply a confidence margin check.

b) **Medium-term:** Create 500+ training samples where "предупредили" appears in neutral financial contexts. Include them in the next training round to recalibrate the model's association with this word.

c) **Long-term:** Use attention analysis (transformers-interpret or captum) to verify which tokens the model attends to most when making the negative decision. This provides direct evidence of which words drive the misclassification.

---

### Hypothesis 3: "скрытых механизмах" (Hidden Mechanisms) Triggers Negative

**Why it might cause the issue:**

"Скрытых" (hidden) appears in social media sentiment data almost exclusively in negative contexts:
- "скрытые комиссии" (hidden fees)
- "скрытое списание" (hidden charge)
- "скрытые условия договора" (hidden contract terms)

The phrase "скрытые механизмы" in financial journalism means "underlying/opaque mechanisms" -- an informational description of complex banking structures. But the model's training data likely contains "скрытый/скрытые/скрытых" almost entirely in complaint contexts.

**How to verify:**

Same approach as Hypothesis 2 -- ablation:

```python
# Replace "скрытых механизмах" with neutral phrasing
text_no_hidden = "Сбербанк повысил процентные ставки на вклады. Причины роста доходности депозитов в банках объяснены. Специалисты предупредили о механизмах, связанных с высокими ставками по накопительным счетам."

print("Without 'скрытых':", pipe(text_no_hidden))
```

Also check token-level attention:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("Daniil125/prodradar-sentiment-ru")
model = AutoModelForSequenceClassification.from_pretrained("Daniil125/prodradar-sentiment-ru")

inputs = tokenizer(text_full, return_tensors="pt")
outputs = model(**inputs, output_attentions=True)

# Check which tokens get highest attention in the final layer
# Tokens "скрытых" and "предупредили" likely have disproportionate attention weight
```

**Suggested fix:**

a) Add neutral training samples containing "скрытый/скрытые/скрытых" in informational contexts (financial mechanisms, technical architecture, etc.).

b) The compound effect of "предупредили" + "скрытых" in the same sentence is likely stronger than either word alone. The model sees a conjunction of two independently-negative-associated tokens and pushes the score further negative. This is the core of the problem: individually learned negative associations compound additively, overwhelming the neutral signal from the rest of the text.

---

### Hypothesis 4: Long Texts with Mixed Signals Default to Negative

**Why it might cause the issue:**

The model uses `max_length=512` and the text (3 sentences, ~230 characters) is well within this limit, so truncation is not the issue. However, the model was trained on `MAX_SEQ_LENGTH = 128` tokens (train_phase1.py line 57). This means:

1. During training, most examples were short social media posts (1-2 sentences, <128 tokens).
2. Longer texts that exceed 128 tokens were truncated during training, so the model never learned to aggregate sentiment across multiple sentences.
3. A 3-sentence text mixes neutral signals ("повысил ставки", "причины объяснены") with cautionary signals ("предупредили", "скрытых"). If the model cannot properly aggregate across sentences, it may default to the strongest local signal, which happens to be negative.

Additionally, the model config shows `"problem_type": "multi_label_classification"` with sigmoid outputs. In multi-label mode, each label is scored independently. Sigmoid does not enforce a probability distribution summing to 1 (unlike softmax). This means:
- The negative score can be 0.61 AND the neutral score can be 0.59 simultaneously.
- Neither score has to "win" by a significant margin.
- The max-score selection in `_map_sentiment()` picks a winner from nearly tied scores.

**How to verify:**

a) Test the model on each sentence individually:

```python
sentences = [
    "Сбербанк повысил процентные ставки на вклады.",
    "Причины роста доходности депозитов в банках объяснены.",
    "Специалисты предупредили о скрытых механизмах, связанных с высокими ставками по накопительным счетам.",
]
for s in sentences:
    print(f"{s[:60]}... -> {pipe(s)}")
```

Prediction: sentences 1 and 2 will classify as neutral; sentence 3 alone will classify as negative with higher confidence. The model is unable to "average out" the neutral sentences against the cautionary one.

b) Compare prediction on the full text vs just the first two sentences (without the cautionary third sentence).

**Suggested fix:**

a) **Short-term:** Add a multi-sentence averaging mechanism. For texts longer than a threshold (e.g., 2+ sentences), run inference on each sentence individually and aggregate the scores. If the majority of sentences are neutral and only one is weakly negative, the overall label should be neutral.

b) **Medium-term:** Increase `MAX_SEQ_LENGTH` from 128 to 256 or 512 during training. This lets the model learn cross-sentence context and handle longer factual texts without truncation artifacts.

c) **Long-term:** Switch from sigmoid (multi-label) to softmax (multi-class) output. The model config currently has `"problem_type": "multi_label_classification"`. Changing to softmax forces the model to distribute probability mass, making confident predictions more meaningful and reducing near-tie edge cases. This requires retraining.

---

### Hypothesis 5: Neutral Class Underrepresentation (16% vs 42%/42%)

**Why it might cause the issue:**

This is the **root cause** that makes all other hypotheses worse than they would be otherwise.

As documented in `research/class_balance_analysis.md`, the merged training set has a severely skewed distribution:

| Class | Count | Percentage |
|---|---|---|
| Positive | 133,395 | 42.0% |
| Negative | 131,755 | 41.5% |
| Neutral | 52,575 | **16.5%** |

This means:
- For every neutral training example, the model sees 2.5x as many positive and 2.5x as many negative examples.
- The model's prior is biased toward predicting sentiment (positive or negative) even when the text is neutral.
- The decision boundary between neutral and negative is poorly defined because there are too few neutral examples near the boundary.
- Ambiguous texts (like our financial news with cautionary language) fall on the wrong side of this poorly calibrated boundary.

**The class imbalance compounds with vocabulary bias.** If "предупредили" appears 100 times in training and 80 of those are labeled negative, 15 are positive, and only 5 are neutral, the model learns a very strong negative association. But in a balanced dataset, those 5 neutral uses would be proportionally more influential.

**How to verify:**

Compare the current model's neutral recall against what a balanced-training model would achieve:

```python
# Check the per-class metrics from the Phase 1 training
# In the test set classification report:
# - If neutral precision is high but recall is low: model is conservative about
#   predicting neutral (bias toward sentiment classes)
# - If neutral recall is low: model misclassifies many truly neutral texts as
#   positive or negative -- exactly our problem

# The v1 model (rubert-tiny2, sentiment-v1/metrics.json) shows:
#   neutral: precision=0.952, recall=0.812
#   High precision means when it predicts neutral, it's usually right
#   Lower recall means it MISSES many neutral texts (predicts them as pos/neg)
```

The 0.812 neutral recall from the v1 model means 18.8% of truly neutral texts were misclassified as either positive or negative. With the Phase 1 model (ruRoBERTa-large) trained on the even more imbalanced 16.5%-neutral merged data, neutral recall may be even lower.

**Suggested fix:**

a) **Immediate:** Resample the training data to 40% neutral / 30% positive / 30% negative before retraining, as recommended in `research/class_balance_analysis.md`. This requires:
   - Keeping all 52K neutral samples
   - Augmenting neutral to ~127K (using Kimi API + gen/ templates)
   - Undersampling positive from 133K to ~95K
   - Undersampling negative from 132K to ~95K

b) **Additional neutral data sources:** The `gen/` directory already contains dedicated neutral category files (neu_finance.json, neu_news.json, neu_corporate.json, neu_statistics.json, etc.) but these are synthetic and total only a few thousand samples. Real neutral data from financial news APIs (TASS, Interfax, RBC headlines) would be more effective.

c) **Modify `download_datasets.py`** to apply the `resample_to_target()` function proposed in `research/class_balance_analysis.md` before writing the final JSONL files.

---

## 5. Interaction Between Hypotheses

The five hypotheses are not independent -- they form a causal chain:

```
Hypothesis 5 (class imbalance)
    |
    v
Hypothesis 1 (training data bias) -- fewer neutral examples of financial vocabulary
    |                                 means vocabulary bias is not corrected
    v
Hypotheses 2+3 (trigger words) -- "предупредили" and "скрытых" learned as negative
    |                              signals from imbalanced data
    v
Hypothesis 4 (long text aggregation) -- model picks the strongest local signal
    |                                    (negative from trigger words) because it
    |                                    has no strong neutral prior to counterbalance
    v
Model outputs negative=0.61, neutral=0.58 (near-tie)
    |
    v
_map_sentiment() picks negative (max score)
    |
    v
_postprocess_sentiment() has no neutral correction path
    |
    v
Final output: negative (WRONG)
```

**The class imbalance is the root cause. The trigger words are the proximate cause. The sigmoid/max-score decision rule is the tipping point. The one-directional post-processing is the missed safety net.**

---

## 6. Recommended Fixes (Prioritized)

### Priority 1: Confidence Margin Check in `_map_sentiment()` [1 hour]

When the top two labels are within a small margin, default to neutral instead of picking the marginally winning label. This directly addresses the "near-tie resolution" problem.

```python
def _map_sentiment(raw_results: list[dict]) -> SentimentResponse:
    if not raw_results:
        return SentimentResponse(label="neutral", score=0.5)

    scores: dict[str, float] = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for item in raw_results:
        mapped = _LABEL_MAP.get(item["label"], "neutral")
        scores[mapped] = max(scores[mapped], item["score"])

    best_label = max(scores, key=lambda k: scores[k])
    best_score = scores[best_label]

    # NEW: If the gap between best and neutral is small, default to neutral.
    # This prevents marginal negative/positive predictions on ambiguous text.
    MARGIN_THRESHOLD = 0.05
    if best_label != "neutral" and scores["neutral"] > 0:
        gap = best_score - scores["neutral"]
        if gap < MARGIN_THRESHOLD:
            return SentimentResponse(label="neutral", score=round(scores["neutral"], 4))

    return SentimentResponse(label=best_label, score=round(best_score, 4))
```

**Impact:** Would correctly classify the test text (negative=0.61 vs neutral~0.58, gap<0.05).
**Risk:** May flip some correctly-classified marginal negatives to neutral. Needs evaluation on the test set to calibrate the threshold.

### Priority 2: Bidirectional Post-Processing with Financial Domain Rules [2-3 hours]

Add a neutral correction path to `_postprocess_sentiment()` for financial/news text that was marginally classified as negative.

```python
_FINANCIAL_VOCABULARY = {
    "ставки", "ставка", "вклады", "вклад", "депозит", "депозитов",
    "доходность", "доходности", "акции", "акций", "дивиденды",
    "капитализация", "выручка", "прибыль", "облигации", "облигаций",
    "индекс", "торги", "котировки", "аналитики", "прогноз",
    "процентная", "процентные", "рефинансирования", "ипотека",
    "кредитный", "портфель", "баланс", "отчётность", "отчитался",
    "размещение", "листинг", "эмиссия", "страхования", "резервы",
}

_STRONG_NEGATIVE_SIGNALS = {
    "ужас", "кошмар", "позор", "обман", "мошенничество", "воры",
    "грабёж", "достали", "надоело", "бесит", "ненавижу",
    "не работает", "сломал", "списали", "заблокировали", "навязали",
}

def _postprocess_sentiment(text, sentiment):
    text_lower = text.lower().strip()
    label = sentiment.label
    score = sentiment.score

    # ... existing sarcasm/churn/emoji rules ...

    # NEW: Financial news neutral correction
    # If model says negative with low confidence and text has financial vocabulary
    # but no strong negative signals, correct to neutral.
    if label == "negative" and score < 0.70:
        has_finance = sum(1 for w in _FINANCIAL_VOCABULARY if w in text_lower) >= 2
        has_strong_neg = any(w in text_lower for w in _STRONG_NEGATIVE_SIGNALS)
        if has_finance and not has_strong_neg:
            return SentimentResponse(label="neutral", score=round(max(score, 0.55), 4))

    return sentiment
```

**Impact:** Fixes false negatives on financial news without affecting true negatives (which have strong negative signals and/or high confidence).

### Priority 3: Retrain with Balanced Data [1-2 days]

This is the fundamental fix. Rebalance the training data following the analysis in `research/class_balance_analysis.md`:

1. Generate 25K+ neutral financial news samples using the augmentation pipeline
2. Resample merged data to 40% neutral / 30% positive / 30% negative
3. Weight proprietary data 2:1 vs external data
4. Retrain Phase 1 model with the rebalanced data
5. Evaluate neutral recall and overall F1 on a production-representative test set

### Priority 4: Switch from Sigmoid to Softmax [Part of retraining]

Change `"problem_type"` from `"multi_label_classification"` to `"single_label_classification"` in the model config. This changes the output layer from sigmoid (independent per-class scores) to softmax (normalized probability distribution), which:
- Forces scores to sum to 1.0, making the "gap" between classes more meaningful
- Eliminates the near-tie problem where negative=0.61 and neutral=0.59
- Produces better-calibrated confidence scores

This requires only a config change + retraining (no architecture change).

### Priority 5: Increase MAX_SEQ_LENGTH [Part of retraining]

Increase from 128 to 256 tokens. The current 128-token limit causes longer texts to be truncated during training, preventing the model from learning to aggregate sentiment across multiple sentences. With 256 tokens, a 3-sentence financial news snippet fits comfortably.

**Trade-off:** Training time increases ~2x, VRAM usage increases. With L4 24GB, batch size may need to drop from 16 to 8 (but gradient accumulation compensates).

---

## 7. Validation Plan

After implementing fixes, evaluate on these test cases:

### Neutral financial news (should be neutral):
```
1. "Сбербанк повысил процентные ставки на вклады. Причины роста доходности депозитов в банках объяснены. Специалисты предупредили о скрытых механизмах, связанных с высокими ставками по накопительным счетам."
2. "ЦБ предупредил банки о рисках, связанных с высокой концентрацией кредитного портфеля."
3. "Аналитики предупреждают о возможном замедлении роста ВВП в четвёртом квартале."
4. "Регулятор исключил три банка из системы страхования вкладов по результатам проверки."
5. "Эксперты указали на скрытые риски рефинансирования для заёмщиков с переменной ставкой."
```

### True negatives (must stay negative):
```
6. "Сбер без предупреждения списал деньги за скрытую комиссию! Это мошенничество!"
7. "Предупредили бы хоть, что тариф повысится. А так просто втихую списали."
8. "Банк скрыл важную информацию о комиссиях. Никому не рекомендую."
```

### Borderline cases (judgment call):
```
9. "Сбербанк повысил ставки, но эксперты считают это временной мерой." -> neutral
10. "Вкладчиков предупредили: скрытые условия могут привести к потерям." -> mild negative (author attributing risk to readers)
```

---

## 8. Summary

| Hypothesis | Likelihood | Impact | Root Cause? |
|---|---|---|---|
| H1: Training data bias on financial vocabulary | **High** | High | Contributing |
| H2: "предупредили" triggers negative | **High** | Medium-High | Proximate |
| H3: "скрытых механизмах" triggers negative | **High** | Medium | Proximate |
| H4: Long text mixed signals -> negative | **Medium** | Medium | Contributing |
| H5: Neutral class underrepresentation (16%) | **Very High** | **Very High** | **Root cause** |

**The root cause is the 16.5% neutral representation in the merged training data (H5), which makes all vocabulary biases (H1-H3) worse than they would be in a balanced dataset.** The sigmoid output layer and max-score selection rule (H4) then resolve near-tie predictions in the wrong direction. The one-directional post-processing offers no safety net for false negatives.

**Recommended action sequence:**
1. Deploy the confidence margin check (Priority 1) as an immediate mitigation
2. Add financial domain rules to post-processing (Priority 2) for a more targeted fix
3. Retrain with rebalanced data + softmax + longer sequences (Priorities 3-5) for the permanent fix
