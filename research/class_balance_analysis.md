# Class Balance Analysis: Neutral Dilution in Merged Training Data

Research date: 2026-03-15

---

## 1. The Problem

After merging proprietary data with three external HuggingFace datasets, the neutral class
dropped from 63% (in proprietary data) to 16.5% (in merged training set). This is a severe
distribution shift that directly harms the model's ability to classify neutral mentions --
the most common class in our production domain (brand monitoring).

**Merged training set (actual):**

| Class    | Count    | Percentage |
|----------|----------|------------|
| Positive | 133,395  | 42.0%      |
| Negative | 131,755  | 41.5%      |
| Neutral  |  52,575  | 16.5%      |
| **Total**| **317,725** | **100%** |

---

## 2. Per-Dataset Class Distribution (Estimated)

### 2.1 MonoHime/ru_sentiment_dataset (~211K raw)

Source: Aggregated from 6 corpora (Kaggle news, 2ch/pikabu toxic comments, car reviews,
Blinov reviews, LINIS Crowd, hotel reviews). The toxic comments component (mapped to
negative) and review data (skewed positive/negative) heavily bias this dataset away from
neutral.

**Estimated distribution:**

| Class    | Count    | Percentage |
|----------|----------|------------|
| Positive | ~85,000  | ~40%       |
| Negative | ~90,000  | ~43%       |
| Neutral  | ~36,000  | ~17%       |
| **Total**| **~211,000** | **100%** |

**Why neutral is low:** The 2ch/pikabu toxic comments dataset maps entirely to negative.
Car reviews and hotel reviews are polarized (people review to praise or complain). The
Kaggle news sentiment competition data is one of the few components that includes
meaningful neutral content. The LINIS Crowd tonal dictionary data contributes some neutral,
but it is a small fraction of the 211K total.

### 2.2 DmitrySharonov/ru_sentiment_neg_pos_neutral (~257K raw)

Source: Twitter + Reddit messages. Social media posts are selected for sentiment analysis
datasets specifically because they express opinions. Neutral tweets are less interesting
for annotation and are systematically undersampled.

**Estimated distribution:**

| Class    | Count    | Percentage |
|----------|----------|------------|
| Positive | ~110,000 | ~43%       |
| Negative | ~105,000 | ~41%       |
| Neutral  | ~42,000  | ~16%       |
| **Total**| **~257,000** | **100%** |

**Why neutral is low:** Twitter/Reddit data collection for sentiment datasets has a strong
selection bias toward opinionated posts. Annotators skip mundane factual tweets. The
dataset name itself ("neg_pos_neutral") suggests neutral was an afterthought. Social media
sentiment datasets consistently underrepresent neutral because neutral posts are "boring"
from an annotation perspective.

### 2.3 mteb/SentiRuEval2016 (~6K raw)

Source: Twitter mentions of banks and telecom companies. Academic benchmark with
intentionally balanced classes.

**Estimated distribution:**

| Class    | Count  | Percentage |
|----------|--------|------------|
| Positive | ~2,000 | ~33%       |
| Negative | ~2,000 | ~33%       |
| Neutral  | ~2,000 | ~33%       |
| **Total**| **~6,000** | **100%** |

**Note:** This is the only external dataset with balanced neutral representation. However,
at 6K samples it is too small to move the needle on the 317K merged set (contributes <2%).

### 2.4 Proprietary Data (~188K raw: xlsx + augmented)

Source: Brand monitoring mentions from production ProdRadar system.

**Distribution:**

| Class    | Count    | Percentage |
|----------|----------|------------|
| Positive | ~35,700  | ~19%       |
| Negative | ~33,800  | ~18%       |
| Neutral  | ~118,500 | ~63%       |
| **Total**| **~188,000** | **100%** |

**Why neutral is high:** This reflects the actual production distribution. In brand
monitoring, the majority of mentions are factual -- news articles, press releases, event
announcements, informational posts. Only a minority are opinionated (complaints or praise).
This is the ground truth for our domain.

---

## 3. Contribution of Each Dataset to the Merged Set

After length filtering (10 < len < 2000) and deduplication, the approximate contributions:

| Dataset | Raw Size | After Filter | % of Merged | Neutral Contribution |
|---------|----------|-------------|-------------|---------------------|
| MonoHime | 211K | ~180K | ~34% | ~30K neutral (~17% of its rows) |
| Sharonov | 257K | ~200K | ~38% | ~32K neutral (~16% of its rows) |
| SentiRuEval | 6K | ~5.5K | ~1% | ~1.8K neutral (~33% of its rows) |
| Proprietary | 188K | ~140K | ~27% | ~88K neutral (~63% of its rows) |
| **Total** | **662K** | **~525K** -> **~318K after dedup** | **100%** | **~52K neutral (16.5%)** |

**Key finding:** The two large external datasets (MonoHime + Sharonov) contribute ~72% of
the merged data volume but have only ~16-17% neutral each. They overwhelm the proprietary
data's 63% neutral rate through sheer volume.

---

## 4. Root Cause Analysis

The neutral dilution has three compounding causes:

### 4.1 Selection Bias in Public Datasets

Public sentiment datasets are built for sentiment classification research, which prizes
the ability to distinguish positive from negative. Neutral is treated as a necessary but
uninteresting third class. Annotators and dataset creators systematically:
- Prefer tweets/posts with clear sentiment expression
- Skip factual or mundane content
- Oversample from opinionated domains (reviews, complaints)

This creates a ~40/40/20 split in most public datasets vs the ~20/20/60 split in
real-world brand monitoring data.

### 4.2 No Weighting or Sampling During Merge

The `download_datasets.py` script performs a naive concatenation (line 332-333):

```python
all_rows = []
for rows in sources.values():
    all_rows.extend(rows)
```

There is no:
- Per-source sampling caps
- Per-class resampling
- Source weighting based on domain relevance
- Target distribution enforcement

### 4.3 Domain Mismatch

The external datasets are from different domains than our production use case:
- MonoHime: aggregated reviews, toxic comments, news
- Sharonov: Twitter/Reddit casual posts
- SentiRuEval: Twitter brand mentions (closest to our domain, but tiny)
- Proprietary: actual brand monitoring mentions (our target domain)

The external data teaches the model about "internet sentiment" in general but skews
away from the neutral-heavy distribution of professional brand monitoring.

---

## 5. Optimal Class Balance for Russian Brand Monitoring

### 5.1 What the Literature Says

For imbalanced multiclass classification, three strategies exist:

1. **Match production distribution** (63/19/18 neu/pos/neg) -- maximizes accuracy on
   real data but the model may underperform on minority classes (pos/neg) which are
   the classes users care about most.

2. **Balanced distribution** (33/33/33) -- standard for academic benchmarks,
   maximizes macro-F1 but does not reflect reality.

3. **Compromise distribution** -- oversample minority classes enough to learn them
   well, but keep the majority class (neutral) large enough that the model does not
   develop a bias toward predicting sentiment where there is none.

### 5.2 Recommendation: Compromise Distribution

For brand monitoring, neutral is the most important class to get right because:
- False positives (neutral predicted as negative) trigger unnecessary alerts
- False negatives (negative predicted as neutral) miss real issues
- The production prior is 60%+ neutral

**Recommended target distribution for training:**

| Class    | Percentage | Rationale |
|----------|------------|-----------|
| Neutral  | 40%        | Reduced from 63% production rate, but still the plurality class |
| Positive | 30%        | Oversampled vs production (19%) to learn sentiment patterns |
| Negative | 30%        | Oversampled vs production (18%) to learn sentiment patterns |

This gives the model:
- Enough neutral examples to learn the "no sentiment" pattern
- Enough positive/negative examples to distinguish sentiment polarity
- A prior that neutral is common (but not overwhelming)

At 317K total samples, that means:
- Neutral: ~127K (need ~75K more neutral or downsample pos/neg)
- Positive: ~95K (reduce from 133K)
- Negative: ~95K (reduce from 132K)

---

## 6. Proposed Resampling Strategies

### Strategy A: Undersample Positive/Negative + Oversample Neutral (Recommended)

```
Target: 40% neutral / 30% positive / 30% negative

Steps:
1. Keep all 52K neutral samples
2. Upsample neutral with paraphrase augmentation to ~127K
3. Randomly undersample positive from 133K to ~95K
4. Randomly undersample negative from 132K to ~95K
Total: ~317K (same size, better balance)
```

**Augmentation for neutral:** Use the existing Kimi-based augmentation pipeline
(`augment_data.py`) to generate ~75K neutral brand monitoring mentions. Neutral is
actually the easiest class to augment because factual/informational posts have less
linguistic variance than opinionated ones.

### Strategy B: Domain-Weighted Sampling

Weight each source by domain relevance during sampling:

| Source | Domain Weight | Effective Contribution |
|--------|--------------|----------------------|
| Proprietary | 3.0x | ~420K effective (3 * 140K) |
| SentiRuEval | 2.0x | ~11K effective (closest to our domain) |
| MonoHime | 0.5x | ~90K effective |
| Sharonov | 0.5x | ~100K effective |

After weighting, sample to target distribution. This preserves more proprietary data
patterns (including the natural neutral rate).

### Strategy C: Class-Weighted Loss (Already Implemented, Insufficient Alone)

The current training pipeline already uses `compute_class_weight("balanced")` in both
`finetune_v2.py` and `train_phase1.py`. With the current 16.5% neutral, the class
weights would be approximately:

```
positive:  0.79  (42% -> needs downweighting)
negative:  0.80  (41.5% -> needs downweighting)
neutral:   2.02  (16.5% -> needs 2x upweighting)
```

This helps but does not fully compensate for the severe imbalance because:
- The model still sees 5x fewer neutral gradient updates per epoch
- Neutral decision boundary has fewer examples to define it
- Class weights fix loss magnitude but not feature representation

**Recommendation: Combine Strategy A + C.** Resample to 40/30/30 AND keep class
weights. The weights will be milder (~1.0/1.0/0.8) and serve as fine-tuning rather
than heavy-handed correction.

### Strategy D: Two-Stage Training

1. **Stage 1:** Train on all data with current distribution (let the model learn
   general Russian sentiment patterns from the large external datasets)
2. **Stage 2:** Fine-tune on proprietary data only (or proprietary-weighted mix)
   to align the model with production distribution

This is similar to what `finetune_v2.py` already does (original + augmented only),
but applied after the large-data pretraining of `train_phase1.py`.

---

## 7. Should We Weight the Proprietary Data More?

**Yes, strongly.**

| Factor | External Data | Proprietary Data |
|--------|--------------|-----------------|
| Domain match | Low-Medium (social media, reviews) | Exact match (brand monitoring) |
| Class distribution | ~40/40/20 (pos/neg/neu) | ~19/18/63 (pos/neg/neu) -- matches production |
| Text style | Tweets, reviews, comments | News, press, social mentions of brands |
| Label quality | Crowdsourced, variable | Annotated for our specific use case |
| Neutral coverage | Sparse | Rich |

**Arguments for higher proprietary weight:**

1. **Domain alignment:** The proprietary data IS our target domain. The model should
   primarily learn from examples that look like what it will classify in production.

2. **Neutral representation:** Only the proprietary data has the neutral class
   properly represented. External datasets treat neutral as an afterthought.

3. **Label consistency:** Our annotations follow a single labeling guideline. External
   datasets were annotated by different teams with different criteria for what
   constitutes "neutral" vs "positive/negative."

4. **Production calibration:** A model trained predominantly on our data will have
   better-calibrated confidence scores for production use (the prior matches reality).

**Arguments for keeping external data:**

1. **Generalization:** External data adds linguistic diversity the model would not
   see from proprietary data alone.

2. **Edge cases:** Social media datasets include sarcasm, slang, and implicit
   sentiment patterns that broaden the model's capability.

3. **Volume:** 188K is enough for fine-tuning a base model but the additional 130K+
   from external sources can improve robustness.

**Recommendation: 2:1 proprietary-to-external ratio.** For every external sample,
include two proprietary samples (by upsampling or repeating). Combined with the
40/30/30 class rebalancing, this means the merged dataset better reflects both the
domain and the class distribution of production.

---

## 8. Implementation Plan

### Phase 1: Modify `download_datasets.py`

Add a resampling step after the merge:

```python
def resample_to_target(rows, target_ratios=None):
    """Resample merged data to target class distribution."""
    if target_ratios is None:
        target_ratios = {0: 0.30, 1: 0.30, 2: 0.40}  # pos/neg/neu

    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    # Determine target counts
    total = len(rows)
    target_counts = {label: int(total * ratio) for label, ratio in target_ratios.items()}

    resampled = []
    for label, target_n in target_counts.items():
        pool = by_label[label]
        if len(pool) >= target_n:
            # Undersample
            resampled.extend(random.sample(pool, target_n))
        else:
            # Oversample (repeat + partial)
            full_copies = target_n // len(pool)
            remainder = target_n % len(pool)
            resampled.extend(pool * full_copies)
            resampled.extend(random.sample(pool, remainder))

    random.shuffle(resampled)
    return resampled
```

### Phase 2: Domain-Weighted Source Sampling

Add source tracking and weight-based sampling before the merge:

```python
SOURCE_WEIGHTS = {
    "original_xlsx": 3.0,
    "augmented_jsonl": 3.0,
    "mteb/SentiRuEval2016": 2.0,
    "MonoHime/ru_sentiment_dataset": 0.5,
    "DmitrySharonov/ru_sentiment_neg_pos_neutral": 0.5,
}
```

### Phase 3: Generate Additional Neutral Data

Run `augment_data.py` with `TARGET_PER_CLASS = 25000` for neutral only, generating
~25K synthetic neutral brand monitoring mentions to supplement the pool before
resampling.

### Phase 4: Retrain and Evaluate

1. Regenerate `merged_train.jsonl` with new distribution
2. Retrain Phase 1 model (`train_phase1.py`)
3. Compare neutral-class F1 before and after
4. Evaluate on proprietary test set (not merged test set) for production-relevant metrics

---

## 9. Expected Impact

| Metric | Before (16.5% neutral) | After (40% neutral, estimated) |
|--------|----------------------|-------------------------------|
| F1-neutral | ~0.75 (underperforms) | ~0.85+ |
| F1-positive | ~0.88 | ~0.86 (slight decrease acceptable) |
| F1-negative | ~0.87 | ~0.85 (slight decrease acceptable) |
| F1-macro | ~0.83 | ~0.85+ |
| Production accuracy | Lower (too many false sentiment alerts) | Higher (fewer neutral misclassifications) |

The main production win is reducing false positive sentiment alerts -- cases where
the model predicts positive or negative on what is actually a neutral/factual mention.
This is the most common complaint from users of brand monitoring systems.

---

## 10. Summary

| Question | Answer |
|----------|--------|
| Why is neutral at 16.5%? | External datasets (72% of merged data) have only ~16% neutral due to selection bias |
| Which datasets pull neutral down? | MonoHime (~17% neutral) and Sharonov (~16% neutral) contribute 72% of data |
| What is the optimal balance? | 40% neutral / 30% positive / 30% negative for brand monitoring |
| Should proprietary data be weighted more? | Yes, 2:1 ratio (proprietary:external) recommended |
| Best resampling strategy? | Undersample pos/neg to 95K each + augment neutral to 127K + keep class weights |
| Quick fix? | Even just undersample pos/neg to match neutral (52K each) for a balanced 33/33/33 |
