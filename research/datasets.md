# Russian Sentiment Datasets for Transformer Fine-Tuning

Research date: 2026-03-14

---

## Summary Table

| Dataset | Size | Labels | Domain | HuggingFace | Quality |
|---------|------|--------|--------|-------------|---------|
| MonoHime/ru_sentiment_dataset | 211K | 3 (pos/neg/neu) | Mixed (reviews, news, social) | Yes | High - aggregated from 6 sources |
| DmitrySharonov/ru_sentiment_neg_pos_neutral | 257K | 3 (pos/neg/neu) | Twitter + Reddit | Yes | Good - large social media corpus |
| ai-forever/ru-reviews-classification | 75K | 3 (pos/neg/neu) | E-commerce reviews | Yes | High - clean splits, Apache 2.0 |
| mteb/SentiRuEval2016 | 6K | 3 (pos/neg/neu) | Twitter (banks, telecom) | Yes | High - balanced, academic benchmark |
| Megnis/RuSentimentUnion | 31K | 5 (pos/neg/neu/skip/speech) | Social media (VKontakte) | Yes | Medium - includes non-sentiment labels |
| sepidmnorozy/Russian_sentiment | 4.2K | 2 (pos/neg) | News/political | Yes | Low - small, binary only |
| sy-volkov/russian-sentiment-analysis-dataset | 3.9K | 3 (Good/Bad/Neutral) | Social media comments | Yes | Low - small |
| sismetanin/rureviews | N/A | 3 (pos/neg/neu) | Women's clothing reviews | Yes (empty) | N/A - dataset currently empty on HF |
| RuSentiment (original) | ~31K | 5 classes | VKontakte posts | GitHub only | High - academic, widely cited |
| LINIS Crowd | Unknown | Tonal scale | Mixed texts | Not directly | Medium - tonal dictionary + texts |
| Kaggle: Sentiment Analysis in Russian | Unknown | 3 (pos/neg/neu) | News | Kaggle | Medium - competition dataset |

---

## Top Recommendations for Fine-Tuning

### Tier 1: Best for production fine-tuning

1. **MonoHime/ru_sentiment_dataset** -- largest aggregated dataset
2. **DmitrySharonov/ru_sentiment_neg_pos_neutral** -- largest single-source social media dataset
3. **ai-forever/ru-reviews-classification** -- cleanest splits, from ai-forever (Sber AI)

### Tier 2: Good for supplementary training or evaluation

4. **mteb/SentiRuEval2016** -- excellent benchmark, balanced classes
5. **Megnis/RuSentimentUnion** -- good size, needs label filtering

### Tier 3: Too small for primary training, useful for testing

6. **sepidmnorozy/Russian_sentiment**
7. **sy-volkov/russian-sentiment-analysis-dataset**

---

## Detailed Dataset Profiles

---

### 1. MonoHime/ru_sentiment_dataset

**HuggingFace:** https://huggingface.co/datasets/MonoHime/ru_sentiment_dataset

| Property | Value |
|----------|-------|
| Total samples | 210,989 |
| Train split | 190,000 |
| Validation split | 21,098 |
| Labels | 0=NEUTRAL, 1=POSITIVE, 2=NEGATIVE |
| Format | CSV / Parquet |
| Download size | 308 MB |
| License | Not specified |

**Domain:** Aggregated from 6 sources:
1. Kaggle "Sentiment Analysis in Russian" -- news articles
2. Russian Language Toxic Comments -- 2ch.hk and pikabu.ru comments
3. Car reviews dataset (Glazkova 2015)
4. Blinov sentiment datasets -- reviews from various domains
5. LINIS Crowd -- tonal dictionary and sentiment texts
6. Russian Hotel Reviews

**Quality Assessment:**
- Largest aggregated Russian sentiment dataset available
- Multi-domain coverage is a strength for generalization
- Text lengths vary wildly (3 to 381K characters) -- needs length filtering
- No test split -- will need to create one from validation
- Contains an unnecessary `Unnamed: 0` index column
- 3-class labels aligned with standard sentiment tasks

**How to load:**
```python
from datasets import load_dataset

dataset = load_dataset("MonoHime/ru_sentiment_dataset")
train = dataset["train"]       # 190,000 samples
val = dataset["validation"]    # 21,098 samples

# Example access
print(train[0])
# {'text': '...', 'sentiment': 0}  # 0=neutral, 1=positive, 2=negative

# Filter by reasonable text length
train_filtered = train.filter(lambda x: 10 < len(x["text"]) < 2000)
```

---

### 2. DmitrySharonov/ru_sentiment_neg_pos_neutral

**HuggingFace:** https://huggingface.co/datasets/DmitrySharonov/ru_sentiment_neg_pos_neutral

| Property | Value |
|----------|-------|
| Total samples | 257,485 |
| Splits | train only (257K) |
| Labels | "negative", "positive", "neutral" (string) |
| Format | CSV / Parquet |
| Download size | 37.1 MB |
| License | Apache 2.0 |

**Domain:** Twitter and Reddit messages in Russian.

**Quality Assessment:**
- Largest single Russian sentiment dataset on HuggingFace
- Social media domain -- ideal for user-generated content tasks
- Short texts (3-244 characters) -- typical of tweets
- Apache 2.0 license -- fully open for commercial use
- Only train split -- must create own val/test splits
- Labels are strings, not integers -- need mapping
- Source quality (Twitter/Reddit) may include noise, slang, transliteration

**How to load:**
```python
from datasets import load_dataset

dataset = load_dataset("DmitrySharonov/ru_sentiment_neg_pos_neutral")
data = dataset["train"]  # 257,485 samples

# Map string labels to integers
label_map = {"negative": 0, "neutral": 1, "positive": 2}
data = data.map(lambda x: {"label_id": label_map[x["label"]]})

# Create train/val/test splits (80/10/10)
split = data.train_test_split(test_size=0.2, seed=42)
train = split["train"]
remaining = split["test"].train_test_split(test_size=0.5, seed=42)
val = remaining["train"]
test = remaining["test"]
```

---

### 3. ai-forever/ru-reviews-classification

**HuggingFace:** https://huggingface.co/datasets/ai-forever/ru-reviews-classification

| Property | Value |
|----------|-------|
| Total samples | 75,000 |
| Train | 45,000 |
| Validation | 15,000 |
| Test | 15,000 |
| Labels | 0=negative, 1=neutral, 2=positive |
| Format | JSON / Parquet |
| Download size | 22.2 MB |
| License | Apache 2.0 |

**Domain:** E-commerce product reviews (likely AliExpress or similar). Categories include clothing, accessories, household items.

**Quality Assessment:**
- Published by ai-forever (Sber AI) -- high institutional quality
- Clean, pre-split train/val/test -- ready for direct use
- Part of RU-MTEB benchmark -- used as standard evaluation
- Apache 2.0 license -- fully open
- Balanced splits
- Reviews domain is well-suited for product sentiment
- Includes both numeric `label` and human-readable `label_text`
- Text length: 1-1000 characters -- reasonable range

**How to load:**
```python
from datasets import load_dataset

dataset = load_dataset("ai-forever/ru-reviews-classification")
train = dataset["train"]       # 45,000
val = dataset["validation"]    # 15,000
test = dataset["test"]         # 15,000

# Example
print(train[0])
# {'text': '...', 'label': 2, 'label_text': 'positive', 'id': '...'}
```

---

### 4. mteb/SentiRuEval2016

**HuggingFace:** https://huggingface.co/datasets/mteb/SentiRuEval2016

| Property | Value |
|----------|-------|
| Total samples | 6,000 |
| Train | 3,000 |
| Test | 3,000 |
| Labels | -1=negative, 0=neutral, 1=positive |
| Format | Parquet |
| Download size | 495 KB |

**Domain:** Russian Twitter -- reputation monitoring of banks and telecom companies.

**Source paper:**
```
Loukachevitch & Rubtsova (2016). "SentiRuEval-2016: overcoming time gap
and data sparsity in tweet sentiment analysis." Computational Linguistics
and Intellectual Technologies, pp. 416-426.
```

**Original repo:** https://github.com/mokoron/sentirueval

**Quality Assessment:**
- Academic benchmark -- high annotation quality
- Perfectly balanced: 1,000 samples per class per split
- Short texts (6-172 chars) -- tweet-length
- Financial/telecom domain -- good for brand sentiment
- Small size limits use as sole training data
- Best used as evaluation benchmark or supplementary data
- Minimal train/test overlap (147 shared texts)
- Used in MTEB (Massive Text Embedding Benchmark)

**How to load:**
```python
from datasets import load_dataset

dataset = load_dataset("mteb/SentiRuEval2016")
train = dataset["train"]  # 3,000 (balanced: 1K per class)
test = dataset["test"]    # 3,000 (balanced: 1K per class)

# Labels: -1 (negative), 0 (neutral), 1 (positive)
print(train[0])
# {'text': '...', 'label': 0}
```

---

### 5. Megnis/RuSentimentUnion

**HuggingFace:** https://huggingface.co/datasets/Megnis/RuSentimentUnion

| Property | Value |
|----------|-------|
| Total samples | 31,185 |
| Splits | train only |
| Labels | "neutral", "negative", "positive", "skip", "speech" |
| Format | Parquet |
| Download size | 2.86 MB |

**Domain:** Russian social media (VKontakte posts, based on RuSentiment source data).

**Quality Assessment:**
- Derived from the academic RuSentiment dataset
- 5-class scheme includes "skip" and "speech" which need filtering for standard sentiment
- After filtering to pos/neg/neu, effective size will be smaller
- Text lengths: 10-800 characters -- reasonable
- Only train split available
- Good for VKontakte/social media domain

**How to load:**
```python
from datasets import load_dataset

dataset = load_dataset("Megnis/RuSentimentUnion")
data = dataset["train"]  # 31,185

# Filter to standard 3-class sentiment
data_filtered = data.filter(
    lambda x: x["label"] in ["positive", "negative", "neutral"]
)

# Map to numeric labels
label_map = {"negative": 0, "neutral": 1, "positive": 2}
data_filtered = data_filtered.map(lambda x: {"label_id": label_map[x["label"]]})
```

---

### 6. sepidmnorozy/Russian_sentiment

**HuggingFace:** https://huggingface.co/datasets/sepidmnorozy/Russian_sentiment

| Property | Value |
|----------|-------|
| Total samples | 4,230 |
| Train | 2,940 |
| Validation | 424 |
| Test | 867 |
| Labels | 0=negative, 1=positive (binary) |
| Format | CSV / Parquet |

**Domain:** News articles, financial reports, economic/political content.

**Quality Assessment:**
- Very small dataset
- Binary only (no neutral class) -- limits use for 3-class tasks
- Has proper train/val/test splits
- Text lengths vary enormously (35 to 390K characters)
- News/political domain -- different from social media
- Best used for news-domain transfer learning only

**How to load:**
```python
from datasets import load_dataset

dataset = load_dataset("sepidmnorozy/Russian_sentiment")
train = dataset["train"]       # 2,940
val = dataset["validation"]    # 424
test = dataset["test"]         # 867
```

---

### 7. sy-volkov/russian-sentiment-analysis-dataset

**HuggingFace:** https://huggingface.co/datasets/sy-volkov/russian-sentiment-analysis-dataset

| Property | Value |
|----------|-------|
| Total samples | 3,873 |
| Splits | train only |
| Labels | G=positive(1), B=negative(2), N=neutral(0) |
| Format | Parquet |
| Download size | 261 KB |

**Domain:** Russian social media comments/posts.

**Quality Assessment:**
- Very small
- 3-class sentiment but uses non-standard label encoding (G/B/N strings + numeric)
- Text lengths: 2-2560 characters
- Social media domain matches target use case
- Too small for primary training

**How to load:**
```python
from datasets import load_dataset

dataset = load_dataset("sy-volkov/russian-sentiment-analysis-dataset")
data = dataset["train"]  # 3,873

# Labels: 'G' (positive), 'B' (negative), 'N' (neutral)
# Also has numeric 'label': 0=N, 1=G, 2=B
```

---

### 8. RuSentiment (Original Academic Dataset)

**GitHub:** https://github.com/sismetanern/rusentiment
**Paper:** Rogers, Romanov, Rumshisky et al. (2018). "RuSentiment: An Enriched Sentiment Analysis Dataset for Social Media in Russian." COLING 2018. (arXiv:1808.09271)

| Property | Value |
|----------|-------|
| Total samples | ~31,000 posts |
| Labels | 5 classes: positive, negative, neutral, speech act, skip |
| Source | VKontakte (Russian social network) |
| Format | CSV/TSV |

**Quality Assessment:**
- The foundational Russian social media sentiment dataset
- Published at COLING 2018 -- well-cited academic work
- Annotated by trained annotators with inter-annotator agreement metrics
- VKontakte data represents authentic Russian social media language
- 5-class scheme (pos/neg/neu/speech/skip) -- filter to 3 classes for standard task
- Not directly on HuggingFace as a loadable dataset (use GitHub)
- The Megnis/RuSentimentUnion dataset above is derived from this

**How to load:**
```python
# Option 1: From GitHub (original)
import pandas as pd

# Clone or download from https://github.com/sismetanern/rusentiment
df = pd.read_csv("rusentiment_random_posts.csv")
# Columns: text, label

# Option 2: Use the HuggingFace derivative
from datasets import load_dataset
dataset = load_dataset("Megnis/RuSentimentUnion")
```

---

### 9. LINIS Crowd

**Website:** http://www.linis-crowd.org/
**Authors:** Sergei Koltcov, Olessia Koltsova, Svetlana Alexeeva

| Property | Value |
|----------|-------|
| Total samples | Unknown (part of MonoHime aggregation) |
| Labels | Tonal scale (sentiment intensity) |
| Domain | Mixed texts |

**Quality Assessment:**
- Russian tonal dictionary + sentiment-annotated texts
- Crowdsourced annotation
- Part of the MonoHime aggregated dataset
- Original website may not be actively maintained
- No direct HuggingFace hosting
- Useful as a component but not standalone for fine-tuning

**How to access:**
- Best accessed through MonoHime/ru_sentiment_dataset which includes it
- Original: http://www.linis-crowd.org/

---

### 10. SentiRuEval (2015-2016 Shared Tasks)

**GitHub:** https://github.com/mokoron/sentirueval
**HuggingFace:** https://huggingface.co/datasets/mteb/SentiRuEval2016

| Property | Value |
|----------|-------|
| Total samples | ~6,000 (2016 edition) |
| Labels | -1 (neg), 0 (neu), 1 (pos) |
| Domain | Twitter (banks and telecom) |

**Details:** See entry #4 above (mteb/SentiRuEval2016).

**Background:**
- SentiRuEval was a series of shared tasks for Russian sentiment analysis
- SentiRuEval-2015 focused on reviews of cars and restaurants
- SentiRuEval-2016 focused on Twitter about banks and telecom companies
- Organized by Loukachevitch and Rubtsova
- Standard evaluation benchmark in Russian NLP community

---

### 11. Kaggle: Sentiment Analysis in Russian

**URL:** https://www.kaggle.com/c/sentiment-analysis-in-russian/data

| Property | Value |
|----------|-------|
| Total samples | Unknown (competition dataset) |
| Labels | positive, negative, neutral |
| Domain | News articles |
| Format | CSV |

**Quality Assessment:**
- Kaggle competition dataset
- News domain
- Included in the MonoHime aggregation
- Access may require Kaggle account
- Competition format typically has good label quality

**How to access:**
```python
# Via Kaggle API
# pip install kaggle
import subprocess
subprocess.run(["kaggle", "competitions", "download", "-c", "sentiment-analysis-in-russian"])

# Or access through MonoHime aggregate:
from datasets import load_dataset
dataset = load_dataset("MonoHime/ru_sentiment_dataset")
```

---

### 12. Kaggle: Russian Language Toxic Comments

**URL:** https://www.kaggle.com/blackmoon/russian-language-toxic-comments/

| Property | Value |
|----------|-------|
| Domain | 2ch.hk and pikabu.ru comments |
| Labels | Toxicity-based (mapped to sentiment) |

**Note:** This is a toxicity dataset, not pure sentiment. Included in MonoHime aggregate where toxic comments are mapped to negative sentiment. Use with caution -- toxicity != negative sentiment.

---

## Additional Datasets Found on HuggingFace

### ScoutieAutoML/weather_russian_regions_with_vectors_sentiment_ner
- 699 downloads, 10K-100K samples
- Telegram weather channels -- not useful for general sentiment

### Megnis/ru_sentiment_dataset-* (various sizes)
- Pre-sampled subsets of MonoHime dataset at different sizes (9K, 20K, 30K, 50K, 100K, 150K)
- Useful if you need a specific training size
- Load: `load_dataset("Megnis/ru_sentiment_dataset-50000")`

### mteb/RuReviewsClassification
- Derivative of ai-forever/ru-reviews-classification for MTEB evaluation
- 10K-100K samples, 3-class sentiment
- Product reviews

---

## Recommended Dataset Combination Strategy

For fine-tuning a transformer model on Russian sentiment, combine datasets for maximum coverage:

```python
from datasets import load_dataset, concatenate_datasets, DatasetDict

# Primary: largest social media dataset (Twitter + Reddit)
social = load_dataset("DmitrySharonov/ru_sentiment_neg_pos_neutral", split="train")
label_map = {"negative": 0, "neutral": 1, "positive": 2}
social = social.map(lambda x: {"text": x["text"], "label": label_map[x["label"]]})
social = social.select_columns(["text", "label"])

# Secondary: product reviews with clean splits
reviews = load_dataset("ai-forever/ru-reviews-classification")
reviews_train = reviews["train"].select_columns(["text", "label"])
reviews_val = reviews["validation"].select_columns(["text", "label"])
reviews_test = reviews["test"].select_columns(["text", "label"])

# Evaluation benchmark
sentirueval = load_dataset("mteb/SentiRuEval2016")
# Note: labels are -1/0/1, remap to 0/1/2
sentirueval_test = sentirueval["test"].map(
    lambda x: {"label": x["label"] + 1}  # -1->0, 0->1, 1->2
)

# Combine social + reviews for training
combined_train = concatenate_datasets([social, reviews_train])

# Use reviews validation for validation
# Use SentiRuEval test for out-of-domain evaluation

print(f"Training samples: {len(combined_train):,}")    # ~302K
print(f"Validation samples: {len(reviews_val):,}")      # 15K
print(f"Test (in-domain): {len(reviews_test):,}")        # 15K
print(f"Test (OOD Twitter): {len(sentirueval_test):,}")  # 3K
```

---

## Label Mapping Reference

Different datasets use different label schemes. Standardize before combining:

| Dataset | Negative | Neutral | Positive |
|---------|----------|---------|----------|
| MonoHime | 2 | 0 | 1 |
| DmitrySharonov | "negative" | "neutral" | "positive" |
| ai-forever | 0 | 1 | 2 |
| SentiRuEval | -1 | 0 | 1 |
| RuSentimentUnion | "negative" | "neutral" | "positive" |
| sepidmnorozy | 0 | N/A | 1 |
| sy-volkov | 2 (B) | 0 (N) | 1 (G) |

**Recommended standard:** negative=0, neutral=1, positive=2 (matches ai-forever)

---

## Pre-trained Russian Sentiment Models (for reference)

Models already fine-tuned on these datasets, useful as baselines or starting points:

- `seara/rubert-tiny2-russian-sentiment` -- trained on sismetanin/rureviews
- `seara/rubert-base-cased-russian-sentiment` -- trained on sismetanin/rureviews
- `blanchefort/rubert-base-cased-sentiment` -- popular Russian sentiment model
- `d010r3s/sbert-large-sentiment` -- trained on SentiRuEval2016

---

## Key Takeaways

1. **Best single dataset for social media:** DmitrySharonov/ru_sentiment_neg_pos_neutral (257K tweets/Reddit, Apache 2.0)
2. **Best curated dataset:** ai-forever/ru-reviews-classification (75K, clean splits, Sber AI quality)
3. **Best aggregated dataset:** MonoHime/ru_sentiment_dataset (211K, 6 sources)
4. **Best evaluation benchmark:** mteb/SentiRuEval2016 (6K, balanced, academic)
5. **Combined training potential:** ~300K+ samples by merging top datasets
6. **All datasets are freely available** on HuggingFace with standard Python loading
7. **Label standardization is critical** -- every dataset uses a different scheme
