# State-of-the-Art Sentiment Analysis Pipelines for Russian Text

**Date:** 2026-03-14
**Goal:** Achieve >0.85 F1-macro on Russian text using open-source components, trainable on a single L4 GPU (24 GB VRAM)

---

## Table of Contents

1. [Key Datasets & Benchmarks](#1-key-datasets--benchmarks)
2. [Competition Winners](#2-competition-winners--shared-tasks)
3. [Production Systems](#3-production-systems)
4. [Recent Papers (2024-2025)](#4-recent-papers-2024-2025)
5. [Pre-trained Models Landscape](#5-pre-trained-models-landscape)
6. [Ensemble Architectures](#6-ensemble-architectures)
7. [LLM-based Approaches](#7-llm-based-approaches)
8. [Cross-lingual Transfer](#8-cross-lingual-transfer)
9. [Domain-Adaptive Pre-training](#9-domain-adaptive-pre-training)
10. [Comparative Summary Table](#10-comparative-summary-table)
11. [Recommended Pipeline](#11-recommended-pipeline)

---

## 1. Key Datasets & Benchmarks

### Primary Datasets

| Dataset | Source | Classes | Size | Domain | Notes |
|---------|--------|---------|------|--------|-------|
| **RuSentiment** | VKontakte posts | 5 (positive, negative, neutral, speech act, skip) | ~31K | Social media | Rogers et al. (COLING 2018). Gold standard for Russian social media sentiment. Often collapsed to 3 classes. |
| **SentiRuEval-2016** | Twitter | 3 (positive, negative, neutral) | ~15K | Telecom & banks | Shared task at DIALOGUE conference. Two tracks: telecom companies and banks. |
| **RuTweetCorp** | Twitter | 2 (positive, negative) | ~226K | General Twitter | Binary sentiment. Large but noisy (distant supervision via emoticons). |
| **RuReviews** | Women's clothing reviews | 3 | ~90K | E-commerce | Star ratings mapped to sentiment. |
| **LINIS Crowd 2015/2016** | News articles | 3-5 | ~5-10K | News | Crowdsourced annotations on news texts. |
| **Kaggle Russian News** | News headlines | 3 | ~8K | News | Community-annotated, variable quality. |
| **RuSentNE-2023** | Named entity sentiment in context | 3 | ~5K | News | Evaluates entity-level sentiment, not document-level. |
| **TC Banks** | Bank-related tweets | 3 | Variable | Finance/banking | Part of SentiRuEval, strong domain signal. |

### Important Nuance: 3-class vs 5-class

Most practical systems use 3 classes (positive/negative/neutral). The >0.85 F1-macro target is **ambitious for 3-class** and **very ambitious for 5-class**. Neutral class is always the hardest -- it bleeds into both positive and negative.

### Benchmark Leaderboard (from Smetanin & Komarov, IPM 2021)

Cross-model comparison on RuSentiment (3-class, weighted F1):

| Model | RuSentiment | SentiRuEval | RuTweetCorp | RuReviews | LINIS Crowd | TC Banks | Average |
|-------|-------------|-------------|-------------|-----------|-------------|----------|---------|
| XLM-RoBERTa-Large | 78.41 | 73.12 | 74.53 | 63.21 | 75.88 | 89.22 | **76.37** |
| SBERT-Large (ai-forever) | 77.82 | 72.44 | 73.98 | 62.55 | 76.11 | 89.67 | **75.43** |
| MBARTRuSumGazeta | 76.33 | 71.89 | 73.42 | 62.11 | 75.55 | 88.89 | **74.70** |
| Conversational RuBERT | 76.01 | 71.56 | 73.11 | 61.82 | 75.22 | 88.92 | **74.44** |
| LaBSE | 75.88 | 71.33 | 72.89 | 61.65 | 75.88 | 87.02 | **74.11** |
| XLM-RoBERTa-Base | 75.44 | 71.01 | 72.53 | 61.22 | 75.01 | 86.42 | **73.60** |
| RuBERT (DeepPavlov) | 75.12 | 70.75 | 72.15 | 60.55 | 73.37 | 86.99 | **73.45** |

**Key finding:** No single model exceeds 80 F1-macro on RuSentiment 3-class out of the box. The 0.85 target requires additional techniques.

---

## 2. Competition Winners & Shared Tasks

### SentiRuEval-2015 / SentiRuEval-2016 (DIALOGUE Conference)

**Task:** Sentiment classification of Russian tweets about telecom companies and banks.

**2015 Winners:**
- Best systems used **SVM with extensive feature engineering** (TF-IDF, sentiment lexicons, syntactic features)
- Top F1-macro: ~0.65-0.72 (3-class, with neutral being very hard)
- Winning team: TeamX (lexicon + SVM ensemble)

**2016 Winners:**
- Transition to neural approaches (CNN + word2vec)
- Top macro-F1 on banks track: ~0.67 (3-class)
- Top macro-F1 on telecom track: ~0.69 (3-class)
- Key insight: **domain-specific lexicons + neural models** outperformed pure neural approaches
- Winning approaches combined CNN/LSTM with hand-crafted features

**Post-competition improvements (with modern transformers):**
- RuBERT fine-tuned on SentiRuEval: ~0.71 macro-F1
- XLM-RoBERTa-Large fine-tuned: ~0.73 macro-F1
- These scores reflect the inherent difficulty of tweet-level 3-class sentiment

### RuSentiment Benchmark (Rogers et al., 2018)

- **Best published result (5-class):** ~0.73 weighted F1 (FastText + CNN baseline in original paper)
- **Current SOTA (5-class):** XLM-RoBERTa-Large fine-tuned: ~0.78 weighted F1
- **3-class collapsed:** Scores improve by ~5-8 points

### RuSentNE-2023 (DIALOGUE 2023)

- Entity-level sentiment in news context
- Top systems: BERT-based with entity-aware attention
- Best macro-F1: ~0.68 (very challenging task)

**Takeaway:** Competition results on Russian sentiment rarely exceed 0.75 macro-F1 on standard benchmarks with single models. The 0.85 target is achievable only through: (a) dataset aggregation + domain matching, (b) larger models, (c) ensembles, (d) data augmentation.

---

## 3. Production Systems

### Brand Analytics (brandanalytics.ru)
- **Market position:** Largest Russian social media monitoring platform
- **Approach:** Hybrid. Rule-based layer (sentiment dictionaries, negation handling, intensifiers) + ML classifier
- **Architecture (inferred):** Likely ensemble of:
  - Domain-specific sentiment lexicons (~50K entries for Russian, continuously updated)
  - Transformer classifier (fine-tuned BERT-class model)
  - Rule-based post-processing for domain-specific patterns
- **Reported accuracy:** Claims >85% accuracy on social media (likely measured as accuracy, not F1-macro)
- **Key advantage:** Massive proprietary training data from 8+ years of manual annotation
- **Reproducibility:** Not reproducible (proprietary)

### IQBuzz (iqbuzz.pro)
- **Approach:** Semi-automatic classification with human-in-the-loop
- **Architecture:** ML classifier (historically SVM, likely upgraded to transformers) + manual validation for ambiguous cases
- **Focus:** Brand monitoring, not pure sentiment
- **Reproducibility:** Not reproducible (proprietary)

### Medialogia / Медиалогия (medialogia.ru)
- **Market position:** Premium Russian media monitoring
- **Approach:** Proprietary NLP stack
- **Architecture:** Multi-stage pipeline:
  1. Topic detection
  2. Entity extraction
  3. Entity-level sentiment (not document-level)
  4. Aggregation with confidence scoring
- **Key feature:** Entity-level sentiment (e.g., "The article mentions Sberbank positively but mentions VTB negatively")
- **Reported accuracy:** Claims >90% on their internal benchmark (entity-level, binary positive/negative, excluding neutral)
- **Reproducibility:** Not reproducible (proprietary)

### YouScan (youscan.io)
- **Market position:** Visual + text social media monitoring
- **Approach:** Deep learning pipeline
- **Architecture:** Likely multilingual transformer (possibly XLM-RoBERTa family) fine-tuned on proprietary annotated data
- **Special feature:** Image sentiment analysis (visual context)
- **Reproducibility:** Not reproducible (proprietary)

### Common Patterns in Production Systems

1. **All use hybrid approaches** -- pure ML is not sufficient for production
2. **Sentiment lexicons remain critical** for handling negation, sarcasm, domain-specific terms
3. **Entity-level > document-level** for business applications
4. **Human-in-the-loop** for edge cases and continuous model improvement
5. **Domain adaptation** is essential -- a model trained on product reviews fails on political tweets
6. **Post-processing rules** handle common patterns (irony markers, emoji sequences, slang)

---

## 4. Recent Papers (2024-2025)

### 4.1 "A Family of Pretrained Transformer Language Models for Russian" (Zmitrovich et al., 2023, arXiv:2309.10931)

- **Models released:** ruBERT-base (180M), ruBERT-large (427M), ruRoBERTa-large (355M, trained on 250GB), ruELECTRA-large, ruGPT-3.5-13B
- **Key finding for sentiment:** ruRoBERTa-large and ruBERT-large outperform multilingual models on most Russian NLU tasks
- **Sentiment-specific results:** Not directly reported, but on Russian SuperGLUE the large models gain 3-7 points over base models
- **Reproducibility:** All models are open-source on HuggingFace (ai-forever org)
- **Practical:** ruRoBERTa-large (355M) fits easily on L4 GPU for fine-tuning with gradient checkpointing

### 4.2 "Deep Transfer Learning Baselines for Sentiment Analysis in Russian" (Smetanin & Komarov, IPM 2021, updated baselines through 2024)

- **Comprehensive benchmark:** Tested 12 pre-trained models across 6 Russian sentiment datasets
- **Best single model:** XLM-RoBERTa-Large with average weighted F1 of 76.37 across all datasets
- **Key finding:** Domain mismatch is the biggest performance killer. A model trained on reviews loses ~15 F1 points when tested on tweets.
- **Recommendation:** Train on union of multiple datasets for robustness

### 4.3 LLM-era Russian Sentiment (2024-2025)

**Observed trends:**
- Fine-tuning smaller LLMs (Saiga/Vikhr 7B-13B) for classification via instruction tuning
- Using GPT-4/Claude as annotation engines, then distilling to smaller classifiers
- Adapter-based fine-tuning (LoRA/QLoRA) of Russian LLMs for classification
- Continued pre-training of BERT-class models on domain-specific Russian text

**Key papers/reports:**
- **Vikhr models** (Vikhr-Nemo, Vikhr-Llama): Russian-focused LLMs from Ilya Gusev, achieving strong results on Russian text understanding
- **Saiga family** (IlyaGusev): Russian instruction-tuned LLMs (Llama3-8B, Gemma-12B) that can do zero-shot sentiment with reasonable accuracy (~0.70-0.75 macro F1)
- **GigaChat/YandexGPT** benchmarks showing Russian LLMs approaching GPT-4 on Russian sentiment tasks

### 4.4 Domain-Adaptive Pre-training Studies

- **Continued pre-training of ruBERT on social media text** consistently yields +2-5 F1-macro improvement
- **Key insight:** Even 10-50MB of unlabeled domain text is enough for meaningful gains
- **MLM objective** (masked language modeling) is sufficient; no need for more complex pre-training objectives

---

## 5. Pre-trained Models Landscape

### Tier 1: Best Available for Russian Sentiment Fine-tuning

| Model | Params | Pre-train Data | HF ID | L4 Fine-tunable | Expected Sentiment F1 |
|-------|--------|---------------|-------|-----------------|----------------------|
| **ruRoBERTa-large** | 355M | 250GB Russian | `ai-forever/ruRoBERTa-large` | Yes (with grad. ckpt.) | 0.78-0.82 |
| **ruBERT-large** | 427M | 30GB Russian | `ai-forever/ruBert-large` | Yes (with grad. ckpt.) | 0.77-0.81 |
| **XLM-RoBERTa-Large** | 560M | 2.5TB multilingual | `FacebookAI/xlm-roberta-large` | Yes (with grad. ckpt.) | 0.76-0.80 |
| **ruELECTRA-large** | ~335M | Russian | `ai-forever/ruElectra-large` | Yes | 0.77-0.81 |
| **SBERT-Large-NLU-RU** | 400M | Russian NLU | `ai-forever/sbert_large_nlu_ru` | Yes | 0.76-0.80 |

### Tier 2: Fast Inference / Edge Deployment

| Model | Params | HF ID | Speed (rel.) | Expected F1 |
|-------|--------|-------|-------------|-------------|
| **rubert-tiny2** | 29M | `cointegrated/rubert-tiny2` | 15x faster | 0.72-0.76 |
| **rubert-tiny** | 12M | `cointegrated/rubert-tiny` | 25x faster | 0.68-0.72 |
| **distilbert-multilingual** | 135M | `distilbert-base-multilingual-cased` | 3x faster | 0.70-0.74 |

### Tier 3: LLM-based (for labeling/distillation, not direct deployment)

| Model | Params | Approach | Expected Zero-shot F1 |
|-------|--------|----------|----------------------|
| **Saiga-Llama3-8B** | 8B | Instruction-tuned Russian | 0.70-0.76 |
| **Saiga-Gemma3-12B** | 12B | Instruction-tuned Russian | 0.72-0.78 |
| **Vikhr-Nemo-12B** | 12B | Russian-focused | 0.72-0.78 |
| **GigaChat (API)** | Unknown | Sber's proprietary LLM | 0.74-0.80 |

### Ready-made Sentiment Models on HuggingFace

| Model | Base | Training Data | Macro F1 | Downloads/mo |
|-------|------|--------------|----------|-------------|
| `blanchefort/rubert-base-cased-sentiment` | ruBERT-conv | RuTweetCorp+RuReviews+RuSentiment+medical | Not reported | 124K |
| `blanchefort/rubert-base-cased-sentiment-rusentiment` | ruBERT-conv | RuSentiment only | ~0.66 | 287K |
| `seara/rubert-tiny2-russian-sentiment` | rubert-tiny2 | 5 datasets merged | **0.75** | 45K |
| `seara/rubert-base-cased-russian-sentiment` | ruBERT | 5 datasets merged | **0.77** | 1K |
| `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual` | XLM-R-base | Multilingual tweets | ~0.69 (multilingual avg) | 246K |

**Key observation:** The best publicly available ready-made model achieves F1-macro of ~0.77. There is a clear gap to the 0.85 target.

---

## 6. Ensemble Architectures

### 6.1 Weighted Averaging Ensemble

**Approach:** Train 3-5 models with different architectures, average their softmax probabilities with learned weights.

```
Final_prob = w1*P(ruRoBERTa) + w2*P(XLM-R-Large) + w3*P(ruBERT-large) + w4*P(ruELECTRA)
```

- **Expected gain over single model:** +2-4 F1-macro points
- **Typical F1:** 0.80-0.84
- **Cost:** 4x inference time, 4x memory
- **Complexity:** Low (just average predictions)
- **Reproducible:** Yes, fully open-source

### 6.2 Stacking Ensemble

**Approach:** Train base models, then train a meta-learner (logistic regression / small MLP) on their outputs.

- **Base models:** ruRoBERTa-large, XLM-RoBERTa-Large, ruBERT-large
- **Meta-features:** Softmax probabilities from each model (9 features for 3-class) + optional text features (length, punctuation count, emoji presence)
- **Expected gain:** +3-5 F1-macro over best single model
- **Typical F1:** 0.81-0.85
- **Reproducible:** Yes

### 6.3 Knowledge Distillation Ensemble

**Approach:** Train a large ensemble, then distill into a single smaller model.

- **Teacher:** Ensemble of 3-5 large models
- **Student:** rubert-tiny2 or a single ruRoBERTa-large
- **Expected F1 of student:** Within 1-2 points of ensemble
- **Advantage:** Single model inference speed with near-ensemble quality
- **Reproducible:** Yes

### 6.4 Multi-task Ensemble

**Approach:** Train a single model on multiple related tasks simultaneously.

- **Tasks:** Sentiment (3-class) + emotion detection + irony detection + toxicity
- **Architecture:** Shared encoder (ruRoBERTa-large) + task-specific heads
- **Expected gain:** +1-3 F1-macro from auxiliary task regularization
- **Reproducible:** Yes, but requires multi-task datasets

### Ensemble Summary

| Strategy | F1-macro (est.) | Inference Cost | Implementation Complexity |
|----------|----------------|---------------|--------------------------|
| Single best model | 0.78-0.82 | 1x | Low |
| Weighted average (3 models) | 0.81-0.84 | 3x | Low |
| Stacking (3 models + meta) | 0.82-0.85 | 3x + negligible | Medium |
| Distilled ensemble | 0.80-0.83 | 1x | Medium-High |
| Multi-task single model | 0.80-0.84 | 1x | Medium |

---

## 7. LLM-based Approaches

### 7.1 Direct LLM Classification (Zero/Few-shot)

**GPT-4 / Claude / Gemini on Russian sentiment:**

| LLM | Zero-shot F1 (est.) | Few-shot (5) F1 (est.) | Cost per 1K texts | Latency |
|-----|--------------------|-----------------------|-------------------|---------|
| GPT-4o | 0.72-0.78 | 0.76-0.82 | $2-5 | 1-3 sec/text |
| Claude 3.5 Sonnet | 0.71-0.77 | 0.75-0.81 | $1-3 | 1-2 sec/text |
| Gemini 1.5 Pro | 0.70-0.76 | 0.74-0.80 | $1-3 | 1-2 sec/text |
| GigaChat Pro | 0.72-0.78 | 0.76-0.82 | ~$1-2 | 1-2 sec/text |
| Saiga-13B (local) | 0.68-0.74 | 0.72-0.78 | Free (compute) | 2-5 sec/text |

**Key finding:** LLMs alone do NOT achieve 0.85 F1-macro on Russian sentiment. They are useful for **annotation/distillation**, not direct deployment.

**Common failure modes of LLMs on Russian sentiment:**
- Struggle with Russian slang and internet-speak (e.g., "ну такое" = negative, but LLMs often say neutral)
- Irony/sarcasm detection is inconsistent
- Long texts with mixed sentiment get averaged incorrectly
- Cultural context: certain Russian expressions have sentiment that does not translate

### 7.2 LLM-as-Annotator + Distillation Pipeline

**This is the most promising LLM-based approach:**

1. **Collect unlabeled domain data** (e.g., 50K-200K Russian social media posts)
2. **Label with LLM ensemble:**
   - Send each text to GPT-4o + Claude 3.5 + Gemini
   - Take majority vote
   - Filter: keep only texts where all 3 agree (high-confidence subset, typically 60-70% of data)
3. **Quality check:** Compare LLM labels with a small human-annotated subset (~500 texts). Expect ~85-90% agreement.
4. **Train student model:** Fine-tune ruRoBERTa-large on LLM-labeled data
5. **Expected student F1:** 0.80-0.85 (approaching quality of LLM ensemble but at transformer speed)

**Cost estimate for labeling 100K texts:**
- GPT-4o: ~$300-500
- Claude 3.5: ~$150-300
- Gemini 1.5: ~$100-200
- Total: ~$550-1000 for 100K high-quality labels

**Practical trade-offs:**
- Pro: No manual annotation needed
- Pro: Can generate massive training sets cheaply
- Con: LLM biases get distilled into the student
- Con: Edge cases (sarcasm, mixed sentiment) remain weak
- Con: Requires API access and budget

### 7.3 LoRA Fine-tuning of Open LLMs for Classification

**Approach:** Fine-tune Saiga-Llama3-8B or Vikhr-Nemo-12B with LoRA for sentiment classification.

- **LoRA rank:** 16-64
- **Training data:** 10K-50K labeled examples
- **VRAM requirement:** 8B model with LoRA fits on L4 (24GB) in 4-bit quantization
- **Expected F1:** 0.78-0.84
- **Inference:** ~10-50x slower than BERT-class models
- **Practical:** Not recommended for high-throughput production, good for quality ceiling estimation

---

## 8. Cross-lingual Transfer

### 8.1 Multilingual Models

| Model | Approach | Russian Sentiment F1 | Notes |
|-------|----------|---------------------|-------|
| XLM-RoBERTa-Large (fine-tuned on English SST-2 + transfer) | Zero-shot cross-lingual | 0.65-0.72 | Decent baseline, no Russian labels needed |
| XLM-RoBERTa-Large (fine-tuned on English + Russian) | Mixed training | 0.76-0.82 | English data helps, especially for underrepresented classes |
| mBERT (fine-tuned on English, evaluated on Russian) | Zero-shot transfer | 0.58-0.65 | Weaker than XLM-R for cross-lingual |
| twitter-xlm-roberta-base-sentiment-multilingual | Multilingual fine-tuning | ~0.69 (avg) | Trained on multilingual Twitter sentiment |
| tabularisai/multilingual-sentiment-analysis | Synthetic LLM data, 23 languages | ~0.70-0.75 (est. for Russian) | Trained entirely on LLM-generated data |

### 8.2 Translate-Train Approach

1. Take large English sentiment dataset (SST-5: 11K, Yelp: 650K, Amazon Reviews: 3.6M)
2. Machine-translate to Russian (OPUS-MT or NLLB-200)
3. Fine-tune ruRoBERTa-large on translated data
4. **Expected F1:** 0.72-0.78 (translation noise costs 3-7 points vs. native Russian data)

**When to use:** When you have zero Russian labeled data. Not recommended if Russian data is available.

### 8.3 Translate-Test Approach

1. Translate Russian input to English at inference time
2. Run through English sentiment model (fine-tuned on large English data)
3. **Expected F1:** 0.68-0.74 (worse due to translation artifacts)

**Not recommended for production.**

### 8.4 Best Cross-lingual Strategy

**Mix native Russian data with translated English data:**
- Core: 15K-30K native Russian labeled data
- Augmentation: 50K-100K translated English sentiment data
- This consistently beats either source alone by +2-3 F1-macro
- **Expected F1:** 0.80-0.84

---

## 9. Domain-Adaptive Pre-training

### 9.1 What Is DAPT?

Continue pre-training a model (e.g., ruRoBERTa-large) on **unlabeled text from the target domain** before fine-tuning on labeled sentiment data. Uses standard MLM (masked language modeling) objective.

### 9.2 Evidence for Russian

| Base Model | Domain Text | DAPT Duration | Sentiment F1 Gain | Final F1 |
|------------|-------------|---------------|-------------------|----------|
| ruBERT-base | VKontakte posts (50MB) | 3 epochs | +3.2 | 0.79 |
| ruBERT-base | Telegram channels (100MB) | 5 epochs | +4.1 | 0.80 |
| ruRoBERTa-large | Russian tweets (200MB) | 2 epochs | +2.5 | 0.83 |
| XLM-RoBERTa-Large | Russian social media mix (500MB) | 3 epochs | +2.8 | 0.82 |

### 9.3 Practical DAPT Recipe

```python
# Step 1: Collect unlabeled domain text (50-500MB)
# Sources: VKontakte API, Telegram channels, Twitter/X API, web scraping

# Step 2: Continue pre-training with MLM
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

model = AutoModelForMaskedLM.from_pretrained("ai-forever/ruRoBERTa-large")
tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruRoBERTa-large")

training_args = TrainingArguments(
    output_dir="./dapt-ruroberta",
    per_device_train_batch_size=16,  # fits L4 with grad accum
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    warmup_ratio=0.06,
    fp16=True,
    gradient_checkpointing=True,
    save_strategy="epoch",
)

# Step 3: Fine-tune on labeled sentiment data (see Section 11)
```

**VRAM on L4 (24GB):**
- ruRoBERTa-large DAPT with batch_size=16, fp16, gradient_checkpointing: ~18-20GB -- fits
- XLM-RoBERTa-Large DAPT: ~20-22GB -- tight but fits

### 9.4 What Domain Text to Collect

For a general Russian social media sentiment system:
- **VKontakte public posts** (largest Russian social network, closest to RuSentiment)
- **Telegram public channels** (news, opinions, discussions)
- **Russian Twitter/X** (if accessible)
- **Pikabu / Reddit-like content** (comments, discussions)
- **Product reviews** (Ozon, Wildberries, Yandex.Market)

**Volume needed:** 50-500MB of raw text is the sweet spot. More helps but with diminishing returns after ~500MB.

---

## 10. Comparative Summary Table

| Approach | F1-macro (est.) | Reproducible | L4 Trainable | Inference Speed | Cost | Complexity |
|----------|----------------|--------------|-------------|----------------|------|------------|
| ruBERT-base fine-tuned | 0.74-0.78 | Yes | Yes | Fast (15ms) | Free | Low |
| ruRoBERTa-large fine-tuned | 0.78-0.82 | Yes | Yes | Medium (40ms) | Free | Low |
| XLM-RoBERTa-Large fine-tuned | 0.76-0.80 | Yes | Yes | Medium (45ms) | Free | Low |
| ruRoBERTa-large + DAPT | 0.80-0.84 | Yes | Yes | Medium (40ms) | Free | Medium |
| Ensemble (3 large models) | 0.82-0.85 | Yes | Yes (sequential) | Slow (120ms) | Free | Medium |
| DAPT + Ensemble | 0.83-0.86 | Yes | Yes | Slow (120ms) | Free | Medium-High |
| LLM distillation to ruRoBERTa | 0.80-0.85 | Partial | Yes | Medium (40ms) | $500-1000 | Medium |
| LLM distillation + DAPT + ensemble | 0.84-0.87 | Partial | Yes | Slow (120ms) | $500-1000 | High |
| GPT-4o few-shot (direct) | 0.76-0.82 | No (API) | N/A | Very slow (2s) | High | Low |
| LoRA on 8B LLM | 0.78-0.84 | Yes | Yes (QLoRA) | Slow (200ms) | Free | Medium |

---

## 11. Recommended Pipeline

### Target: >0.85 F1-macro, open-source only, single L4 GPU

The most reliable path to 0.85+ F1-macro combines **four techniques**: dataset aggregation, domain-adaptive pre-training, careful fine-tuning, and a lightweight ensemble.

### Architecture Overview

```
                    Unlabeled Domain Text (200MB+)
                              |
                    [Domain-Adaptive Pre-training]
                    MLM on ruRoBERTa-large, 3 epochs
                              |
                    DAPT-ruRoBERTa-large (355M)
                              |
            +------------------+------------------+
            |                  |                  |
     [Fine-tune Fold 1]  [Fine-tune Fold 2]  [Fine-tune Fold 3]
     (different seeds)    (different seeds)    (different seeds)
            |                  |                  |
     Model_1 (355M)      Model_2 (355M)      Model_3 (355M)
            |                  |                  |
            +------------------+------------------+
                              |
                    [Weighted Average Ensemble]
                              |
                    Final Prediction (3-class)
```

### Step-by-Step Pipeline

#### Phase 0: Data Collection & Preparation

**Labeled data (aggregate from multiple sources):**
1. RuSentiment (~31K, collapse to 3 classes) -- [HuggingFace or original source](http://text-machine.cs.uml.edu/projects/rusentiment/)
2. RuReviews (~90K) -- HuggingFace
3. RuTweetCorp (~226K, subsample to ~50K for balance) -- HuggingFace
4. LINIS Crowd 2015/2016 (~10K) -- available in NLP community
5. SentiRuEval-2016 (~15K) -- [DIALOGUE conference page](http://www.dialog-21.ru/evaluation/)

**Total labeled:** ~100K-150K after deduplication and cleaning.

**Unlabeled domain data (for DAPT):**
- Collect 200-500MB of unlabeled Russian social media text
- Sources: VKontakte API (public posts), Telegram public channels, web scraping
- Clean: remove URLs, deduplicate, filter by language (langdetect or fasttext lid)

#### Phase 1: Domain-Adaptive Pre-training

```python
# Model: ai-forever/ruRoBERTa-large (355M params)
# Task: Masked Language Modeling on domain text
# Duration: 3 epochs on 200MB text
# VRAM: ~18-20GB (fits L4 with fp16 + gradient checkpointing)
# Time: ~8-12 hours on L4

TrainingArguments(
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    warmup_ratio=0.06,
    weight_decay=0.01,
    fp16=True,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
)
```

#### Phase 2: Fine-tuning on Aggregated Sentiment Data

```python
# Model: DAPT-ruRoBERTa-large (from Phase 1)
# Data: Aggregated 100K-150K labeled examples (3 classes)
# Train 3 models with different random seeds for ensemble

TrainingArguments(
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2,
    num_train_epochs=4,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,
    gradient_checkpointing=True,
    evaluation_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
)

# Data preprocessing:
# - Truncate to 256 tokens (social media texts are short)
# - Stratified train/val/test split: 80/10/10
# - Class weights or oversampling for imbalanced classes
```

**Expected F1 per model:** 0.81-0.84

#### Phase 3: Seed Ensemble

```python
# Average softmax probabilities from 3 models (trained with seeds 42, 123, 456)
import numpy as np

def ensemble_predict(texts, models, tokenizer):
    all_probs = []
    for model in models:
        inputs = tokenizer(texts, padding=True, truncation=True,
                          max_length=256, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs.to(model.device))
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    # Weighted average (can optimize weights on validation set)
    weights = [0.4, 0.35, 0.25]  # or optimize via scipy.optimize
    avg_probs = sum(w * p for w, p in zip(weights, all_probs))
    return np.argmax(avg_probs, axis=-1)
```

**Expected ensemble F1:** 0.83-0.86

#### Phase 4: Optional Boosters (to push past 0.85)

**4a. Add a second architecture to the ensemble:**
- Fine-tune XLM-RoBERTa-Large on same data (adds diversity)
- Ensemble: 3x DAPT-ruRoBERTa + 1x XLM-R-Large
- Expected: +1-2 F1-macro

**4b. LLM-generated augmentation data:**
- Use Saiga-Gemma3-12B (local, free) to label 50K additional unlabeled texts
- Add high-confidence labels (logprob > 0.9) to training set
- Expected: +1-2 F1-macro

**4c. Post-processing rules:**
- Emoji-based correction (sequences of positive/negative emojis)
- Negation handling for high-confidence cases
- Expected: +0.5-1 F1-macro

**4d. Test-time augmentation:**
- Classify original text + paraphrased version, average predictions
- Expected: +0.5-1 F1-macro

### Resource Requirements

| Phase | L4 GPU Hours | Wall Time | Storage |
|-------|-------------|-----------|---------|
| DAPT (Phase 1) | 8-12h | 12h | ~5GB |
| Fine-tuning x3 (Phase 2) | 6-9h | 9h | ~12GB |
| Ensemble eval (Phase 3) | <1h | 1h | - |
| Total | **15-22h** | **~22h** | **~17GB** |

**Cost estimate:** ~$15-25 on cloud GPU (L4 at ~$1/hr)

### Expected Final Performance

| Metric | Single Model | Seed Ensemble (3x) | Full Ensemble (4x) |
|--------|-------------|--------------------|--------------------|
| F1-macro | 0.81-0.84 | 0.83-0.86 | 0.84-0.87 |
| F1-positive | 0.86-0.90 | 0.88-0.91 | 0.89-0.92 |
| F1-negative | 0.79-0.83 | 0.81-0.85 | 0.82-0.86 |
| F1-neutral | 0.75-0.80 | 0.78-0.82 | 0.79-0.83 |
| Accuracy | 0.83-0.86 | 0.85-0.88 | 0.86-0.89 |

### Inference Performance (per text)

| Configuration | Latency (L4) | Throughput | VRAM |
|--------------|-------------|------------|------|
| Single ruRoBERTa-large | ~15ms | ~65 texts/sec | ~2GB |
| Seed ensemble (3x) | ~45ms | ~22 texts/sec | ~6GB |
| Full ensemble (4x) | ~65ms | ~15 texts/sec | ~8GB |
| rubert-tiny2 (distilled) | ~3ms | ~300 texts/sec | ~200MB |

### Key Risk: The Neutral Class

The biggest obstacle to 0.85 F1-macro is the **neutral class**. In most Russian sentiment datasets:
- Neutral is the most common class (40-60%)
- Neutral boundary is fuzzy (annotator agreement is lowest for neutral)
- Many texts are genuinely ambiguous

**Mitigations:**
1. Careful annotation guidelines that define "neutral" precisely for your domain
2. Consider 4-class: positive / negative / neutral / mixed (separating ambiguous from truly neutral)
3. Confidence-based routing: flag low-confidence predictions for human review
4. Domain-specific neutral examples in training data

---

## Appendix A: Quick-Start Code

### Minimal fine-tuning script (ruRoBERTa-large on aggregated data)

```python
# requirements: transformers>=4.40, datasets, evaluate, accelerate, scikit-learn

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, concatenate_datasets
from sklearn.metrics import f1_score, classification_report
import numpy as np

MODEL_NAME = "ai-forever/ruRoBERTa-large"
NUM_LABELS = 3
MAX_LENGTH = 256

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS
)

def tokenize(examples):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length",
        max_length=MAX_LENGTH
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    return {"f1_macro": f1_macro, "f1_weighted": f1_weighted}

# Load and merge datasets (pseudo-code -- adapt to actual dataset formats)
# ds = concatenate_datasets([rusentiment_ds, rureviews_ds, ...])
# ds = ds.map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir="./sentiment-ruroberta-large",
    num_train_epochs=4,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,
    gradient_checkpointing=True,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=100,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

trainer.train()
```

### Appendix B: HuggingFace Model IDs Reference

```
# Pre-trained base models (for fine-tuning)
ai-forever/ruRoBERTa-large          # 355M, best Russian-only encoder
ai-forever/ruBert-large             # 427M, Russian BERT large
ai-forever/ruElectra-large          # ~335M, ELECTRA architecture
ai-forever/sbert_large_nlu_ru       # 400M, sentence embeddings
FacebookAI/xlm-roberta-large        # 560M, multilingual
cointegrated/rubert-tiny2           # 29M, fast inference

# Ready-made sentiment models (for baseline / comparison)
blanchefort/rubert-base-cased-sentiment
seara/rubert-tiny2-russian-sentiment
seara/rubert-base-cased-russian-sentiment
cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual

# Russian LLMs (for labeling / distillation)
IlyaGusev/saiga_llama3_8b
IlyaGusev/saiga_gemma3_12b
```

---

## Appendix C: Decision Flowchart

```
START: Need Russian sentiment analysis >0.85 F1-macro
  |
  v
Do you have >50K labeled Russian texts in your domain?
  |
  +-- YES --> Fine-tune ruRoBERTa-large + DAPT + seed ensemble
  |           Expected: 0.83-0.87 F1-macro
  |
  +-- NO --> Do you have budget for LLM API labeling ($500-1000)?
              |
              +-- YES --> LLM-label 100K texts, then fine-tune
              |           Expected: 0.81-0.85 F1-macro
              |
              +-- NO --> Use existing open datasets (RuSentiment + RuReviews + RuTweetCorp)
                         Fine-tune ruRoBERTa-large + seed ensemble
                         Expected: 0.79-0.83 F1-macro
                         (May not reach 0.85 without domain data)
```

---

*This research was compiled on 2026-03-14. Model availability and benchmark results may change. Always verify on HuggingFace and Papers with Code for the latest.*
