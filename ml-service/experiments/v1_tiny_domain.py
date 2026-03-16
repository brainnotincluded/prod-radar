"""
Experiment V1: rubert-tiny2 trained ONLY on in-domain data (Мобайл 188K dataset).

Hypothesis: A small model (29M params) fine-tuned exclusively on our domain data
            will be more accurate for mobile-operator sentiment than a larger model
            trained on mixed/general data.

Dataset:    data/dataset.xlsx
            Columns: Заголовок, Текст, Тональность
Base model: cointegrated/rubert-tiny2 (29M params)
Labels:     позитив→0, негатив→1, нейтрально→2
Split:      85% train / 7.5% val / 7.5% test (stratified)

Run:
  cd ml-service && python3 experiments/v1_tiny_domain.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = Path("data/dataset.xlsx")
# Output dir is relative to ml-service/ (the expected working directory)
OUTPUT_DIR = Path("models/v1_tiny_domain")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "cointegrated/rubert-tiny2"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 2e-5
SEED = 42

SENTIMENT_MAP = {"позитив": 0, "негатив": 1, "нейтрально": 2}
SENTIMENT_LABELS = ["positive", "negative", "neutral"]

# Previous best result for comparison (rubert-tiny2 + augmented data, finetune_v2.py)
BASELINE_F1 = 0.7579


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SentimentDataset(torch.utils.data.Dataset):
    """Simple dataset wrapping pre-tokenized encodings and integer labels."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Metrics callback
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1_per_class = f1_score(labels, preds, average=None)
    return {
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "f1_positive": float(f1_per_class[0]),
        "f1_negative": float(f1_per_class[1]),
        "f1_neutral": float(f1_per_class[2]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    if device == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── 1. Load Data ──────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("LOADING DATA")
    log.info("=" * 60)
    log.info(f"Source: {DATA_PATH}")

    df = pd.read_excel(DATA_PATH)
    log.info(f"Raw rows: {len(df)}")

    # ── 2. Combine Заголовок + Текст ─────────────────────────────────
    df["text"] = (
        df["Заголовок"].fillna("").astype(str)
        + " "
        + df["Текст"].fillna("").astype(str)
    ).str.strip()

    # Drop rows with too-short text (likely empty/corrupted)
    df = df[df["text"].str.len() > 5].copy()

    # ── 3. Map Тональность to integer labels ─────────────────────────
    df["sentiment_raw"] = df["Тональность"].str.strip().str.lower()
    df = df[df["sentiment_raw"].isin(SENTIMENT_MAP.keys())].copy()
    df["label"] = df["sentiment_raw"].map(SENTIMENT_MAP)
    df = df[["text", "label"]].reset_index(drop=True)

    log.info(f"After cleaning: {len(df)} rows")
    log.info(f"Label distribution:\n{df['label'].value_counts().sort_index().to_string()}")
    for label_name, label_id in SENTIMENT_MAP.items():
        count = (df["label"] == label_id).sum()
        log.info(f"  {label_name} ({label_id}): {count} ({count / len(df) * 100:.1f}%)")

    # ── 4. Stratified Split: 85% train, 7.5% val, 7.5% test ─────────
    log.info("=" * 60)
    log.info("SPLITTING DATA (85/7.5/7.5 stratified)")
    log.info("=" * 60)

    train_df, temp_df = train_test_split(
        df, test_size=0.15, random_state=SEED, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["label"]
    )

    # Shuffle training data
    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    log.info(f"Train: {len(train_df)}")
    log.info(f"Val:   {len(val_df)}")
    log.info(f"Test:  {len(test_df)}")

    # ── 5a. Class Weights (balanced) ──────────────────────────────────
    class_weights = compute_class_weight(
        "balanced",
        classes=np.array([0, 1, 2]),
        y=train_df["label"].values,
    )
    log.info(f"Class weights: {dict(zip(SENTIMENT_LABELS, class_weights.round(4)))}")
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # ── 5b. Tokenize ──────────────────────────────────────────────────
    log.info(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    log.info("Tokenizing...")
    train_enc = tokenizer(
        train_df["text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",
    )
    val_enc = tokenizer(
        val_df["text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",
    )
    test_enc = tokenizer(
        test_df["text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",
    )

    train_dataset = SentimentDataset(train_enc, train_df["label"].values)
    val_dataset = SentimentDataset(val_enc, val_df["label"].values)
    test_dataset = SentimentDataset(test_enc, test_df["label"].values)

    # ── 5c. Model ─────────────────────────────────────────────────────
    log.info(f"Loading model: {BASE_MODEL}")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=3, ignore_mismatched_sizes=True
    )
    model.config.id2label = {0: "positive", 1: "negative", 2: "neutral"}
    model.config.label2id = {"positive": 0, "negative": 1, "neutral": 2}

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Total params: {total_params:,}")
    log.info(f"Trainable params: {trainable_params:,}")

    # ── 5d. Training Args ─────────────────────────────────────────────
    # Compute eval_steps: evaluate ~4 times per epoch
    steps_per_epoch = len(train_dataset) // BATCH_SIZE
    eval_steps = max(steps_per_epoch // 4, 100)
    log.info(f"Steps per epoch: {steps_per_epoch}, eval every {eval_steps} steps")

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=100,
        fp16=(device == "cuda"),
        dataloader_num_workers=4,
        report_to="none",
        seed=SEED,
    )

    # ── 5e. Weighted Trainer ──────────────────────────────────────────
    class WeightedTrainer(Trainer):
        """Trainer with weighted cross-entropy to handle class imbalance."""

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits, labels, weight=weights_tensor
            )
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ── 5f. Train ─────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("EXPERIMENT V1: rubert-tiny2 on DOMAIN DATA ONLY")
    log.info("=" * 60)
    log.info(f"  Model:         {BASE_MODEL} ({trainable_params:,} params)")
    log.info(f"  Data:          {len(train_dataset)} train / {len(val_dataset)} val / {len(test_dataset)} test")
    log.info(f"  Batch size:    {BATCH_SIZE}")
    log.info(f"  Epochs:        {EPOCHS}")
    log.info(f"  LR:            {LEARNING_RATE}")
    log.info(f"  Max seq len:   {MAX_SEQ_LENGTH}")
    log.info(f"  FP16:          {device == 'cuda'}")
    log.info(f"  Loss:          Weighted CE (balanced)")
    log.info(f"  Early stop:    patience=3 on f1_macro")
    log.info("=" * 60)

    trainer.train()

    # ── 6. Evaluate on Test Set ───────────────────────────────────────
    log.info("=" * 60)
    log.info("EVALUATING ON TEST SET")
    log.info("=" * 60)

    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)

    report_str = classification_report(
        test_df["label"].values, preds, target_names=SENTIMENT_LABELS, digits=4
    )
    report_dict = classification_report(
        test_df["label"].values, preds, target_names=SENTIMENT_LABELS, output_dict=True
    )
    log.info(f"\n{report_str}")

    f1_macro = report_dict["macro avg"]["f1-score"]
    f1_weighted = report_dict["weighted avg"]["f1-score"]
    f1_neg = report_dict["negative"]["f1-score"]

    # ── 6. Save Model ─────────────────────────────────────────────────
    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Saving model to {final_dir}")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # ── 7. Save metrics.json ──────────────────────────────────────────
    metrics = {
        "experiment": "v1_tiny_domain",
        "hypothesis": "Small model (rubert-tiny2) on in-domain data only is more accurate than bigger model on mixed data",
        "base_model": BASE_MODEL,
        "base_model_params": total_params,
        "data_source": str(DATA_PATH),
        "data_description": "Мобайл X-Prod (mobile operator reviews)",
        "data_domain_only": True,
        "total_rows": len(df),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "config": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "max_seq_length": MAX_SEQ_LENGTH,
            "loss": "weighted_cross_entropy",
            "early_stopping_patience": 3,
            "fp16": device == "cuda",
            "seed": SEED,
        },
        "class_weights": {
            SENTIMENT_LABELS[i]: round(float(class_weights[i]), 4)
            for i in range(3)
        },
        "results": {
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "f1_positive": round(report_dict["positive"]["f1-score"], 4),
            "f1_negative": round(f1_neg, 4),
            "f1_neutral": round(report_dict["neutral"]["f1-score"], 4),
        },
        "baseline_f1_macro": BASELINE_F1,
        "improvement_over_baseline": round(f1_macro - BASELINE_F1, 4),
        "report": report_dict,
    }

    metrics_path = final_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t_start

    log.info("=" * 60)
    log.info("V1 EXPERIMENT RESULTS vs BASELINE")
    log.info("=" * 60)
    log.info(f"")
    log.info(f"  {'Metric':<20} {'Baseline':>10} {'V1 Domain':>10} {'Delta':>10}")
    log.info(f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 10}")
    log.info(f"  {'F1-macro':<20} {BASELINE_F1:>10.4f} {f1_macro:>10.4f} {f1_macro - BASELINE_F1:>+10.4f}")
    log.info(f"  {'F1-negative':<20} {'0.7056':>10} {f1_neg:>10.4f}")
    log.info(f"  {'F1-positive':<20} {'':>10} {report_dict['positive']['f1-score']:>10.4f}")
    log.info(f"  {'F1-neutral':<20} {'':>10} {report_dict['neutral']['f1-score']:>10.4f}")
    log.info(f"  {'F1-weighted':<20} {'':>10} {f1_weighted:>10.4f}")
    log.info(f"")

    if f1_macro > BASELINE_F1:
        log.info(f"  >>> HYPOTHESIS SUPPORTED: domain-only improved by {f1_macro - BASELINE_F1:+.4f} <<<")
    else:
        log.info(f"  >>> HYPOTHESIS NOT SUPPORTED: domain-only did not beat baseline <<<")

    log.info(f"")
    log.info(f"  Training time: {elapsed / 60:.1f} minutes")
    log.info(f"  Model saved to: {final_dir}")
    log.info(f"  Metrics saved to: {metrics_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
