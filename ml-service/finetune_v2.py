"""
Fine-tune v2: rubert-tiny2 on 188K original + 6.8K synthetic augmented data.
Run: cd ~/prod-radar-ml && source venv/bin/activate && python3 src/finetune_v2.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────

ORIGINAL_DATA = Path("data/dataset.xlsx")
AUGMENTED_DATA = Path("data/augmented_merged.jsonl")
OUTPUT_DIR = Path("models/sentiment-v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "cointegrated/rubert-tiny2-cedr-emotion-detection"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 2e-5
SEED = 42

SENTIMENT_MAP = {"позитив": 0, "негатив": 1, "нейтрально": 2}
SENTIMENT_LABELS = ["positive", "negative", "neutral"]


# ─── Dataset ──────────────────────────────────────────────────────────

class SentimentDataset(torch.utils.data.Dataset):
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


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "f1_negative": f1_score(labels, preds, average=None)[1],
    }


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # ── Load Original Data ────────────────────────────────────────────
    log.info(f"Loading original data: {ORIGINAL_DATA}")
    df = pd.read_excel(ORIGINAL_DATA)
    df["text"] = (df["Заголовок"].fillna("") + " " + df["Текст"].fillna("")).str.strip()
    df = df[df["text"].str.len() > 5].copy()
    df["sentiment_raw"] = df["Тональность"].str.strip().str.lower()
    df = df[df["sentiment_raw"].isin(SENTIMENT_MAP.keys())].copy()
    df["label"] = df["sentiment_raw"].map(SENTIMENT_MAP)
    df = df[["text", "label"]].copy()
    df["source"] = "original"
    log.info(f"Original: {len(df)} rows")

    # ── Load Augmented Data ───────────────────────────────────────────
    log.info(f"Loading augmented data: {AUGMENTED_DATA}")
    aug_rows = []
    with open(AUGMENTED_DATA, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            sentiment = item["sentiment"]
            if sentiment in SENTIMENT_MAP:
                aug_rows.append({
                    "text": item["text"],
                    "label": SENTIMENT_MAP[sentiment],
                    "source": "augmented",
                })
    aug_df = pd.DataFrame(aug_rows)
    log.info(f"Augmented: {len(aug_df)} rows")

    # ── Combine ───────────────────────────────────────────────────────
    combined = pd.concat([df, aug_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)
    log.info(f"Combined (deduplicated): {len(combined)} rows")
    log.info(f"Label distribution:\n{combined['label'].value_counts().sort_index()}")

    # ── Split: augmented data goes ONLY into training ─────────────────
    # Test set should be original data only (for fair evaluation)
    original_mask = combined["source"] == "original"
    original_data = combined[original_mask]
    augmented_data = combined[~original_mask]

    # Split original into train/val/test
    orig_train, orig_temp = train_test_split(
        original_data, test_size=0.15, random_state=SEED, stratify=original_data["label"]
    )
    orig_val, orig_test = train_test_split(
        orig_temp, test_size=0.5, random_state=SEED, stratify=orig_temp["label"]
    )

    # Add ALL augmented data to training set
    train_df = pd.concat([orig_train, augmented_data], ignore_index=True).sample(
        frac=1, random_state=SEED
    )
    val_df = orig_val
    test_df = orig_test

    log.info(f"Train: {len(train_df)} (original: {len(orig_train)}, augmented: {len(augmented_data)})")
    log.info(f"Val: {len(val_df)} (original only)")
    log.info(f"Test: {len(test_df)} (original only — fair evaluation)")

    # ── Class Weights ─────────────────────────────────────────────────
    class_weights = compute_class_weight(
        "balanced", classes=np.array([0, 1, 2]), y=train_df["label"].values
    )
    log.info(f"Class weights: {class_weights}")
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # ── Tokenize ──────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    log.info("Tokenizing...")
    train_enc = tokenizer(train_df["text"].tolist(), truncation=True, padding="max_length",
                          max_length=MAX_SEQ_LENGTH, return_tensors="pt")
    val_enc = tokenizer(val_df["text"].tolist(), truncation=True, padding="max_length",
                        max_length=MAX_SEQ_LENGTH, return_tensors="pt")
    test_enc = tokenizer(test_df["text"].tolist(), truncation=True, padding="max_length",
                         max_length=MAX_SEQ_LENGTH, return_tensors="pt")

    train_dataset = SentimentDataset(train_enc, train_df["label"].values)
    val_dataset = SentimentDataset(val_enc, val_df["label"].values)
    test_dataset = SentimentDataset(test_enc, test_df["label"].values)

    # ── Model ─────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=3, ignore_mismatched_sizes=True
    )
    model.config.id2label = {0: "positive", 1: "negative", 2: "neutral"}
    model.config.label2id = {"positive": 0, "negative": 1, "neutral": 2}

    # ── Train ─────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
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

    class WeightedTrainer(Trainer):
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

    log.info("=" * 60)
    log.info("STARTING FINE-TUNING V2 (original + augmented)")
    log.info("=" * 60)
    trainer.train()

    # ── Evaluate on ORIGINAL test set ─────────────────────────────────
    log.info("=" * 60)
    log.info("EVALUATING ON ORIGINAL TEST SET (fair comparison)")
    log.info("=" * 60)
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)

    report_str = classification_report(
        test_df["label"].values, preds, target_names=SENTIMENT_LABELS
    )
    report_dict = classification_report(
        test_df["label"].values, preds, target_names=SENTIMENT_LABELS, output_dict=True
    )
    log.info(f"\n{report_str}")

    f1_macro = report_dict["macro avg"]["f1-score"]
    f1_neg = report_dict["negative"]["f1-score"]
    log.info(f"F1-macro: {f1_macro:.4f} (v1 was 0.7579)")
    log.info(f"F1-negative: {f1_neg:.4f} (v1 was 0.7056)")

    # ── Save ──────────────────────────────────────────────────────────
    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    with open(final_dir / "metrics.json", "w") as f:
        json.dump({
            "base_model": BASE_MODEL,
            "original_size": len(df),
            "augmented_size": len(augmented_data),
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "f1_macro": round(f1_macro, 4),
            "f1_negative": round(f1_neg, 4),
            "f1_macro_v1": 0.7579,
            "improvement": round(f1_macro - 0.7579, 4),
            "report": report_dict,
        }, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t_start
    log.info(f"\nDone in {elapsed / 60:.1f} minutes")
    log.info(f"Model saved to {final_dir}")


if __name__ == "__main__":
    main()
