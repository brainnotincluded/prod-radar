"""
Experiment V2: Two-stage hierarchical classifier for Russian sentiment.

Architecture:
  Stage 1: Binary classifier — neutral vs sentiment (positive/negative)
  Stage 2: Binary classifier — positive vs negative (only for non-neutral)

Hypothesis:
  Separating neutral detection (the hardest class) from pos/neg classification
  improves overall accuracy. The single 3-class model wastes capacity confusing
  neutral with negative; dedicated binary heads should each be sharper.

Pipeline at inference:
  text -> Stage1(neutral?) --yes--> predict neutral (label=2)
                           --no---> Stage2(pos/neg?) -> predict positive(0) or negative(1)

Data format: {"text": "...", "label": 0|1|2}
  0 = positive, 1 = negative, 2 = neutral

Run: cd ml-service && python3 experiments/v2_hierarchical.py
"""

import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
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
TRAIN_DATA = Path("data/merged_train.jsonl")
VAL_DATA = Path("data/merged_val.jsonl")
TEST_DATA = Path("data/merged_test.jsonl")
OUTPUT_DIR = Path("models/v2_hierarchical")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "cointegrated/rubert-tiny2"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 2e-5
SEED = 42
EVAL_STEPS = 500
EARLY_STOPPING_PATIENCE = 3

# Original 3-class labels
SENTIMENT_LABELS = ["positive", "negative", "neutral"]

# Stage 1: neutral (1) vs sentiment (0)
STAGE1_LABELS = ["sentiment", "neutral"]

# Stage 2: positive (0) vs negative (1)
STAGE2_LABELS = ["positive", "negative"]

# Baseline from v1 single-model approach
BASELINE_F1 = 0.758


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SentimentDataset(torch.utils.data.Dataset):
    """Simple dataset: stores pre-tokenized encodings + labels."""

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
# Metrics
# ---------------------------------------------------------------------------
def make_compute_metrics(label_names):
    """Factory: returns a compute_metrics function for the given label set."""

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        f1_per_class = f1_score(labels, preds, average=None)
        result = {
            "f1_macro": f1_score(labels, preds, average="macro"),
            "f1_weighted": f1_score(labels, preds, average="weighted"),
        }
        for i, name in enumerate(label_names):
            if i < len(f1_per_class):
                result[f"f1_{name}"] = float(f1_per_class[i])
        return result

    return compute_metrics


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_jsonl(path: Path):
    """Load JSONL file with format: {"text": "...", "label": 0|1|2}"""
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            texts.append(item["text"])
            labels.append(item["label"])
    return texts, labels


def make_stage1_data(texts, labels):
    """Convert 3-class labels to binary: neutral (1) vs sentiment (0).

    Original: 0=positive, 1=negative, 2=neutral
    Stage 1:  0=sentiment (was positive or negative), 1=neutral (was neutral)
    """
    stage1_labels = [1 if lbl == 2 else 0 for lbl in labels]
    return texts, stage1_labels


def make_stage2_data(texts, labels):
    """Filter to non-neutral samples, keep binary: positive (0) vs negative (1).

    Original: 0=positive, 1=negative
    Stage 2:  0=positive, 1=negative (same mapping, just filter out neutral)
    """
    s2_texts, s2_labels = [], []
    for text, lbl in zip(texts, labels):
        if lbl != 2:  # skip neutral
            s2_texts.append(text)
            s2_labels.append(lbl)  # 0=positive, 1=negative (unchanged)
    return s2_texts, s2_labels


# ---------------------------------------------------------------------------
# Weighted Trainer
# ---------------------------------------------------------------------------
class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy loss."""

    def __init__(self, *args, class_weights_tensor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights_tensor = class_weights_tensor

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits, labels, weight=self.class_weights_tensor
        )
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Train one stage
# ---------------------------------------------------------------------------
def train_stage(
    stage_name: str,
    train_texts: list,
    train_labels: list,
    val_texts: list,
    val_labels: list,
    label_names: list,
    output_dir: Path,
    device: str,
):
    """Train a single binary classifier stage."""
    num_labels = len(label_names)
    log.info("=" * 60)
    log.info(f"TRAINING {stage_name}: {' vs '.join(label_names)} ({num_labels}-class)")
    log.info("=" * 60)

    # Label distribution
    train_dist = Counter(train_labels)
    val_dist = Counter(val_labels)
    log.info(f"  Train: {len(train_texts)} samples — {dict(train_dist)}")
    log.info(f"  Val:   {len(val_texts)} samples — {dict(val_dist)}")

    # Class weights
    classes = np.arange(num_labels)
    class_weights = compute_class_weight(
        "balanced", classes=classes, y=np.array(train_labels)
    )
    log.info(f"  Class weights: {dict(zip(label_names, class_weights))}")
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    log.info("  Tokenizing train...")
    train_enc = tokenizer(
        train_texts, truncation=True, padding="max_length",
        max_length=MAX_SEQ_LENGTH, return_tensors="pt",
    )
    log.info("  Tokenizing val...")
    val_enc = tokenizer(
        val_texts, truncation=True, padding="max_length",
        max_length=MAX_SEQ_LENGTH, return_tensors="pt",
    )

    train_dataset = SentimentDataset(train_enc, train_labels)
    val_dataset = SentimentDataset(val_enc, val_labels)

    # Model
    log.info(f"  Loading model: {BASE_MODEL} (num_labels={num_labels})")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=num_labels, ignore_mismatched_sizes=True,
    )
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}
    model.config.id2label = id2label
    model.config.label2id = label2id

    # Training args
    stage_output = output_dir / "checkpoints"
    training_args = TrainingArguments(
        output_dir=str(stage_output),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=EVAL_STEPS,
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

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=make_compute_metrics(label_names),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
        class_weights_tensor=weights_tensor,
    )

    log.info(f"  Starting training ({EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LEARNING_RATE})...")
    trainer.train()

    # Save
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    log.info(f"  Model saved to {final_dir}")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Hierarchical inference
# ---------------------------------------------------------------------------
def hierarchical_predict(
    texts: list,
    stage1_model,
    stage2_model,
    tokenizer,
    device: str,
    batch_size: int = 128,
) -> np.ndarray:
    """
    Two-stage hierarchical inference.

    Stage 1: predict neutral (1) vs sentiment (0)
    Stage 2: for sentiment samples, predict positive (0) vs negative (1)

    Returns: array of original 3-class labels (0=positive, 1=negative, 2=neutral)
    """
    stage1_model.eval()
    stage2_model.eval()

    all_preds = np.zeros(len(texts), dtype=np.int64)

    # --- Stage 1: neutral vs sentiment ---
    log.info("  Stage 1: classifying neutral vs sentiment...")
    stage1_preds = _batch_predict(stage1_model, tokenizer, texts, device, batch_size)

    # neutral (stage1 pred = 1) -> final label 2
    neutral_mask = stage1_preds == 1
    all_preds[neutral_mask] = 2
    log.info(f"    Neutral: {neutral_mask.sum()} / {len(texts)} ({neutral_mask.mean() * 100:.1f}%)")

    # --- Stage 2: positive vs negative (only for sentiment samples) ---
    sentiment_indices = np.where(~neutral_mask)[0]
    if len(sentiment_indices) > 0:
        sentiment_texts = [texts[i] for i in sentiment_indices]
        log.info(f"  Stage 2: classifying {len(sentiment_texts)} sentiment samples as pos/neg...")
        stage2_preds = _batch_predict(stage2_model, tokenizer, sentiment_texts, device, batch_size)

        # stage2: 0=positive, 1=negative (matches original labels)
        for idx, pred in zip(sentiment_indices, stage2_preds):
            all_preds[idx] = pred
    else:
        log.info("  Stage 2: no sentiment samples to classify (all predicted neutral)")

    return all_preds


def _batch_predict(
    model, tokenizer, texts: list, device: str, batch_size: int
) -> np.ndarray:
    """Run batched inference, return predicted class indices."""
    all_preds = []
    model.to(device)

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = model(**encodings)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()

    # --- Device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    if device == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Load Data ──────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("LOADING DATA")
    log.info("=" * 60)

    train_texts, train_labels = load_jsonl(TRAIN_DATA)
    val_texts, val_labels = load_jsonl(VAL_DATA)
    test_texts, test_labels = load_jsonl(TEST_DATA)

    log.info(f"Train: {len(train_texts)} samples")
    log.info(f"Val:   {len(val_texts)} samples")
    log.info(f"Test:  {len(test_texts)} samples")

    train_dist = Counter(train_labels)
    log.info(f"Train distribution: { {SENTIMENT_LABELS[k]: v for k, v in sorted(train_dist.items())} }")

    # ── Prepare Stage Data ─────────────────────────────────────────────
    log.info("")
    log.info("Preparing stage-specific datasets...")

    # Stage 1: neutral vs sentiment
    s1_train_texts, s1_train_labels = make_stage1_data(train_texts, train_labels)
    s1_val_texts, s1_val_labels = make_stage1_data(val_texts, val_labels)

    s1_train_dist = Counter(s1_train_labels)
    log.info(f"Stage 1 train: {dict(s1_train_dist)} (0=sentiment, 1=neutral)")

    # Stage 2: positive vs negative (non-neutral only)
    s2_train_texts, s2_train_labels = make_stage2_data(train_texts, train_labels)
    s2_val_texts, s2_val_labels = make_stage2_data(val_texts, val_labels)

    s2_train_dist = Counter(s2_train_labels)
    log.info(f"Stage 2 train: {dict(s2_train_dist)} (0=positive, 1=negative)")

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 1: Neutral vs Sentiment
    # ══════════════════════════════════════════════════════════════════
    stage1_dir = OUTPUT_DIR / "stage1_neutral_detector"

    stage1_model, tokenizer = train_stage(
        stage_name="STAGE 1 (neutral detector)",
        train_texts=s1_train_texts,
        train_labels=s1_train_labels,
        val_texts=s1_val_texts,
        val_labels=s1_val_labels,
        label_names=STAGE1_LABELS,
        output_dir=stage1_dir,
        device=device,
    )

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 2: Positive vs Negative
    # ══════════════════════════════════════════════════════════════════
    stage2_dir = OUTPUT_DIR / "stage2_posneg"

    stage2_model, _ = train_stage(
        stage_name="STAGE 2 (positive vs negative)",
        train_texts=s2_train_texts,
        train_labels=s2_train_labels,
        val_texts=s2_val_texts,
        val_labels=s2_val_labels,
        label_names=STAGE2_LABELS,
        output_dir=stage2_dir,
        device=device,
    )

    # ══════════════════════════════════════════════════════════════════
    #  COMBINED EVALUATION ON TEST SET
    # ══════════════════════════════════════════════════════════════════
    log.info("")
    log.info("=" * 60)
    log.info("COMBINED HIERARCHICAL EVALUATION ON TEST SET")
    log.info("=" * 60)

    combined_preds = hierarchical_predict(
        texts=test_texts,
        stage1_model=stage1_model,
        stage2_model=stage2_model,
        tokenizer=tokenizer,
        device=device,
        batch_size=BATCH_SIZE * 2,
    )

    test_labels_arr = np.array(test_labels)

    report_str = classification_report(
        test_labels_arr, combined_preds, target_names=SENTIMENT_LABELS, digits=4,
    )
    report_dict = classification_report(
        test_labels_arr, combined_preds, target_names=SENTIMENT_LABELS, output_dict=True,
    )
    log.info(f"\n{report_str}")

    f1_macro = report_dict["macro avg"]["f1-score"]
    f1_weighted = report_dict["weighted avg"]["f1-score"]
    f1_pos = report_dict["positive"]["f1-score"]
    f1_neg = report_dict["negative"]["f1-score"]
    f1_neu = report_dict["neutral"]["f1-score"]

    # ── Also evaluate each stage individually ─────────────────────────
    log.info("--- Stage 1 standalone evaluation (on test set) ---")
    s1_test_texts, s1_test_labels = make_stage1_data(test_texts, test_labels)
    s1_test_preds = _batch_predict(stage1_model, tokenizer, s1_test_texts, device, BATCH_SIZE * 2)
    s1_report = classification_report(
        s1_test_labels, s1_test_preds, target_names=STAGE1_LABELS, digits=4,
    )
    s1_report_dict = classification_report(
        s1_test_labels, s1_test_preds, target_names=STAGE1_LABELS, output_dict=True,
    )
    log.info(f"\n{s1_report}")

    log.info("--- Stage 2 standalone evaluation (on non-neutral test samples) ---")
    s2_test_texts, s2_test_labels = make_stage2_data(test_texts, test_labels)
    s2_test_preds = _batch_predict(stage2_model, tokenizer, s2_test_texts, device, BATCH_SIZE * 2)
    s2_report = classification_report(
        s2_test_labels, s2_test_preds, target_names=STAGE2_LABELS, digits=4,
    )
    s2_report_dict = classification_report(
        s2_test_labels, s2_test_preds, target_names=STAGE2_LABELS, output_dict=True,
    )
    log.info(f"\n{s2_report}")

    # ── Save Combined Metrics ─────────────────────────────────────────
    metrics = {
        "experiment": "v2_hierarchical",
        "hypothesis": "Separating neutral detection from pos/neg improves overall accuracy",
        "architecture": "Stage1(neutral_detector) -> Stage2(pos_neg_classifier)",
        "base_model": BASE_MODEL,
        "config": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "max_seq_length": MAX_SEQ_LENGTH,
            "fp16": device == "cuda",
            "eval_steps": EVAL_STEPS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        },
        "data": {
            "train_total": len(train_texts),
            "val_total": len(val_texts),
            "test_total": len(test_texts),
            "stage1_train": len(s1_train_texts),
            "stage2_train": len(s2_train_texts),
            "stage2_val": len(s2_val_texts),
        },
        "combined_results": {
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "f1_positive": round(f1_pos, 4),
            "f1_negative": round(f1_neg, 4),
            "f1_neutral": round(f1_neu, 4),
        },
        "stage1_results": {
            "f1_macro": round(s1_report_dict["macro avg"]["f1-score"], 4),
            "f1_neutral": round(s1_report_dict["neutral"]["f1-score"], 4),
            "f1_sentiment": round(s1_report_dict["sentiment"]["f1-score"], 4),
        },
        "stage2_results": {
            "f1_macro": round(s2_report_dict["macro avg"]["f1-score"], 4),
            "f1_positive": round(s2_report_dict["positive"]["f1-score"], 4),
            "f1_negative": round(s2_report_dict["negative"]["f1-score"], 4),
        },
        "baseline_f1_macro": BASELINE_F1,
        "improvement_over_baseline": round(f1_macro - BASELINE_F1, 4),
        "combined_report": report_dict,
        "stage1_report": s1_report_dict,
        "stage2_report": s2_report_dict,
    }

    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    log.info(f"\nMetrics saved to {metrics_path}")

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    log.info("")
    log.info("=" * 60)
    log.info("V2 HIERARCHICAL — RESULTS vs BASELINE")
    log.info("=" * 60)
    log.info(f"")
    log.info(f"  {'Metric':<20} {'Baseline':>10} {'V2 Hier.':>10} {'Delta':>10}")
    log.info(f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 10}")
    log.info(f"  {'F1-macro':<20} {BASELINE_F1:>10.4f} {f1_macro:>10.4f} {f1_macro - BASELINE_F1:>+10.4f}")
    log.info(f"  {'F1-positive':<20} {'—':>10} {f1_pos:>10.4f} {'':>10}")
    log.info(f"  {'F1-negative':<20} {'—':>10} {f1_neg:>10.4f} {'':>10}")
    log.info(f"  {'F1-neutral':<20} {'—':>10} {f1_neu:>10.4f} {'':>10}")
    log.info(f"  {'F1-weighted':<20} {'—':>10} {f1_weighted:>10.4f} {'':>10}")
    log.info(f"")
    log.info(f"  Stage 1 (neutral detector):    F1-macro = {s1_report_dict['macro avg']['f1-score']:.4f}")
    log.info(f"  Stage 2 (pos/neg classifier):  F1-macro = {s2_report_dict['macro avg']['f1-score']:.4f}")
    log.info(f"")

    if f1_macro >= 0.80:
        log.info("  >>> TARGET REACHED: F1-macro >= 0.80 <<<")
    elif f1_macro > BASELINE_F1:
        log.info(f"  >>> IMPROVED over baseline by {f1_macro - BASELINE_F1:+.4f} <<<")
    else:
        log.info(f"  >>> NO IMPROVEMENT over baseline <<<")

    log.info(f"")
    log.info(f"  Training time: {elapsed / 60:.1f} minutes")
    log.info(f"  Models saved to:")
    log.info(f"    Stage 1: {stage1_dir / 'final'}")
    log.info(f"    Stage 2: {stage2_dir / 'final'}")
    log.info(f"  Combined metrics: {metrics_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
