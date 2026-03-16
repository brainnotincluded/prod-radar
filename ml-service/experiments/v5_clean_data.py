"""
VARIANT 5: Aggressive data cleaning before training.

Hypothesis: Removing noisy, mislabeled, and near-duplicate samples from
training data improves model quality more than adding more data.

Cleaning pipeline:
  1. Remove texts shorter than 15 chars or longer than 1000 chars
  2. Remove texts that are >90% non-Cyrillic characters
  3. Remove exact and near-duplicate texts (normalize whitespace, lowercase)
  4. Remove likely mislabeled samples: train a quick 1-epoch probe model,
     find samples where the model is very confident (>0.95) but the label
     disagrees -- these are likely mislabeled

Comparison: train the same model on raw data vs cleaned data.

Model:  cointegrated/rubert-tiny2
Config: batch_size=64, epochs=5, lr=2e-5, max_seq_len=128, fp16
Save:   models/v5_clean_data/final/

Run:
  cd ml-service && python3 experiments/v5_clean_data.py
"""

import json
import logging
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
BASE_DIR = Path(__file__).resolve().parent.parent          # ml-service/
TRAIN_DATA = BASE_DIR / "data" / "merged_train.jsonl"
VAL_DATA = BASE_DIR / "data" / "merged_val.jsonl"
TEST_DATA = BASE_DIR / "data" / "merged_test.jsonl"
OUTPUT_DIR = BASE_DIR / "models" / "v5_clean_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "cointegrated/rubert-tiny2"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 2e-5
SEED = 42

# Cleaning thresholds
MIN_TEXT_LEN = 15
MAX_TEXT_LEN = 1000
NON_CYRILLIC_THRESHOLD = 0.90     # remove if >90% non-Cyrillic
MISLABEL_CONFIDENCE = 0.95        # confidence above which disagreement = mislabel
PROBE_EPOCHS = 1                  # quick probe model for mislabel detection

SENTIMENT_LABELS = ["positive", "negative", "neutral"]  # 0, 1, 2


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
# Data I/O
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file. Each line: {"text": "...", "label": 0|1|2}."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(rows: list[dict], path: Path):
    """Save list of dicts to JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Cleaning functions
# ---------------------------------------------------------------------------
def clean_length(rows: list[dict]) -> list[dict]:
    """Remove texts shorter than MIN_TEXT_LEN or longer than MAX_TEXT_LEN."""
    before = len(rows)
    cleaned = [r for r in rows if MIN_TEXT_LEN <= len(r["text"]) <= MAX_TEXT_LEN]
    removed = before - len(cleaned)
    log.info(f"  [length] Removed {removed} samples "
             f"(< {MIN_TEXT_LEN} or > {MAX_TEXT_LEN} chars). "
             f"{len(cleaned)} remain.")
    return cleaned


def cyrillic_ratio(text: str) -> float:
    """Return fraction of alphabetic characters that are Cyrillic."""
    alpha_chars = [ch for ch in text if ch.isalpha()]
    if not alpha_chars:
        return 0.0
    cyrillic = sum(1 for ch in alpha_chars if "\u0400" <= ch <= "\u04ff")
    return cyrillic / len(alpha_chars)


def clean_non_cyrillic(rows: list[dict]) -> list[dict]:
    """Remove texts where >90% of alphabetic characters are non-Cyrillic."""
    before = len(rows)
    cleaned = [r for r in rows if cyrillic_ratio(r["text"]) >= (1 - NON_CYRILLIC_THRESHOLD)]
    removed = before - len(cleaned)
    log.info(f"  [non-cyrillic] Removed {removed} samples "
             f"(>{NON_CYRILLIC_THRESHOLD*100:.0f}% non-Cyrillic). "
             f"{len(cleaned)} remain.")
    return cleaned


def _normalize_for_dedup(text: str) -> str:
    """Normalize text for deduplication: lowercase, collapse whitespace, strip."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    # Remove punctuation for near-duplicate matching
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_duplicates(rows: list[dict]) -> list[dict]:
    """Remove exact and near-duplicate texts (after normalization)."""
    before = len(rows)
    seen = set()
    cleaned = []
    for r in rows:
        key = _normalize_for_dedup(r["text"])
        if key not in seen:
            seen.add(key)
            cleaned.append(r)
    removed = before - len(cleaned)
    log.info(f"  [duplicates] Removed {removed} exact/near-duplicates. "
             f"{len(cleaned)} remain.")
    return cleaned


def clean_mislabeled(
    rows: list[dict],
    tokenizer,
    device: str,
) -> list[dict]:
    """
    Train a quick 1-epoch probe model, then find samples where the model
    is very confident (>0.95 softmax) but the label disagrees. These are
    likely mislabeled and should be removed.
    """
    before = len(rows)
    texts = [r["text"] for r in rows]
    labels = [r["label"] for r in rows]

    log.info(f"  [mislabel] Training 1-epoch probe model on {before} samples...")

    # Tokenize
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",
    )
    dataset = SentimentDataset(encodings, labels)

    # Quick probe model
    probe_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        ignore_mismatched_sizes=True,
    )

    probe_output_dir = OUTPUT_DIR / "probe_checkpoints"
    probe_output_dir.mkdir(parents=True, exist_ok=True)

    probe_args = TrainingArguments(
        output_dir=str(probe_output_dir),
        num_train_epochs=PROBE_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=200,
        save_strategy="no",
        fp16=(device == "cuda"),
        dataloader_num_workers=4,
        report_to="none",
        seed=SEED,
    )

    probe_trainer = Trainer(
        model=probe_model,
        args=probe_args,
        train_dataset=dataset,
    )
    probe_trainer.train()

    # Get predictions on training data
    log.info("  [mislabel] Getting probe predictions on training data...")
    predictions = probe_trainer.predict(dataset)
    logits = predictions.predictions  # (N, 3)
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()

    # Find mislabeled candidates
    max_probs = probs.max(axis=1)           # confidence
    pred_labels = probs.argmax(axis=1)      # predicted class
    true_labels = np.array(labels)

    # Sample is "likely mislabeled" if the probe is very confident AND disagrees
    mislabel_mask = (max_probs > MISLABEL_CONFIDENCE) & (pred_labels != true_labels)
    mislabel_indices = set(np.where(mislabel_mask)[0])

    # Log some examples of suspected mislabels
    mislabel_examples = [i for i in list(mislabel_indices)[:10]]
    if mislabel_examples:
        log.info(f"  [mislabel] Example suspected mislabels:")
        for idx in mislabel_examples[:5]:
            snippet = rows[idx]["text"][:80].replace("\n", " ")
            log.info(
                f"    idx={idx}, true={SENTIMENT_LABELS[rows[idx]['label']]}, "
                f"pred={SENTIMENT_LABELS[pred_labels[idx]]}, "
                f"conf={max_probs[idx]:.3f}, text=\"{snippet}...\""
            )

    # Per-class breakdown of suspected mislabels
    mislabel_by_class = Counter(rows[i]["label"] for i in mislabel_indices)
    for cls_id, cls_name in enumerate(SENTIMENT_LABELS):
        cnt = mislabel_by_class.get(cls_id, 0)
        log.info(f"    {cls_name}: {cnt} suspected mislabels")

    cleaned = [r for i, r in enumerate(rows) if i not in mislabel_indices]
    removed = before - len(cleaned)
    log.info(f"  [mislabel] Removed {removed} likely mislabeled samples "
             f"(probe conf > {MISLABEL_CONFIDENCE}). {len(cleaned)} remain.")

    # Clean up probe model to free memory
    del probe_model, probe_trainer, predictions, logits, probs
    if device == "cuda":
        torch.cuda.empty_cache()

    return cleaned


# ---------------------------------------------------------------------------
# Metrics
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
# Training helper
# ---------------------------------------------------------------------------
def train_and_evaluate(
    name: str,
    train_rows: list[dict],
    val_rows: list[dict],
    test_rows: list[dict],
    tokenizer,
    device: str,
    output_subdir: Path,
    save_model: bool = False,
) -> dict:
    """Train rubert-tiny2 on given data and evaluate on test set. Return metrics dict."""

    log.info(f"\n{'='*60}")
    log.info(f"TRAINING: {name}")
    log.info(f"{'='*60}")

    train_texts = [r["text"] for r in train_rows]
    train_labels = [r["label"] for r in train_rows]
    val_texts = [r["text"] for r in val_rows]
    val_labels = [r["label"] for r in val_rows]
    test_texts = [r["text"] for r in test_rows]
    test_labels = [r["label"] for r in test_rows]

    log.info(f"  Train: {len(train_texts)} samples")
    log.info(f"  Val:   {len(val_texts)} samples")
    log.info(f"  Test:  {len(test_texts)} samples")

    # Label distribution
    dist = Counter(train_labels)
    log.info(f"  Train distribution: {
        {SENTIMENT_LABELS[k]: v for k, v in sorted(dist.items())}
    }")

    # Class weights
    class_weights = compute_class_weight(
        "balanced",
        classes=np.array([0, 1, 2]),
        y=np.array(train_labels),
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    log.info(f"  Class weights: {dict(zip(SENTIMENT_LABELS, [round(w, 3) for w in class_weights]))}")

    # Tokenize
    log.info("  Tokenizing...")
    train_enc = tokenizer(
        train_texts, truncation=True, padding="max_length",
        max_length=MAX_SEQ_LENGTH, return_tensors="pt",
    )
    val_enc = tokenizer(
        val_texts, truncation=True, padding="max_length",
        max_length=MAX_SEQ_LENGTH, return_tensors="pt",
    )
    test_enc = tokenizer(
        test_texts, truncation=True, padding="max_length",
        max_length=MAX_SEQ_LENGTH, return_tensors="pt",
    )

    train_dataset = SentimentDataset(train_enc, train_labels)
    val_dataset = SentimentDataset(val_enc, val_labels)
    test_dataset = SentimentDataset(test_enc, test_labels)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        ignore_mismatched_sizes=True,
    )
    model.config.id2label = {0: "positive", 1: "negative", 2: "neutral"}
    model.config.label2id = {"positive": 0, "negative": 1, "neutral": 2}

    # Training args
    checkpoint_dir = output_subdir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
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

    # Weighted loss trainer
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = F.cross_entropy(outputs.logits, labels, weight=weights_tensor)
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    log.info(f"  Starting training ({EPOCHS} epochs, bs={BATCH_SIZE}, lr={LEARNING_RATE})...")
    trainer.train()

    # Evaluate
    log.info(f"  Evaluating on test set...")
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)

    report_str = classification_report(
        test_labels, preds, target_names=SENTIMENT_LABELS, digits=4,
    )
    report_dict = classification_report(
        test_labels, preds, target_names=SENTIMENT_LABELS, output_dict=True,
    )
    log.info(f"\n{report_str}")

    results = {
        "name": name,
        "train_size": len(train_texts),
        "f1_macro": round(report_dict["macro avg"]["f1-score"], 4),
        "f1_weighted": round(report_dict["weighted avg"]["f1-score"], 4),
        "f1_positive": round(report_dict["positive"]["f1-score"], 4),
        "f1_negative": round(report_dict["negative"]["f1-score"], 4),
        "f1_neutral": round(report_dict["neutral"]["f1-score"], 4),
        "report": report_dict,
    }

    # Save model if requested
    if save_model:
        final_dir = output_subdir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"  Saving model to {final_dir}")
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        results["model_path"] = str(final_dir)

    # Free GPU memory
    del model, trainer, predictions
    if device == "cuda":
        torch.cuda.empty_cache()

    return results


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

    # --- Load tokenizer early (shared across cleaning + training) ---
    log.info(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # ======================================================================
    # LOAD DATA
    # ======================================================================
    log.info("=" * 60)
    log.info("LOADING DATA")
    log.info("=" * 60)

    if not TRAIN_DATA.exists():
        log.error(
            f"Training data not found: {TRAIN_DATA}\n"
            f"Run 'python3 download_datasets.py' first to generate merged splits."
        )
        sys.exit(1)

    raw_train = load_jsonl(TRAIN_DATA)
    val_rows = load_jsonl(VAL_DATA)
    test_rows = load_jsonl(TEST_DATA)

    log.info(f"Raw train:  {len(raw_train)} samples")
    log.info(f"Val:        {len(val_rows)} samples")
    log.info(f"Test:       {len(test_rows)} samples")

    raw_dist = Counter(r["label"] for r in raw_train)
    log.info(f"Raw train distribution: {
        {SENTIMENT_LABELS[k]: v for k, v in sorted(raw_dist.items())}
    }")

    # ======================================================================
    # AGGRESSIVE DATA CLEANING PIPELINE
    # ======================================================================
    log.info("\n" + "=" * 60)
    log.info("AGGRESSIVE DATA CLEANING PIPELINE")
    log.info("=" * 60)
    log.info(f"Starting with {len(raw_train)} training samples\n")

    # Step 1: Length filter
    log.info("Step 1/4: Length filtering")
    cleaned = clean_length(raw_train)

    # Step 2: Non-Cyrillic filter
    log.info("\nStep 2/4: Non-Cyrillic filtering")
    cleaned = clean_non_cyrillic(cleaned)

    # Step 3: Duplicate removal
    log.info("\nStep 3/4: Duplicate removal (normalize + compare)")
    cleaned = clean_duplicates(cleaned)

    # Step 4: Mislabel detection via probe model
    log.info("\nStep 4/4: Mislabel detection (1-epoch probe)")
    cleaned = clean_mislabeled(cleaned, tokenizer, device)

    # --- Summary ---
    total_removed = len(raw_train) - len(cleaned)
    pct_removed = 100.0 * total_removed / len(raw_train) if raw_train else 0
    log.info(f"\n{'='*60}")
    log.info(f"CLEANING SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"  Original:  {len(raw_train)} samples")
    log.info(f"  Cleaned:   {len(cleaned)} samples")
    log.info(f"  Removed:   {total_removed} samples ({pct_removed:.1f}%)")

    clean_dist = Counter(r["label"] for r in cleaned)
    log.info(f"  Cleaned distribution: {
        {SENTIMENT_LABELS[k]: v for k, v in sorted(clean_dist.items())}
    }")

    # Save cleaned data for reproducibility
    cleaned_path = OUTPUT_DIR / "cleaned_train.jsonl"
    save_jsonl(cleaned, cleaned_path)
    log.info(f"  Cleaned data saved to: {cleaned_path}")

    # ======================================================================
    # TRAIN: RAW DATA (baseline for comparison)
    # ======================================================================
    raw_results = train_and_evaluate(
        name="RAW DATA (baseline)",
        train_rows=raw_train,
        val_rows=val_rows,
        test_rows=test_rows,
        tokenizer=tokenizer,
        device=device,
        output_subdir=OUTPUT_DIR / "raw_baseline",
        save_model=False,
    )

    # ======================================================================
    # TRAIN: CLEANED DATA (experiment)
    # ======================================================================
    clean_results = train_and_evaluate(
        name="CLEANED DATA (v5 experiment)",
        train_rows=cleaned,
        val_rows=val_rows,
        test_rows=test_rows,
        tokenizer=tokenizer,
        device=device,
        output_subdir=OUTPUT_DIR,
        save_model=True,   # this is the model we save
    )

    # ======================================================================
    # COMPARISON
    # ======================================================================
    elapsed = time.time() - t_start

    log.info("\n" + "=" * 60)
    log.info("V5 EXPERIMENT RESULTS: RAW vs CLEANED DATA")
    log.info("=" * 60)
    log.info("")
    log.info(f"  {'Metric':<20} {'Raw':>12} {'Cleaned':>12} {'Delta':>12}")
    log.info(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")

    for metric_key, metric_name in [
        ("f1_macro", "F1-macro"),
        ("f1_weighted", "F1-weighted"),
        ("f1_positive", "F1-positive"),
        ("f1_negative", "F1-negative"),
        ("f1_neutral", "F1-neutral"),
    ]:
        raw_val = raw_results[metric_key]
        clean_val = clean_results[metric_key]
        delta = clean_val - raw_val
        log.info(f"  {metric_name:<20} {raw_val:>12.4f} {clean_val:>12.4f} {delta:>+12.4f}")

    log.info(f"\n  {'Train size':<20} {raw_results['train_size']:>12d} {clean_results['train_size']:>12d} "
             f"{clean_results['train_size'] - raw_results['train_size']:>+12d}")

    delta_f1 = clean_results["f1_macro"] - raw_results["f1_macro"]
    log.info("")
    if delta_f1 > 0.005:
        log.info(f"  >>> HYPOTHESIS CONFIRMED: Cleaning improved F1-macro by {delta_f1:+.4f}")
        log.info(f"      despite using {total_removed} fewer samples ({pct_removed:.1f}% removed)")
    elif delta_f1 > -0.005:
        log.info(f"  >>> INCONCLUSIVE: F1-macro delta = {delta_f1:+.4f} (within noise)")
    else:
        log.info(f"  >>> HYPOTHESIS REJECTED: Cleaning hurt F1-macro by {delta_f1:+.4f}")

    log.info(f"\n  Training time: {elapsed / 60:.1f} minutes")
    log.info(f"  Cleaned model saved to: {OUTPUT_DIR / 'final'}")

    # --- Save experiment metadata ---
    experiment_meta = {
        "experiment": "v5_clean_data",
        "hypothesis": (
            "Removing noisy, mislabeled, and near-duplicate samples from "
            "training data improves model quality more than adding more data."
        ),
        "base_model": BASE_MODEL,
        "config": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "max_seq_length": MAX_SEQ_LENGTH,
            "fp16": device == "cuda",
            "seed": SEED,
        },
        "cleaning": {
            "min_text_len": MIN_TEXT_LEN,
            "max_text_len": MAX_TEXT_LEN,
            "non_cyrillic_threshold": NON_CYRILLIC_THRESHOLD,
            "mislabel_confidence_threshold": MISLABEL_CONFIDENCE,
            "probe_epochs": PROBE_EPOCHS,
            "original_train_size": len(raw_train),
            "cleaned_train_size": len(cleaned),
            "samples_removed": total_removed,
            "pct_removed": round(pct_removed, 2),
        },
        "raw_results": {k: v for k, v in raw_results.items() if k != "report"},
        "clean_results": {k: v for k, v in clean_results.items() if k != "report"},
        "comparison": {
            "f1_macro_delta": round(delta_f1, 4),
            "hypothesis_confirmed": delta_f1 > 0.005,
        },
        "raw_report": raw_results["report"],
        "clean_report": clean_results["report"],
        "training_time_minutes": round(elapsed / 60, 1),
    }

    metrics_path = OUTPUT_DIR / "final" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(experiment_meta, f, indent=2, ensure_ascii=False)
    log.info(f"  Metrics saved to: {metrics_path}")

    log.info("=" * 60)
    log.info("DONE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
