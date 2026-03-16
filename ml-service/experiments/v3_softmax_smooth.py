"""
Experiment V3: ruRoBERTa-large with SOFTMAX output + label smoothing.

Hypothesis:
    Our current Phase 1 model uses sigmoid (multi-label style) output which
    produces nearly-tied scores across classes.  Switching to softmax forces
    a proper probability distribution where classes compete.  Label smoothing
    (0.1) prevents overconfidence by spreading a small amount of probability
    mass to non-target classes, which acts as a regularizer.

Key changes vs Phase 1 (train_phase1.py):
    - problem_type="single_label_classification" -> forces softmax CE, NOT sigmoid
    - label_smoothing_factor=0.1 in TrainingArguments (native HF support)
    - Balanced class weights in custom loss
    - max_seq_len=256 (vs 128) -- captures more review context
    - epochs=3 (vs 5) -- label smoothing converges faster, less overfitting
    - No Focal/FGM/R-Drop/LLRD -- isolates the softmax+smoothing hypothesis

Baseline (Phase 1, commit 5903044):
    F1-macro = 0.868

Run:
    cd ml-service && python3 experiments/v3_softmax_smooth.py
"""

import json
import logging
import sys
import time
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
BASE_DIR = Path(__file__).resolve().parent.parent  # ml-service/
TRAIN_DATA = BASE_DIR / "data" / "merged_train.jsonl"
VAL_DATA = BASE_DIR / "data" / "merged_val.jsonl"
TEST_DATA = BASE_DIR / "data" / "merged_test.jsonl"
OUTPUT_DIR = BASE_DIR / "models" / "v3_softmax_smooth"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "ai-forever/ruRoBERTa-large"
MAX_SEQ_LENGTH = 256        # longer than Phase 1's 128 -- captures more context
BATCH_SIZE = 16             # fits on L4 24GB with fp16 + seq_len 256
GRADIENT_ACCUMULATION_STEPS = 4   # effective batch = 16 * 4 = 64
EPOCHS = 3                  # label smoothing converges faster
LEARNING_RATE = 2e-5
LABEL_SMOOTHING = 0.1       # prevents overconfidence
SEED = 42
EVAL_STEPS = 500
EARLY_STOPPING_PATIENCE = 3
WARMUP_RATIO = 0.1

SENTIMENT_LABELS = ["positive", "negative", "neutral"]
LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Phase 1 baseline (from commit 5903044)
BASELINE_F1_MACRO = 0.868
BASELINE_NAME = "Phase 1 (UltraTrainer)"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SentimentDataset(torch.utils.data.Dataset):
    """Stores pre-tokenized encodings + integer labels."""

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
# Weighted Softmax Trainer
# ---------------------------------------------------------------------------
class WeightedSoftmaxTrainer(Trainer):
    """
    Custom Trainer that applies balanced class weights to the cross-entropy
    loss.  The model itself uses problem_type="single_label_classification"
    which means it outputs softmax logits.  HF's built-in label smoothing
    is applied via TrainingArguments.label_smoothing_factor, but that uses
    unweighted CE internally.  We override compute_loss to combine both
    class weighting AND label smoothing.
    """

    def __init__(self, *args, class_weights=None, label_smoothing=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights  # tensor on device
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # (batch, num_classes)

        # Cross-entropy with class weights + label smoothing
        loss = F.cross_entropy(
            logits,
            labels,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

        inputs["labels"] = labels  # restore for other callbacks
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
    return {
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "f1_positive": float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
        "f1_negative": float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
        "f1_neutral": float(f1_per_class[2]) if len(f1_per_class) > 2 else 0.0,
    }


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

    # -- Load Data ---------------------------------------------------------
    log.info("=" * 60)
    log.info("LOADING DATA")
    log.info("=" * 60)

    for path in [TRAIN_DATA, VAL_DATA, TEST_DATA]:
        if not path.exists():
            log.error(
                f"Data file not found: {path}\n"
                f"Run first: cd ml-service && python3 download_datasets.py"
            )
            sys.exit(1)

    train_texts, train_labels = load_jsonl(TRAIN_DATA)
    val_texts, val_labels = load_jsonl(VAL_DATA)
    test_texts, test_labels = load_jsonl(TEST_DATA)

    log.info(f"Train: {len(train_texts):,} samples")
    log.info(f"Val:   {len(val_texts):,} samples")
    log.info(f"Test:  {len(test_texts):,} samples")

    # Label distribution
    train_dist = Counter(train_labels)
    log.info(f"Train label distribution: {  {SENTIMENT_LABELS[k]: v for k, v in sorted(train_dist.items())}  }")

    # -- Class Weights -----------------------------------------------------
    class_weights = compute_class_weight(
        "balanced",
        classes=np.array([0, 1, 2]),
        y=np.array(train_labels),
    )
    log.info(f"Class weights: {dict(zip(SENTIMENT_LABELS, class_weights))}")
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # -- Tokenize ----------------------------------------------------------
    log.info(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    log.info(f"Tokenizing (max_length={MAX_SEQ_LENGTH})...")
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

    # -- Model -------------------------------------------------------------
    log.info(f"Loading model: {BASE_MODEL}")
    log.info("  problem_type = single_label_classification (SOFTMAX, not sigmoid)")

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        problem_type="single_label_classification",  # FORCES SOFTMAX CE
        ignore_mismatched_sizes=True,
    )
    model.config.id2label = ID2LABEL
    model.config.label2id = LABEL2ID

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Total params: {total_params:,}")
    log.info(f"Trainable params: {trainable_params:,}")

    # -- Training Args -----------------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=WARMUP_RATIO,
        # Label smoothing is handled in our custom loss (combined with class weights).
        # We set it to 0 here so HF's internal loss does not double-smooth.
        label_smoothing_factor=0.0,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=EVAL_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        fp16=(device == "cuda"),
        dataloader_num_workers=4,
        report_to="none",
        seed=SEED,
    )

    # -- Trainer -----------------------------------------------------------
    trainer = WeightedSoftmaxTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
        class_weights=weights_tensor,
        label_smoothing=LABEL_SMOOTHING,
    )

    # -- Print config summary ----------------------------------------------
    log.info("=" * 60)
    log.info("V3 EXPERIMENT CONFIG: SOFTMAX + LABEL SMOOTHING")
    log.info("=" * 60)
    log.info(f"  Model:              {BASE_MODEL}")
    log.info(f"  problem_type:       single_label_classification (SOFTMAX)")
    log.info(f"  Label smoothing:    {LABEL_SMOOTHING}")
    log.info(f"  Class weights:      YES (balanced)")
    log.info(f"  Batch size:         {BATCH_SIZE} x {GRADIENT_ACCUMULATION_STEPS} = {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS} effective")
    log.info(f"  Epochs:             {EPOCHS}")
    log.info(f"  Learning rate:      {LEARNING_RATE}")
    log.info(f"  Max seq length:     {MAX_SEQ_LENGTH}")
    log.info(f"  FP16:               {device == 'cuda'}")
    log.info(f"  Eval steps:         {EVAL_STEPS}")
    log.info(f"  Early stopping:     patience={EARLY_STOPPING_PATIENCE}")
    log.info(f"  Warmup ratio:       {WARMUP_RATIO}")
    log.info(f"  Baseline:           {BASELINE_NAME} F1={BASELINE_F1_MACRO}")

    # -- Train -------------------------------------------------------------
    log.info("=" * 60)
    log.info("STARTING V3 TRAINING")
    log.info("  Hypothesis: softmax + label smoothing > sigmoid (Phase 1)")
    log.info("=" * 60)
    trainer.train()

    # -- Evaluate on test set ----------------------------------------------
    log.info("=" * 60)
    log.info("EVALUATING ON TEST SET")
    log.info("=" * 60)

    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)

    report_str = classification_report(
        test_labels, preds, target_names=SENTIMENT_LABELS, digits=4,
    )
    report_dict = classification_report(
        test_labels, preds, target_names=SENTIMENT_LABELS, output_dict=True,
    )
    log.info(f"\n{report_str}")

    f1_macro = report_dict["macro avg"]["f1-score"]
    f1_weighted = report_dict["weighted avg"]["f1-score"]
    f1_pos = report_dict["positive"]["f1-score"]
    f1_neg = report_dict["negative"]["f1-score"]
    f1_neu = report_dict["neutral"]["f1-score"]

    # -- Score distribution analysis ---------------------------------------
    # Show how softmax probabilities are distributed (the whole point of V3)
    log.info("=" * 60)
    log.info("SOFTMAX SCORE DISTRIBUTION ANALYSIS")
    log.info("=" * 60)

    probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()
    top_probs = np.max(probs, axis=-1)
    margin = np.sort(probs, axis=-1)[:, -1] - np.sort(probs, axis=-1)[:, -2]

    log.info(f"  Top-class probability:  mean={np.mean(top_probs):.4f}  std={np.std(top_probs):.4f}")
    log.info(f"  Top-class probability:  min={np.min(top_probs):.4f}  max={np.max(top_probs):.4f}")
    log.info(f"  Confidence margin (p1-p2): mean={np.mean(margin):.4f}  std={np.std(margin):.4f}")
    log.info(f"  Low-confidence (<0.5):  {np.sum(top_probs < 0.5)} / {len(top_probs)} samples")
    log.info(f"  High-confidence (>0.9): {np.sum(top_probs > 0.9)} / {len(top_probs)} samples")

    # -- Save model --------------------------------------------------------
    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Saving model to {final_dir}")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # -- Save metrics ------------------------------------------------------
    metrics = {
        "experiment": "v3_softmax_smooth",
        "hypothesis": "softmax + label smoothing produces better-separated probabilities than sigmoid",
        "base_model": BASE_MODEL,
        "config": {
            "problem_type": "single_label_classification",
            "label_smoothing": LABEL_SMOOTHING,
            "class_weights": "balanced",
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "max_seq_length": MAX_SEQ_LENGTH,
            "warmup_ratio": WARMUP_RATIO,
            "fp16": device == "cuda",
            "eval_steps": EVAL_STEPS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        },
        "data": {
            "train_size": len(train_texts),
            "val_size": len(val_texts),
            "test_size": len(test_texts),
        },
        "results": {
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "f1_positive": round(f1_pos, 4),
            "f1_negative": round(f1_neg, 4),
            "f1_neutral": round(f1_neu, 4),
        },
        "score_distribution": {
            "top_prob_mean": round(float(np.mean(top_probs)), 4),
            "top_prob_std": round(float(np.std(top_probs)), 4),
            "margin_mean": round(float(np.mean(margin)), 4),
            "margin_std": round(float(np.std(margin)), 4),
            "low_confidence_count": int(np.sum(top_probs < 0.5)),
            "high_confidence_count": int(np.sum(top_probs > 0.9)),
        },
        "baseline": {
            "name": BASELINE_NAME,
            "f1_macro": BASELINE_F1_MACRO,
        },
        "improvement_over_baseline": round(f1_macro - BASELINE_F1_MACRO, 4),
        "report": report_dict,
    }

    with open(final_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # -- Comparison table --------------------------------------------------
    elapsed = time.time() - t_start
    log.info("=" * 60)
    log.info("V3 RESULTS vs BASELINE")
    log.info("=" * 60)
    log.info("")
    log.info(f"  {'Metric':<20} {BASELINE_NAME:>16} {'V3 Softmax+LS':>16} {'Delta':>10}")
    log.info(f"  {'-'*20} {'-'*16} {'-'*16} {'-'*10}")
    log.info(f"  {'F1-macro':<20} {BASELINE_F1_MACRO:>16.4f} {f1_macro:>16.4f} {f1_macro - BASELINE_F1_MACRO:>+10.4f}")
    log.info(f"  {'F1-weighted':<20} {'':>16} {f1_weighted:>16.4f} {'':>10}")
    log.info(f"  {'F1-positive':<20} {'':>16} {f1_pos:>16.4f} {'':>10}")
    log.info(f"  {'F1-negative':<20} {'':>16} {f1_neg:>16.4f} {'':>10}")
    log.info(f"  {'F1-neutral':<20} {'':>16} {f1_neu:>16.4f} {'':>10}")
    log.info("")

    log.info("  Key differences from Phase 1:")
    log.info("    - SOFTMAX output (single_label_classification) vs sigmoid")
    log.info(f"    - Label smoothing = {LABEL_SMOOTHING}")
    log.info(f"    - Max seq length = {MAX_SEQ_LENGTH} (vs 128)")
    log.info(f"    - Epochs = {EPOCHS} (vs 5)")
    log.info("    - No Focal/FGM/R-Drop/LLRD (isolating softmax+smoothing effect)")
    log.info("")

    log.info("  Score separation (softmax confidence):")
    log.info(f"    - Mean top-class prob: {np.mean(top_probs):.4f}")
    log.info(f"    - Mean margin (p1-p2): {np.mean(margin):.4f}")
    log.info(f"    - Low-confidence samples: {np.sum(top_probs < 0.5)}")
    log.info("")

    if f1_macro > BASELINE_F1_MACRO:
        log.info(f"  >>> IMPROVED over baseline by {f1_macro - BASELINE_F1_MACRO:+.4f} <<<")
    elif abs(f1_macro - BASELINE_F1_MACRO) < 0.005:
        log.info(f"  >>> COMPARABLE to baseline (delta < 0.005) <<<")
    else:
        log.info(f"  >>> BELOW baseline by {f1_macro - BASELINE_F1_MACRO:+.4f} <<<")

    log.info("")
    log.info(f"  Training time: {elapsed / 60:.1f} minutes")
    log.info(f"  Model saved to: {final_dir}")
    log.info(f"  Metrics saved to: {final_dir / 'metrics.json'}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
