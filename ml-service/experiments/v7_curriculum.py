"""
VARIANT 7: Curriculum Learning -- easy examples first, hard examples later.

Hypothesis:
    Training on clear-cut sentiment first, then gradually introducing
    ambiguous texts, builds a stronger foundation.

Approach:
    1. Score difficulty: train rubert-tiny2 for 1 epoch on all data,
       record per-sample cross-entropy loss. High loss = hard example.
    2. Curriculum schedule (5 epochs):
         Epoch 1: easiest 40%
         Epoch 2: easiest 60%
         Epoch 3: easiest 80%
         Epoch 4-5: all 100%
    3. Compare with standard training (same model, same hyperparams,
       no curriculum) as a controlled baseline.

Data:
    data/merged_train.jsonl   -- {"text": "...", "label": 0|1|2}
    data/merged_val.jsonl
    data/merged_test.jsonl

    Labels: 0=positive, 1=negative, 2=neutral

Run:
    cd ml-service && python3 experiments/v7_curriculum.py
"""

import copy
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
TRAIN_DATA = Path("data/merged_train.jsonl")
VAL_DATA = Path("data/merged_val.jsonl")
TEST_DATA = Path("data/merged_test.jsonl")
OUTPUT_DIR = Path("models/v7_curriculum")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "cointegrated/rubert-tiny2"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 64
LEARNING_RATE = 2e-5
SEED = 42

# Curriculum schedule: fraction of training data per epoch
CURRICULUM_SCHEDULE = {
    1: 0.40,  # easiest 40%
    2: 0.60,  # easiest 60%
    3: 0.80,  # easiest 80%
    4: 1.00,  # all data
    5: 1.00,  # all data
}
CURRICULUM_EPOCHS = 5
BASELINE_EPOCHS = 5

# Difficulty scoring: 1 epoch of training to measure per-sample loss
DIFFICULTY_SCORING_EPOCHS = 1

SENTIMENT_LABELS = ["positive", "negative", "neutral"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SentimentDataset(torch.utils.data.Dataset):
    """Stores pre-tokenized encodings + labels."""

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


class IndexedSentimentDataset(torch.utils.data.Dataset):
    """Same as SentimentDataset but also returns the sample index.

    Used during difficulty scoring so we can map each loss back to
    its original position in the training set.
    """

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
            "index": idx,
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
    return {
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "f1_positive": float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
        "f1_negative": float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
        "f1_neutral": float(f1_per_class[2]) if len(f1_per_class) > 2 else 0.0,
    }


# ---------------------------------------------------------------------------
# Weighted Trainer (class-balanced CE)
# ---------------------------------------------------------------------------
class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy loss."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights  # tensor on device

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = F.cross_entropy(outputs.logits, labels, weight=self.class_weights)
        inputs["labels"] = labels
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_jsonl(path: Path):
    """Load JSONL with format: {"text": "...", "label": 0|1|2}"""
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
# Phase 0: Difficulty scoring
# ---------------------------------------------------------------------------
def compute_difficulty_scores(
    train_encodings,
    train_labels,
    val_dataset,
    device,
    class_weights_tensor,
):
    """Train for 1 epoch, then compute per-sample loss as difficulty score.

    Returns:
        difficulty: np.ndarray of shape (n_train,) with per-sample CE loss.
        sort_indices: np.ndarray, indices that sort training data easy -> hard.
    """
    log.info("=" * 60)
    log.info("PHASE 0: COMPUTING DIFFICULTY SCORES (1 epoch)")
    log.info("=" * 60)

    n_train = len(train_labels)

    # Fresh model for difficulty scoring
    scorer_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=3, ignore_mismatched_sizes=True,
    )

    scorer_dir = OUTPUT_DIR / "difficulty_scorer"
    scorer_dir.mkdir(parents=True, exist_ok=True)

    scorer_args = TrainingArguments(
        output_dir=str(scorer_dir),
        num_train_epochs=DIFFICULTY_SCORING_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=100,
        eval_strategy="no",
        save_strategy="no",
        fp16=(device == "cuda"),
        dataloader_num_workers=4,
        report_to="none",
        seed=SEED,
    )

    train_dataset = SentimentDataset(train_encodings, train_labels)

    scorer_trainer = WeightedTrainer(
        model=scorer_model,
        args=scorer_args,
        train_dataset=train_dataset,
        class_weights=class_weights_tensor,
    )

    log.info("Training difficulty scorer for 1 epoch...")
    scorer_trainer.train()

    # Now compute per-sample loss on all training data
    log.info("Computing per-sample losses on training set...")
    scorer_model.eval()
    scorer_model.to(device)

    difficulty = np.zeros(n_train, dtype=np.float32)

    # Use a DataLoader to iterate over all training samples in order
    eval_dataset = SentimentDataset(train_encodings, train_labels)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0,
    )

    sample_idx = 0
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = scorer_model(input_ids=input_ids, attention_mask=attention_mask)
            # Per-sample cross-entropy (no reduction)
            losses = F.cross_entropy(outputs.logits, labels, reduction="none")
            batch_losses = losses.cpu().numpy()

            batch_size = len(batch_losses)
            difficulty[sample_idx : sample_idx + batch_size] = batch_losses
            sample_idx += batch_size

    assert sample_idx == n_train, f"Expected {n_train} samples, got {sample_idx}"

    # Sort easy -> hard (ascending loss)
    sort_indices = np.argsort(difficulty)

    # Log difficulty statistics
    log.info(f"Difficulty scores computed for {n_train} samples")
    log.info(f"  Min loss (easiest):  {difficulty[sort_indices[0]]:.4f}")
    log.info(f"  Median loss:         {np.median(difficulty):.4f}")
    log.info(f"  Mean loss:           {np.mean(difficulty):.4f}")
    log.info(f"  Max loss (hardest):  {difficulty[sort_indices[-1]]:.4f}")
    log.info(f"  Std:                 {np.std(difficulty):.4f}")

    # Per-class difficulty
    labels_arr = np.array(train_labels)
    for label_id, label_name in enumerate(SENTIMENT_LABELS):
        mask = labels_arr == label_id
        if mask.sum() > 0:
            log.info(
                f"  {label_name:>10s}: mean_loss={difficulty[mask].mean():.4f}, "
                f"std={difficulty[mask].std():.4f}, n={mask.sum()}"
            )

    # Clean up scorer model to free memory
    del scorer_model, scorer_trainer
    if device == "cuda":
        torch.cuda.empty_cache()

    return difficulty, sort_indices


# ---------------------------------------------------------------------------
# Curriculum subset dataset
# ---------------------------------------------------------------------------
def make_curriculum_subset(train_encodings, train_labels, sort_indices, fraction):
    """Create a SentimentDataset containing the easiest `fraction` of samples.

    Args:
        train_encodings: tokenized full training set
        train_labels: list of labels for full training set
        sort_indices: indices sorted by difficulty (easy first)
        fraction: float in (0, 1], fraction of data to include

    Returns:
        SentimentDataset with the selected subset
    """
    n_total = len(train_labels)
    n_select = max(1, int(n_total * fraction))
    selected_indices = sort_indices[:n_select]

    # Sort selected indices for sequential memory access
    selected_indices = np.sort(selected_indices)

    subset_encodings = {
        "input_ids": train_encodings["input_ids"][selected_indices],
        "attention_mask": train_encodings["attention_mask"][selected_indices],
    }
    subset_labels = [train_labels[i] for i in selected_indices]

    return SentimentDataset(subset_encodings, subset_labels)


# ---------------------------------------------------------------------------
# Train one full run (curriculum or baseline)
# ---------------------------------------------------------------------------
def train_model(
    name,
    train_encodings,
    train_labels,
    val_dataset,
    test_dataset,
    test_labels,
    device,
    class_weights_tensor,
    sort_indices=None,
    use_curriculum=False,
):
    """Train a model with or without curriculum learning.

    When use_curriculum=True, trains epoch-by-epoch with expanding data subsets.
    When use_curriculum=False, trains normally on all data for BASELINE_EPOCHS.

    Returns:
        dict with metrics and the trained model.
    """
    log.info("=" * 60)
    log.info(f"TRAINING: {name}")
    log.info("=" * 60)

    t_start = time.time()
    n_train = len(train_labels)

    # Initialize a fresh model
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=3, ignore_mismatched_sizes=True,
    )
    model.config.id2label = {0: "positive", 1: "negative", 2: "neutral"}
    model.config.label2id = {"positive": 0, "negative": 1, "neutral": 2}

    if use_curriculum:
        assert sort_indices is not None, "sort_indices required for curriculum"
        # ── Curriculum training: manual epoch loop ──
        # We cannot use HF Trainer's multi-epoch loop because the dataset
        # changes each epoch. Instead we train for 1 epoch at a time,
        # swapping in the appropriate data subset.

        run_dir = OUTPUT_DIR / "curriculum_run"
        run_dir.mkdir(parents=True, exist_ok=True)

        best_f1 = -1.0
        best_model_state = None
        epoch_metrics = []

        for epoch in range(1, CURRICULUM_EPOCHS + 1):
            fraction = CURRICULUM_SCHEDULE[epoch]
            n_select = max(1, int(n_train * fraction))
            subset_dataset = make_curriculum_subset(
                train_encodings, train_labels, sort_indices, fraction,
            )

            log.info(f"\n--- Curriculum Epoch {epoch}/{CURRICULUM_EPOCHS}: "
                     f"{fraction*100:.0f}% data ({n_select}/{n_train} samples) ---")

            # Log class distribution of this epoch's subset
            subset_labels_list = subset_dataset.labels.tolist()
            dist = Counter(subset_labels_list)
            for lid, lname in enumerate(SENTIMENT_LABELS):
                cnt = dist.get(lid, 0)
                log.info(f"  {lname}: {cnt} ({100*cnt/len(subset_labels_list):.1f}%)")

            epoch_args = TrainingArguments(
                output_dir=str(run_dir / f"epoch_{epoch}"),
                num_train_epochs=1,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE * 2,
                learning_rate=LEARNING_RATE,
                weight_decay=0.01,
                warmup_ratio=0.1,
                eval_strategy="epoch",
                save_strategy="no",
                logging_steps=100,
                fp16=(device == "cuda"),
                dataloader_num_workers=4,
                report_to="none",
                seed=SEED,
            )

            trainer = WeightedTrainer(
                model=model,
                args=epoch_args,
                train_dataset=subset_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                class_weights=class_weights_tensor,
            )

            trainer.train()

            # Evaluate on validation set
            val_results = trainer.evaluate()
            val_f1 = val_results.get("eval_f1_macro", 0.0)
            log.info(f"  Epoch {epoch} val F1-macro: {val_f1:.4f}")

            epoch_metrics.append({
                "epoch": epoch,
                "fraction": fraction,
                "n_samples": n_select,
                "val_f1_macro": round(val_f1, 4),
                "val_f1_weighted": round(val_results.get("eval_f1_weighted", 0.0), 4),
                "val_f1_positive": round(val_results.get("eval_f1_positive", 0.0), 4),
                "val_f1_negative": round(val_results.get("eval_f1_negative", 0.0), 4),
                "val_f1_neutral": round(val_results.get("eval_f1_neutral", 0.0), 4),
            })

            # Track best model by val F1
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model_state = copy.deepcopy(model.state_dict())
                log.info(f"  New best model at epoch {epoch} (val F1={val_f1:.4f})")

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            log.info(f"Restored best model (val F1={best_f1:.4f})")

    else:
        # ── Standard (baseline) training: all data, all epochs ──
        run_dir = OUTPUT_DIR / "baseline_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        epoch_metrics = []

        full_train_dataset = SentimentDataset(train_encodings, train_labels)

        baseline_args = TrainingArguments(
            output_dir=str(run_dir / "checkpoints"),
            num_train_epochs=BASELINE_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE * 2,
            learning_rate=LEARNING_RATE,
            weight_decay=0.01,
            warmup_ratio=0.1,
            eval_strategy="epoch",
            save_strategy="epoch",
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
            args=baseline_args,
            train_dataset=full_train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            class_weights=class_weights_tensor,
        )

        log.info(f"Training baseline for {BASELINE_EPOCHS} epochs on all {n_train} samples...")
        trainer.train()

        # Collect per-epoch metrics from trainer log history
        for entry in trainer.state.log_history:
            if "eval_f1_macro" in entry:
                epoch_metrics.append({
                    "epoch": entry.get("epoch", 0),
                    "fraction": 1.0,
                    "n_samples": n_train,
                    "val_f1_macro": round(entry["eval_f1_macro"], 4),
                    "val_f1_weighted": round(entry.get("eval_f1_weighted", 0.0), 4),
                    "val_f1_positive": round(entry.get("eval_f1_positive", 0.0), 4),
                    "val_f1_negative": round(entry.get("eval_f1_negative", 0.0), 4),
                    "val_f1_neutral": round(entry.get("eval_f1_neutral", 0.0), 4),
                })

    # ── Evaluate on test set ──
    log.info(f"\nEvaluating {name} on test set...")
    model.eval()
    model.to(device)

    # Use Trainer for prediction to handle batching
    eval_args = TrainingArguments(
        output_dir=str(run_dir / "eval"),
        per_device_eval_batch_size=BATCH_SIZE * 2,
        fp16=(device == "cuda"),
        report_to="none",
        dataloader_num_workers=4,
    )

    eval_trainer = Trainer(
        model=model,
        args=eval_args,
    )

    predictions = eval_trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)

    report_str = classification_report(
        test_labels, preds, target_names=SENTIMENT_LABELS, digits=4,
    )
    report_dict = classification_report(
        test_labels, preds, target_names=SENTIMENT_LABELS, output_dict=True,
    )
    log.info(f"\n{name} -- Test Results:\n{report_str}")

    elapsed = time.time() - t_start

    results = {
        "name": name,
        "f1_macro": round(report_dict["macro avg"]["f1-score"], 4),
        "f1_weighted": round(report_dict["weighted avg"]["f1-score"], 4),
        "f1_positive": round(report_dict["positive"]["f1-score"], 4),
        "f1_negative": round(report_dict["negative"]["f1-score"], 4),
        "f1_neutral": round(report_dict["neutral"]["f1-score"], 4),
        "accuracy": round(report_dict["accuracy"], 4),
        "epoch_metrics": epoch_metrics,
        "training_time_s": round(elapsed, 1),
        "report": report_dict,
    }

    return results, model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()

    # ── Device ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    if device == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Load data ──
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
    log.info(f"Train distribution: {dict(sorted(train_dist.items()))}")
    log.info(f"  " + ", ".join(
        f"{SENTIMENT_LABELS[k]}: {v} ({100*v/len(train_labels):.1f}%)"
        for k, v in sorted(train_dist.items())
    ))

    # ── Class weights ──
    class_weights = compute_class_weight(
        "balanced",
        classes=np.array([0, 1, 2]),
        y=np.array(train_labels),
    )
    log.info(f"Class weights: {dict(zip(SENTIMENT_LABELS, class_weights))}")
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # ── Tokenize ──
    log.info(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    log.info("Tokenizing train set...")
    train_enc = tokenizer(
        train_texts, truncation=True, padding="max_length",
        max_length=MAX_SEQ_LENGTH, return_tensors="pt",
    )
    log.info("Tokenizing val set...")
    val_enc = tokenizer(
        val_texts, truncation=True, padding="max_length",
        max_length=MAX_SEQ_LENGTH, return_tensors="pt",
    )
    log.info("Tokenizing test set...")
    test_enc = tokenizer(
        test_texts, truncation=True, padding="max_length",
        max_length=MAX_SEQ_LENGTH, return_tensors="pt",
    )

    val_dataset = SentimentDataset(val_enc, val_labels)
    test_dataset = SentimentDataset(test_enc, test_labels)

    # ==================================================================
    # PHASE 0: Compute per-sample difficulty scores
    # ==================================================================
    difficulty, sort_indices = compute_difficulty_scores(
        train_enc, train_labels, val_dataset, device, weights_tensor,
    )

    # Save difficulty scores for analysis
    difficulty_path = OUTPUT_DIR / "difficulty_scores.json"
    log.info(f"Saving difficulty scores to {difficulty_path}")
    difficulty_data = []
    for i in range(len(train_labels)):
        difficulty_data.append({
            "index": i,
            "text": train_texts[i][:200],  # truncate for readability
            "label": train_labels[i],
            "label_name": SENTIMENT_LABELS[train_labels[i]],
            "difficulty_loss": round(float(difficulty[i]), 6),
        })
    # Sort by difficulty for readability
    difficulty_data.sort(key=lambda x: x["difficulty_loss"])
    with open(difficulty_path, "w", encoding="utf-8") as f:
        json.dump(difficulty_data, f, indent=2, ensure_ascii=False)

    # Show easiest and hardest examples
    log.info("\nTop 5 EASIEST examples:")
    for i in range(min(5, len(sort_indices))):
        idx = sort_indices[i]
        log.info(f"  [{difficulty[idx]:.4f}] ({SENTIMENT_LABELS[train_labels[idx]]}) "
                 f"{train_texts[idx][:100]}...")

    log.info("\nTop 5 HARDEST examples:")
    for i in range(min(5, len(sort_indices))):
        idx = sort_indices[-(i + 1)]
        log.info(f"  [{difficulty[idx]:.4f}] ({SENTIMENT_LABELS[train_labels[idx]]}) "
                 f"{train_texts[idx][:100]}...")

    # ==================================================================
    # EXPERIMENT A: Curriculum learning
    # ==================================================================
    curriculum_results, curriculum_model = train_model(
        name="Curriculum Learning (easy->hard)",
        train_encodings=train_enc,
        train_labels=train_labels,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        test_labels=test_labels,
        device=device,
        class_weights_tensor=weights_tensor,
        sort_indices=sort_indices,
        use_curriculum=True,
    )

    # Free memory before baseline training
    del curriculum_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ==================================================================
    # EXPERIMENT B: Standard training (no curriculum, baseline)
    # ==================================================================
    baseline_results, baseline_model = train_model(
        name="Standard Training (baseline)",
        train_encodings=train_enc,
        train_labels=train_labels,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        test_labels=test_labels,
        device=device,
        class_weights_tensor=weights_tensor,
        sort_indices=None,
        use_curriculum=False,
    )

    # ==================================================================
    # Save the best model
    # ==================================================================
    curriculum_f1 = curriculum_results["f1_macro"]
    baseline_f1 = baseline_results["f1_macro"]

    if curriculum_f1 >= baseline_f1:
        winner = "curriculum"
        winner_f1 = curriculum_f1
        log.info("\nCurriculum model wins -- retraining to save...")
        # Retrain curriculum model to save (since we deleted it to free memory)
        # Instead, we re-run just the curriculum training and save at the end
        save_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=3, ignore_mismatched_sizes=True,
        )
        save_model.config.id2label = {0: "positive", 1: "negative", 2: "neutral"}
        save_model.config.label2id = {"positive": 0, "negative": 1, "neutral": 2}

        save_dir = OUTPUT_DIR / "curriculum_run"
        best_f1_retrain = -1.0

        for epoch in range(1, CURRICULUM_EPOCHS + 1):
            fraction = CURRICULUM_SCHEDULE[epoch]
            subset_dataset = make_curriculum_subset(
                train_enc, train_labels, sort_indices, fraction,
            )
            epoch_args = TrainingArguments(
                output_dir=str(save_dir / f"retrain_epoch_{epoch}"),
                num_train_epochs=1,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE * 2,
                learning_rate=LEARNING_RATE,
                weight_decay=0.01,
                warmup_ratio=0.1,
                eval_strategy="epoch",
                save_strategy="no",
                logging_steps=100,
                fp16=(device == "cuda"),
                dataloader_num_workers=4,
                report_to="none",
                seed=SEED,
            )
            retrain_trainer = WeightedTrainer(
                model=save_model,
                args=epoch_args,
                train_dataset=subset_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                class_weights=weights_tensor,
            )
            retrain_trainer.train()

            val_res = retrain_trainer.evaluate()
            val_f1 = val_res.get("eval_f1_macro", 0.0)
            if val_f1 > best_f1_retrain:
                best_f1_retrain = val_f1
                best_state = copy.deepcopy(save_model.state_dict())

        save_model.load_state_dict(best_state)
    else:
        winner = "baseline"
        winner_f1 = baseline_f1
        save_model = baseline_model

    # Save best model
    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"\nSaving best model ({winner}) to {final_dir}")
    save_model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # ==================================================================
    # Comparison report
    # ==================================================================
    log.info("\n" + "=" * 70)
    log.info("EXPERIMENT RESULTS: CURRICULUM LEARNING vs STANDARD TRAINING")
    log.info("=" * 70)

    header = f"  {'Metric':<20} {'Curriculum':>12} {'Baseline':>12} {'Delta':>10}"
    separator = f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}"
    log.info(header)
    log.info(separator)

    for metric_key, metric_name in [
        ("f1_macro", "F1-macro"),
        ("f1_weighted", "F1-weighted"),
        ("f1_positive", "F1-positive"),
        ("f1_negative", "F1-negative"),
        ("f1_neutral", "F1-neutral"),
        ("accuracy", "Accuracy"),
    ]:
        c_val = curriculum_results[metric_key]
        b_val = baseline_results[metric_key]
        delta = c_val - b_val
        log.info(f"  {metric_name:<20} {c_val:>12.4f} {b_val:>12.4f} {delta:>+10.4f}")

    log.info(separator)
    log.info(f"  {'Training time':<20} "
             f"{curriculum_results['training_time_s']:>10.0f}s "
             f"{baseline_results['training_time_s']:>10.0f}s")
    log.info("")

    if curriculum_f1 > baseline_f1:
        log.info(f"  >>> CURRICULUM WINS by {curriculum_f1 - baseline_f1:+.4f} F1-macro <<<")
    elif baseline_f1 > curriculum_f1:
        log.info(f"  >>> BASELINE WINS by {baseline_f1 - curriculum_f1:+.4f} F1-macro <<<")
    else:
        log.info("  >>> TIE -- both methods equal <<<")

    log.info("")
    log.info("  Curriculum schedule:")
    for ep, frac in CURRICULUM_SCHEDULE.items():
        n = max(1, int(len(train_labels) * frac))
        log.info(f"    Epoch {ep}: {frac*100:5.0f}% ({n:>6d} samples)")

    # Per-epoch progression for curriculum
    log.info("\n  Curriculum epoch-by-epoch val F1-macro:")
    for em in curriculum_results["epoch_metrics"]:
        log.info(f"    Epoch {em['epoch']}: {em['fraction']*100:5.0f}% data -> "
                 f"F1={em['val_f1_macro']:.4f}")

    log.info("\n  Baseline epoch-by-epoch val F1-macro:")
    for em in baseline_results["epoch_metrics"]:
        log.info(f"    Epoch {em['epoch']}: F1={em['val_f1_macro']:.4f}")

    # ==================================================================
    # Save metrics
    # ==================================================================
    metrics = {
        "experiment": "v7_curriculum_learning",
        "hypothesis": (
            "Training on clear-cut sentiment first, then gradually introducing "
            "ambiguous texts, builds a stronger foundation."
        ),
        "base_model": BASE_MODEL,
        "config": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_seq_length": MAX_SEQ_LENGTH,
            "curriculum_epochs": CURRICULUM_EPOCHS,
            "baseline_epochs": BASELINE_EPOCHS,
            "difficulty_scoring_epochs": DIFFICULTY_SCORING_EPOCHS,
            "curriculum_schedule": {str(k): v for k, v in CURRICULUM_SCHEDULE.items()},
            "fp16": device == "cuda",
            "seed": SEED,
        },
        "data": {
            "train_size": len(train_texts),
            "val_size": len(val_texts),
            "test_size": len(test_texts),
            "train_distribution": {SENTIMENT_LABELS[k]: v for k, v in sorted(train_dist.items())},
        },
        "difficulty_stats": {
            "min_loss": round(float(difficulty.min()), 4),
            "max_loss": round(float(difficulty.max()), 4),
            "mean_loss": round(float(difficulty.mean()), 4),
            "median_loss": round(float(np.median(difficulty)), 4),
            "std_loss": round(float(difficulty.std()), 4),
        },
        "curriculum_results": curriculum_results,
        "baseline_results": baseline_results,
        "winner": winner,
        "winner_f1_macro": winner_f1,
        "delta_f1_macro": round(curriculum_f1 - baseline_f1, 4),
    }

    # Remove nested report dicts for cleaner JSON (they're verbose)
    for key in ["curriculum_results", "baseline_results"]:
        if "report" in metrics[key]:
            del metrics[key]["report"]

    metrics_path = final_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    log.info(f"\nMetrics saved to {metrics_path}")

    # Also save full classification reports separately
    reports_path = OUTPUT_DIR / "classification_reports.json"
    with open(reports_path, "w", encoding="utf-8") as f:
        json.dump({
            "curriculum": curriculum_results.get("report", {}),
            "baseline": baseline_results.get("report", {}),
        }, f, indent=2, ensure_ascii=False)
    log.info(f"Full classification reports saved to {reports_path}")

    elapsed = time.time() - t_start
    log.info(f"\nTotal experiment time: {elapsed / 60:.1f} minutes")
    log.info(f"Best model ({winner}) saved to: {final_dir}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
