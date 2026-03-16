"""
Variant 6: Knowledge Distillation from ruRoBERTa-large -> rubert-tiny2

Hypothesis: The large teacher model (F1 0.868) captured knowledge the tiny
student model can learn from. Soft labels (softmax probabilities) from the
teacher are more informative than hard labels because they encode inter-class
similarity -- e.g. "slightly negative" vs "strongly negative" becomes visible
in the probability distribution over classes.

Distillation loss:
    L = alpha * T^2 * KL_div(student_logits/T, teacher_probs/T)
      + (1 - alpha) * CE(student_logits, hard_labels)

Where:
    T = 4.0   (temperature -- higher = softer distributions)
    alpha = 0.7  (more weight on soft labels from teacher)

The T^2 scaling compensates for the 1/T^2 factor in the KL gradient that
appears when using temperature scaling (Hinton et al., 2015).

Comparison targets:
    - Teacher:     ruRoBERTa-large (Phase 1, F1-macro 0.868)
    - Baseline:    rubert-tiny2 trained with hard labels only (F1-macro ~0.758)
    - Student v6:  rubert-tiny2 trained with distillation

Run:
    cd ml-service && python3 experiments/v6_distillation.py
"""

import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
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
# Paths (relative to ml-service/ working directory)
TRAIN_DATA = Path("data/merged_train.jsonl")
VAL_DATA = Path("data/merged_val.jsonl")
TEST_DATA = Path("data/merged_test.jsonl")
TEACHER_MODEL_DIR = Path("models/phase1-ruroberta/final")
OUTPUT_DIR = Path("models/v6_distillation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Models
TEACHER_MODEL_NAME = "ai-forever/ruRoBERTa-large"  # for logging only
STUDENT_BASE_MODEL = "cointegrated/rubert-tiny2"
NUM_LABELS = 3

# Distillation hyperparameters
TEMPERATURE = 4.0       # softens probability distributions
ALPHA = 0.7             # weight for soft-label KL loss (1-alpha for hard CE)

# Training hyperparameters
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EVAL_STEPS = 500
EARLY_STOPPING_PATIENCE = 3
SEED = 42

# Teacher inference
TEACHER_BATCH_SIZE = 128  # teacher is in eval mode, can use larger batches

SENTIMENT_LABELS = ["positive", "negative", "neutral"]
TEACHER_F1 = 0.868       # Phase 1 ruRoBERTa-large result
BASELINE_TINY_F1 = 0.758  # rubert-tiny2 with hard labels (v1)


# ---------------------------------------------------------------------------
# Dataset with soft labels
# ---------------------------------------------------------------------------
class DistillationDataset(Dataset):
    """
    Dataset for knowledge distillation.
    Each sample contains:
      - input_ids, attention_mask (tokenized text for the student)
      - labels (hard label, int 0/1/2)
      - teacher_probs (soft label, float tensor of shape [num_labels])
    """

    def __init__(self, encodings, labels, teacher_probs):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.teacher_probs = torch.tensor(teacher_probs, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
            "teacher_probs": self.teacher_probs[idx],
        }


class EvalDataset(Dataset):
    """Standard evaluation dataset (no teacher probs needed)."""

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
# Teacher inference: generate soft labels
# ---------------------------------------------------------------------------
class TeacherProber:
    """
    Generates soft-label probability distributions from the teacher model.

    Runs the teacher in eval mode (no gradient) over all training data
    and returns softmax probabilities for each sample.
    """

    def __init__(self, model_dir: Path, device: str, batch_size: int = 128):
        self.device = device
        self.batch_size = batch_size

        log.info(f"Loading teacher model from {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir),
            num_labels=NUM_LABELS,
        ).to(device)
        self.model.eval()

        total_params = sum(p.numel() for p in self.model.parameters())
        log.info(f"Teacher loaded: {total_params:,} parameters")

    @torch.no_grad()
    def generate_soft_labels(self, texts: list[str]) -> np.ndarray:
        """
        Generate soft-label probabilities for a list of texts.

        Returns:
            np.ndarray of shape (len(texts), NUM_LABELS) with softmax probs
        """
        log.info(f"Generating soft labels for {len(texts)} samples "
                 f"(batch_size={self.batch_size})...")

        all_probs = []
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1

            if batch_num % 50 == 0 or batch_num == num_batches:
                log.info(f"  Teacher inference: batch {batch_num}/{num_batches}")

            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=MAX_SEQ_LENGTH,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**encodings)
            # Plain softmax (no temperature here -- temperature is applied in the loss)
            probs = F.softmax(outputs.logits, dim=-1)
            all_probs.append(probs.cpu().numpy())

        soft_labels = np.concatenate(all_probs, axis=0)
        log.info(f"Soft labels generated: shape {soft_labels.shape}")

        # Sanity check: teacher's own accuracy on these samples
        return soft_labels

    def evaluate_teacher(self, texts: list[str], labels: list[int]) -> dict:
        """Evaluate teacher on a dataset to get its metrics."""
        soft_labels = self.generate_soft_labels(texts)
        preds = np.argmax(soft_labels, axis=-1)
        report = classification_report(
            labels, preds, target_names=SENTIMENT_LABELS, output_dict=True
        )
        return report


# ---------------------------------------------------------------------------
# Distillation Trainer
# ---------------------------------------------------------------------------
class DistillationTrainer(Trainer):
    """
    Custom Trainer implementing knowledge distillation loss.

    Loss = alpha * T^2 * KL_div(student_soft, teacher_soft)
         + (1 - alpha) * CE(student_logits, hard_labels)

    Where:
        student_soft = log_softmax(student_logits / T)
        teacher_soft = softmax(teacher_probs_logit_scale / T)
                     = softmax(log(teacher_probs) / T)

    The T^2 factor compensates for the gradient magnitude reduction
    caused by temperature scaling (see Hinton et al., 2015).
    """

    def __init__(self, *args, temperature=4.0, alpha=0.7,
                 class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.alpha = alpha
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        teacher_probs = inputs.pop("teacher_probs", None)
        outputs = model(**inputs)
        student_logits = outputs.logits

        # --- Hard-label cross-entropy loss ---
        ce_loss = F.cross_entropy(
            student_logits, labels, weight=self.class_weights
        )

        if teacher_probs is not None and self.alpha > 0:
            # --- Soft-label KL divergence loss ---
            T = self.temperature

            # Student: log-softmax of logits / T
            student_log_soft = F.log_softmax(student_logits / T, dim=-1)

            # Teacher: we have raw probabilities from the teacher.
            # To apply temperature, convert back to logit scale then re-softmax:
            #   teacher_soft = softmax(log(teacher_probs) / T)
            # Clamp to avoid log(0)
            teacher_logits_approx = torch.log(teacher_probs.clamp(min=1e-8))
            teacher_soft = F.softmax(teacher_logits_approx / T, dim=-1)

            # KL divergence: KL(teacher || student)
            # F.kl_div expects input=log_probs, target=probs
            kl_loss = F.kl_div(
                student_log_soft,
                teacher_soft,
                reduction="batchmean",
            )

            # Combined loss with T^2 scaling on KL term
            loss = (
                self.alpha * (T ** 2) * kl_loss
                + (1 - self.alpha) * ce_loss
            )
        else:
            # Fallback: pure CE (used during evaluation or if no teacher probs)
            loss = ce_loss

        # Restore inputs for potential reuse
        inputs["labels"] = labels
        if teacher_probs is not None:
            inputs["teacher_probs"] = teacher_probs

        return (loss, outputs) if return_outputs else loss


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
# Data loading
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> tuple[list[str], list[int]]:
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
# Baseline: train tiny model with hard labels only (for comparison)
# ---------------------------------------------------------------------------
def train_baseline_tiny(
    tokenizer,
    train_texts, train_labels,
    val_texts, val_labels,
    device: str,
    class_weights_tensor: torch.Tensor,
) -> tuple:
    """
    Train a baseline rubert-tiny2 with hard labels only (standard CE).
    Returns (model, trainer) for evaluation.
    """
    log.info("=" * 60)
    log.info("BASELINE: Training rubert-tiny2 with hard labels only")
    log.info("=" * 60)

    baseline_dir = OUTPUT_DIR / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    # Tokenize
    train_enc = tokenizer(
        train_texts, truncation=True, padding="max_length",
        max_length=MAX_SEQ_LENGTH, return_tensors="pt",
    )
    val_enc = tokenizer(
        val_texts, truncation=True, padding="max_length",
        max_length=MAX_SEQ_LENGTH, return_tensors="pt",
    )

    train_dataset = EvalDataset(train_enc, train_labels)
    val_dataset = EvalDataset(val_enc, val_labels)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_BASE_MODEL,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
    )
    model.config.id2label = {0: "positive", 1: "negative", 2: "neutral"}
    model.config.label2id = {"positive": 0, "negative": 1, "neutral": 2}

    # Weighted CE trainer
    _weights = class_weights_tensor

    class BaselineTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = F.cross_entropy(outputs.logits, labels, weight=_weights)
            inputs["labels"] = labels
            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=str(baseline_dir / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
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

    trainer = BaselineTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    trainer.train()
    return model, trainer


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

    # =====================================================================
    # STEP 1: Load Data
    # =====================================================================
    log.info("=" * 60)
    log.info("STEP 1: LOADING DATA")
    log.info("=" * 60)

    train_texts, train_labels = load_jsonl(TRAIN_DATA)
    val_texts, val_labels = load_jsonl(VAL_DATA)
    test_texts, test_labels = load_jsonl(TEST_DATA)

    log.info(f"Train: {len(train_texts)} samples")
    log.info(f"Val:   {len(val_texts)} samples")
    log.info(f"Test:  {len(test_texts)} samples")

    train_dist = Counter(train_labels)
    log.info(f"Train label distribution: "
             f"{ {SENTIMENT_LABELS[k]: v for k, v in sorted(train_dist.items())} }")

    # Class weights for hard-label CE component
    class_weights = compute_class_weight(
        "balanced",
        classes=np.array([0, 1, 2]),
        y=np.array(train_labels),
    )
    log.info(f"Class weights: {dict(zip(SENTIMENT_LABELS, class_weights))}")
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # =====================================================================
    # STEP 2: Generate soft labels from teacher
    # =====================================================================
    log.info("=" * 60)
    log.info("STEP 2: GENERATING SOFT LABELS FROM TEACHER")
    log.info("=" * 60)

    if not TEACHER_MODEL_DIR.exists():
        log.warning(
            f"Teacher model not found at {TEACHER_MODEL_DIR}. "
            f"Attempting to load from HuggingFace Hub: "
            f"Daniil125/prodradar-sentiment-ru"
        )
        teacher_path = "Daniil125/prodradar-sentiment-ru"
    else:
        teacher_path = TEACHER_MODEL_DIR

    teacher = TeacherProber(
        model_dir=Path(str(teacher_path)),
        device=device,
        batch_size=TEACHER_BATCH_SIZE,
    )

    # Generate soft labels for training data
    train_soft_labels = teacher.generate_soft_labels(train_texts)

    # Verify teacher quality on validation set
    log.info("Evaluating teacher on validation set...")
    teacher_val_report = teacher.evaluate_teacher(val_texts, val_labels)
    teacher_val_f1 = teacher_val_report["macro avg"]["f1-score"]
    log.info(f"Teacher val F1-macro: {teacher_val_f1:.4f}")

    # Evaluate teacher on test set (for final comparison)
    log.info("Evaluating teacher on test set...")
    teacher_test_report = teacher.evaluate_teacher(test_texts, test_labels)
    teacher_test_f1 = teacher_test_report["macro avg"]["f1-score"]
    log.info(f"Teacher test F1-macro: {teacher_test_f1:.4f}")

    # Analyze soft label distribution
    teacher_preds = np.argmax(train_soft_labels, axis=-1)
    teacher_train_acc = np.mean(teacher_preds == np.array(train_labels))
    avg_confidence = np.mean(np.max(train_soft_labels, axis=-1))
    avg_entropy = -np.mean(np.sum(
        train_soft_labels * np.log(train_soft_labels + 1e-8), axis=-1
    ))
    log.info(f"Teacher train accuracy: {teacher_train_acc:.4f}")
    log.info(f"Avg confidence (max prob): {avg_confidence:.4f}")
    log.info(f"Avg entropy: {avg_entropy:.4f} (max={np.log(NUM_LABELS):.4f})")

    # Show examples where teacher disagrees with hard labels
    disagreements = np.sum(teacher_preds != np.array(train_labels))
    log.info(f"Teacher-label disagreements: {disagreements}/{len(train_labels)} "
             f"({100 * disagreements / len(train_labels):.1f}%)")

    # Free teacher model to save memory before student training
    del teacher
    if device == "cuda":
        torch.cuda.empty_cache()
    log.info("Teacher model freed from memory")

    # =====================================================================
    # STEP 3: Load student tokenizer and tokenize data
    # =====================================================================
    log.info("=" * 60)
    log.info("STEP 3: TOKENIZING DATA FOR STUDENT")
    log.info("=" * 60)

    student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_BASE_MODEL)
    log.info(f"Student tokenizer: {STUDENT_BASE_MODEL}")

    log.info("Tokenizing train set...")
    train_enc = student_tokenizer(
        train_texts, truncation=True, padding="max_length",
        max_length=MAX_SEQ_LENGTH, return_tensors="pt",
    )
    log.info("Tokenizing val set...")
    val_enc = student_tokenizer(
        val_texts, truncation=True, padding="max_length",
        max_length=MAX_SEQ_LENGTH, return_tensors="pt",
    )
    log.info("Tokenizing test set...")
    test_enc = student_tokenizer(
        test_texts, truncation=True, padding="max_length",
        max_length=MAX_SEQ_LENGTH, return_tensors="pt",
    )

    # Build datasets
    train_dataset = DistillationDataset(train_enc, train_labels, train_soft_labels)
    val_dataset = EvalDataset(val_enc, val_labels)
    test_dataset = EvalDataset(test_enc, test_labels)

    # =====================================================================
    # STEP 4: Train student with distillation
    # =====================================================================
    log.info("=" * 60)
    log.info("STEP 4: TRAINING STUDENT WITH KNOWLEDGE DISTILLATION")
    log.info("=" * 60)

    student_model = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_BASE_MODEL,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
    )
    student_model.config.id2label = {0: "positive", 1: "negative", 2: "neutral"}
    student_model.config.label2id = {"positive": 0, "negative": 1, "neutral": 2}

    total_params = sum(p.numel() for p in student_model.parameters())
    trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    log.info(f"Student model: {STUDENT_BASE_MODEL}")
    log.info(f"Total params: {total_params:,}")
    log.info(f"Trainable params: {trainable_params:,}")

    # Distillation configuration summary
    log.info("")
    log.info("  DISTILLATION CONFIG:")
    log.info(f"    Teacher:         {TEACHER_MODEL_NAME} (F1={TEACHER_F1})")
    log.info(f"    Student:         {STUDENT_BASE_MODEL}")
    log.info(f"    Temperature:     {TEMPERATURE}")
    log.info(f"    Alpha:           {ALPHA} (soft={ALPHA}, hard={1-ALPHA})")
    log.info(f"    Batch size:      {BATCH_SIZE}")
    log.info(f"    Epochs:          {EPOCHS}")
    log.info(f"    Learning rate:   {LEARNING_RATE}")
    log.info(f"    Max seq length:  {MAX_SEQ_LENGTH}")
    log.info(f"    FP16:            {device == 'cuda'}")
    log.info(f"    Eval steps:      {EVAL_STEPS}")
    log.info(f"    Early stopping:  patience={EARLY_STOPPING_PATIENCE}")
    log.info("")

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
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

    trainer = DistillationTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
        temperature=TEMPERATURE,
        alpha=ALPHA,
        class_weights=weights_tensor,
    )

    log.info("Starting distillation training...")
    trainer.train()

    # =====================================================================
    # STEP 5: Evaluate distilled student
    # =====================================================================
    log.info("=" * 60)
    log.info("STEP 5: EVALUATING DISTILLED STUDENT ON TEST SET")
    log.info("=" * 60)

    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)

    student_report_str = classification_report(
        test_labels, preds, target_names=SENTIMENT_LABELS, digits=4,
    )
    student_report_dict = classification_report(
        test_labels, preds, target_names=SENTIMENT_LABELS, output_dict=True,
    )
    log.info(f"\nDistilled Student:\n{student_report_str}")

    student_f1_macro = student_report_dict["macro avg"]["f1-score"]
    student_f1_weighted = student_report_dict["weighted avg"]["f1-score"]

    # =====================================================================
    # STEP 6: Train baseline for fair comparison (optional -- skip if slow)
    # =====================================================================
    log.info("=" * 60)
    log.info("STEP 6: TRAINING BASELINE (hard labels only) FOR COMPARISON")
    log.info("=" * 60)

    baseline_model, baseline_trainer = train_baseline_tiny(
        tokenizer=student_tokenizer,
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        device=device,
        class_weights_tensor=weights_tensor,
    )

    baseline_preds_output = baseline_trainer.predict(test_dataset)
    baseline_preds = np.argmax(baseline_preds_output.predictions, axis=-1)

    baseline_report_str = classification_report(
        test_labels, baseline_preds, target_names=SENTIMENT_LABELS, digits=4,
    )
    baseline_report_dict = classification_report(
        test_labels, baseline_preds, target_names=SENTIMENT_LABELS, output_dict=True,
    )
    log.info(f"\nBaseline (hard labels only):\n{baseline_report_str}")

    baseline_f1_macro = baseline_report_dict["macro avg"]["f1-score"]
    baseline_f1_weighted = baseline_report_dict["weighted avg"]["f1-score"]

    # =====================================================================
    # STEP 7: Save distilled student model
    # =====================================================================
    log.info("=" * 60)
    log.info("STEP 7: SAVING DISTILLED STUDENT MODEL")
    log.info("=" * 60)

    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Saving to {final_dir}")
    trainer.save_model(str(final_dir))
    student_tokenizer.save_pretrained(str(final_dir))

    # =====================================================================
    # STEP 8: Final comparison
    # =====================================================================
    log.info("=" * 60)
    log.info("FINAL COMPARISON: Teacher vs Baseline vs Distilled Student")
    log.info("=" * 60)

    # Collect all F1 scores
    teacher_f1_pos = teacher_test_report["positive"]["f1-score"]
    teacher_f1_neg = teacher_test_report["negative"]["f1-score"]
    teacher_f1_neu = teacher_test_report["neutral"]["f1-score"]
    teacher_f1_w = teacher_test_report["weighted avg"]["f1-score"]

    baseline_f1_pos = baseline_report_dict["positive"]["f1-score"]
    baseline_f1_neg = baseline_report_dict["negative"]["f1-score"]
    baseline_f1_neu = baseline_report_dict["neutral"]["f1-score"]

    student_f1_pos = student_report_dict["positive"]["f1-score"]
    student_f1_neg = student_report_dict["negative"]["f1-score"]
    student_f1_neu = student_report_dict["neutral"]["f1-score"]

    header = f"  {'Metric':<16} {'Teacher':>10} {'Baseline':>10} {'Distilled':>10} {'D-B Delta':>10}"
    sep = f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10} {'-'*10}"

    log.info("")
    log.info(header)
    log.info(sep)
    log.info(f"  {'F1-macro':<16} {teacher_test_f1:>10.4f} {baseline_f1_macro:>10.4f} "
             f"{student_f1_macro:>10.4f} {student_f1_macro - baseline_f1_macro:>+10.4f}")
    log.info(f"  {'F1-weighted':<16} {teacher_f1_w:>10.4f} {baseline_f1_weighted:>10.4f} "
             f"{student_f1_weighted:>10.4f} {student_f1_weighted - baseline_f1_weighted:>+10.4f}")
    log.info(f"  {'F1-positive':<16} {teacher_f1_pos:>10.4f} {baseline_f1_pos:>10.4f} "
             f"{student_f1_pos:>10.4f} {student_f1_pos - baseline_f1_pos:>+10.4f}")
    log.info(f"  {'F1-negative':<16} {teacher_f1_neg:>10.4f} {baseline_f1_neg:>10.4f} "
             f"{student_f1_neg:>10.4f} {student_f1_neg - baseline_f1_neg:>+10.4f}")
    log.info(f"  {'F1-neutral':<16} {teacher_f1_neu:>10.4f} {baseline_f1_neu:>10.4f} "
             f"{student_f1_neu:>10.4f} {student_f1_neu - baseline_f1_neu:>+10.4f}")
    log.info("")

    # Knowledge transfer efficiency: how much of the teacher-baseline gap
    # does the student close?
    gap_closed = teacher_test_f1 - baseline_f1_macro
    if gap_closed > 0:
        transfer_efficiency = (student_f1_macro - baseline_f1_macro) / gap_closed
        log.info(f"  Knowledge transfer efficiency: {transfer_efficiency:.1%}")
        log.info(f"    (Student closed {transfer_efficiency:.1%} of the "
                 f"teacher-baseline F1 gap of {gap_closed:.4f})")
    else:
        transfer_efficiency = 0.0
        log.info("  Teacher did not outperform baseline on this test set")

    log.info("")
    if student_f1_macro > baseline_f1_macro + 0.005:
        log.info("  >>> DISTILLATION HELPED: Student outperforms baseline <<<")
    elif student_f1_macro > baseline_f1_macro - 0.005:
        log.info("  >>> NO SIGNIFICANT DIFFERENCE between distilled and baseline <<<")
    else:
        log.info("  >>> DISTILLATION HURT: Baseline outperforms distilled student <<<")

    # Model size comparison
    teacher_params_approx = 355_000_000  # ruRoBERTa-large
    log.info("")
    log.info(f"  Model sizes:")
    log.info(f"    Teacher (ruRoBERTa-large):  ~{teacher_params_approx:,} params")
    log.info(f"    Student (rubert-tiny2):       {trainable_params:,} params")
    log.info(f"    Compression ratio:           {teacher_params_approx / trainable_params:.1f}x smaller")

    # =====================================================================
    # Save metrics
    # =====================================================================
    elapsed = time.time() - t_start

    metrics = {
        "experiment": "v6_distillation",
        "hypothesis": (
            "Soft labels from the teacher (F1 0.868) encode inter-class "
            "similarity that hard labels miss, allowing the tiny student "
            "to learn a better decision boundary."
        ),
        "teacher": {
            "model": TEACHER_MODEL_NAME,
            "model_dir": str(TEACHER_MODEL_DIR),
            "test_f1_macro": round(teacher_test_f1, 4),
            "test_f1_positive": round(teacher_f1_pos, 4),
            "test_f1_negative": round(teacher_f1_neg, 4),
            "test_f1_neutral": round(teacher_f1_neu, 4),
            "params_approx": teacher_params_approx,
        },
        "baseline": {
            "model": STUDENT_BASE_MODEL,
            "method": "weighted_cross_entropy_hard_labels",
            "test_f1_macro": round(baseline_f1_macro, 4),
            "test_f1_positive": round(baseline_f1_pos, 4),
            "test_f1_negative": round(baseline_f1_neg, 4),
            "test_f1_neutral": round(baseline_f1_neu, 4),
        },
        "distilled_student": {
            "model": STUDENT_BASE_MODEL,
            "method": "knowledge_distillation",
            "test_f1_macro": round(student_f1_macro, 4),
            "test_f1_weighted": round(student_f1_weighted, 4),
            "test_f1_positive": round(student_f1_pos, 4),
            "test_f1_negative": round(student_f1_neg, 4),
            "test_f1_neutral": round(student_f1_neu, 4),
            "params": trainable_params,
            "report": student_report_dict,
        },
        "distillation_config": {
            "temperature": TEMPERATURE,
            "alpha": ALPHA,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "max_seq_length": MAX_SEQ_LENGTH,
            "fp16": device == "cuda",
            "eval_steps": EVAL_STEPS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        },
        "data": {
            "train_size": len(train_texts),
            "val_size": len(val_texts),
            "test_size": len(test_texts),
        },
        "analysis": {
            "teacher_train_accuracy": round(float(teacher_train_acc), 4),
            "teacher_avg_confidence": round(float(avg_confidence), 4),
            "teacher_avg_entropy": round(float(avg_entropy), 4),
            "teacher_label_disagreements": int(disagreements),
            "distillation_improvement_over_baseline": round(
                student_f1_macro - baseline_f1_macro, 4
            ),
            "knowledge_transfer_efficiency": round(transfer_efficiency, 4),
            "compression_ratio": round(teacher_params_approx / trainable_params, 1),
        },
        "training_time_minutes": round(elapsed / 60, 1),
    }

    metrics_path = final_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    log.info("")
    log.info(f"  Training time: {elapsed / 60:.1f} minutes")
    log.info(f"  Model saved to: {final_dir}")
    log.info(f"  Metrics saved to: {metrics_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
