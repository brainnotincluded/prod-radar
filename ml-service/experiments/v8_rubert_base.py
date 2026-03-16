"""
Experiment V8: DeepPavlov/rubert-base-cased-sentence with LLRD + Focal Loss

Hypothesis: A mid-size model (110M params) hits the sweet spot -- better than tiny
            (29M), faster than large (355M). rubert-base-cased-sentence has
            sentence-level pre-training which is ideal for sentiment classification.

Techniques:
  1) LLRD  -- Layer-wise Learning Rate Decay (decay=0.85, 12 layers)
  2) Focal Loss -- gamma=2.0 with balanced class weights

Config:
  Base model:     DeepPavlov/rubert-base-cased-sentence (110M params, 12 layers)
  Batch size:     32 x 2 gradient_accumulation = 64 effective
  Epochs:         5
  Learning rate:  3e-5
  LLRD decay:     0.85
  Focal gamma:    2.0
  Max seq len:    128
  FP16:           True (on CUDA)
  Early stopping: patience=3 on f1_macro

Data:
  data/merged_train.jsonl  -- {"text": "...", "label": 0|1|2}
  data/merged_val.jsonl
  data/merged_test.jsonl

Labels:
  0 = positive, 1 = negative, 2 = neutral

Run:
  cd ml-service && python3 experiments/v8_rubert_base.py
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
OUTPUT_DIR = Path("models/v8_rubert_base")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "DeepPavlov/rubert-base-cased-sentence"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 2  # effective batch = 32 * 2 = 64
EPOCHS = 5
LEARNING_RATE = 3e-5
LLRD_DECAY_FACTOR = 0.85
FOCAL_GAMMA = 2.0
SEED = 42
EVAL_STEPS = 500
EARLY_STOPPING_PATIENCE = 3

SENTIMENT_LABELS = ["positive", "negative", "neutral"]

# Previous best for comparison
BASELINE_F1 = 0.868  # Phase 1+2: ruRoBERTa-large


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
# Focal Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss: down-weights well-classified examples so training focuses
    on hard, misclassified ones.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma=0 recovers standard weighted CE.
    gamma=2 is a strong default for imbalanced classification.
    """

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # class weights tensor, shape (num_classes,)
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        focal_loss = ((1 - pt) ** self.gamma * ce).mean()
        return focal_loss


# ---------------------------------------------------------------------------
# LLRD -- Layer-wise Learning Rate Decay
# ---------------------------------------------------------------------------
def get_llrd_param_groups(model, base_lr=3e-5, decay_factor=0.85, weight_decay=0.01):
    """
    Assign different learning rates to each transformer layer.

    Top layers (close to output) get higher LR -- they need to adapt most.
    Bottom layers (close to input) get lower LR -- they capture general
    linguistic features that should be preserved.

    For rubert-base-cased-sentence with 12 layers:
      - Classifier head:  base_lr * 5
      - Layer 11 (top):   base_lr
      - Layer 10:         base_lr * 0.85
      - ...
      - Layer 0 (bottom): base_lr * 0.85^11
      - Embeddings:       base_lr * 0.85^12
    """
    param_groups = []
    num_layers = model.config.num_hidden_layers  # 12 for rubert-base

    # Track assigned parameters to ensure nothing is missed
    assigned_params = set()

    # --- Transformer layers (from top to bottom) ---
    for layer_idx in range(num_layers - 1, -1, -1):
        lr = base_lr * (decay_factor ** (num_layers - 1 - layer_idx))
        layer_params = []
        layer_params_no_decay = []
        for name, param in model.named_parameters():
            if f"layer.{layer_idx}." in name or f"layers.{layer_idx}." in name:
                if param.requires_grad:
                    assigned_params.add(name)
                    if "bias" in name or "LayerNorm" in name or "layernorm" in name:
                        layer_params_no_decay.append(param)
                    else:
                        layer_params.append(param)

        if layer_params:
            param_groups.append({
                "params": layer_params,
                "lr": lr,
                "weight_decay": weight_decay,
            })
        if layer_params_no_decay:
            param_groups.append({
                "params": layer_params_no_decay,
                "lr": lr,
                "weight_decay": 0.0,
            })

    # --- Embeddings get the smallest LR ---
    embed_lr = base_lr * (decay_factor ** num_layers)
    embed_params = []
    embed_params_no_decay = []
    for name, param in model.named_parameters():
        if "embeddings" in name and name not in assigned_params:
            if param.requires_grad:
                assigned_params.add(name)
                if "bias" in name or "LayerNorm" in name or "layernorm" in name:
                    embed_params_no_decay.append(param)
                else:
                    embed_params.append(param)

    if embed_params:
        param_groups.append({
            "params": embed_params,
            "lr": embed_lr,
            "weight_decay": weight_decay,
        })
    if embed_params_no_decay:
        param_groups.append({
            "params": embed_params_no_decay,
            "lr": embed_lr,
            "weight_decay": 0.0,
        })

    # --- Classifier head gets the highest LR ---
    head_lr = base_lr * 5
    head_params = []
    head_params_no_decay = []
    for name, param in model.named_parameters():
        if name not in assigned_params and param.requires_grad:
            assigned_params.add(name)
            if "bias" in name or "LayerNorm" in name or "layernorm" in name:
                head_params_no_decay.append(param)
            else:
                head_params.append(param)

    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": head_lr,
            "weight_decay": 0.0,  # no decay on classifier
        })
    if head_params_no_decay:
        param_groups.append({
            "params": head_params_no_decay,
            "lr": head_lr,
            "weight_decay": 0.0,
        })

    # Sanity check: all params accounted for
    total_assigned = sum(p.numel() for g in param_groups for p in g["params"])
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_assigned != total_trainable:
        log.warning(
            f"LLRD param mismatch: assigned {total_assigned:,} vs trainable {total_trainable:,}. "
            f"Missing {total_trainable - total_assigned:,} parameters."
        )
    else:
        log.info(f"LLRD: all {total_trainable:,} trainable params assigned to {len(param_groups)} groups")

    return param_groups


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
# FocalLLRDTrainer -- combines LLRD + Focal Loss
# ---------------------------------------------------------------------------
class FocalLLRDTrainer(Trainer):
    """
    Custom HuggingFace Trainer that integrates LLRD and Focal Loss.

    - LLRD: overrides create_optimizer to assign per-layer learning rates
    - Focal Loss: overrides compute_loss to use focal loss with class weights
    """

    def __init__(self, *args, focal_loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss_fn = focal_loss_fn

    def create_optimizer(self):
        """Override to use LLRD parameter groups instead of uniform LR."""
        if self.optimizer is None:
            param_groups = get_llrd_param_groups(
                self.model,
                base_lr=self.args.learning_rate,
                decay_factor=LLRD_DECAY_FACTOR,
                weight_decay=self.args.weight_decay,
            )
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Use Focal Loss instead of standard cross-entropy."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.focal_loss_fn(logits, labels)
        inputs["labels"] = labels  # restore for potential downstream use
        return (loss, outputs) if return_outputs else loss


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
        log.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ---- Load Data -------------------------------------------------------
    log.info("=" * 60)
    log.info("LOADING DATA")
    log.info("=" * 60)

    train_texts, train_labels = load_jsonl(TRAIN_DATA)
    val_texts, val_labels = load_jsonl(VAL_DATA)
    test_texts, test_labels = load_jsonl(TEST_DATA)

    log.info(f"Train: {len(train_texts)} samples")
    log.info(f"Val:   {len(val_texts)} samples")
    log.info(f"Test:  {len(test_texts)} samples")

    # Label distribution
    train_dist = Counter(train_labels)
    log.info(f"Train label distribution: { {SENTIMENT_LABELS[k]: v for k, v in sorted(train_dist.items())} }")

    # ---- Class Weights ----------------------------------------------------
    class_weights = compute_class_weight(
        "balanced",
        classes=np.array([0, 1, 2]),
        y=np.array(train_labels),
    )
    log.info(f"Class weights: {dict(zip(SENTIMENT_LABELS, class_weights))}")
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # ---- Tokenize ---------------------------------------------------------
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

    train_dataset = SentimentDataset(train_enc, train_labels)
    val_dataset = SentimentDataset(val_enc, val_labels)
    test_dataset = SentimentDataset(test_enc, test_labels)

    # ---- Model ------------------------------------------------------------
    log.info(f"Loading model: {BASE_MODEL}")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        ignore_mismatched_sizes=True,
    )
    model.config.id2label = {0: "positive", 1: "negative", 2: "neutral"}
    model.config.label2id = {"positive": 0, "negative": 1, "neutral": 2}

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Total params: {total_params:,}")
    log.info(f"Trainable params: {trainable_params:,}")
    log.info(f"Num hidden layers: {model.config.num_hidden_layers}")

    # ---- Focal Loss -------------------------------------------------------
    focal_loss_fn = FocalLoss(alpha=weights_tensor, gamma=FOCAL_GAMMA)
    log.info(f"Focal Loss: gamma={FOCAL_GAMMA}, alpha={class_weights.tolist()}")

    # ---- Training Args ----------------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
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

    # ---- Trainer ----------------------------------------------------------
    log.info("=" * 60)
    log.info("V8 EXPERIMENT CONFIGURATION")
    log.info("=" * 60)
    log.info(f"  Model:           {BASE_MODEL}")
    log.info(f"  Params:          ~110M ({total_params:,} total)")
    log.info(f"  Hidden layers:   {model.config.num_hidden_layers}")
    log.info(f"  Batch size:      {BATCH_SIZE} x {GRADIENT_ACCUMULATION_STEPS} = {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS} effective")
    log.info(f"  Epochs:          {EPOCHS}")
    log.info(f"  Base LR:         {LEARNING_RATE}")
    log.info(f"  LLRD decay:      {LLRD_DECAY_FACTOR}")
    log.info(f"  Focal gamma:     {FOCAL_GAMMA}")
    log.info(f"  Max seq length:  {MAX_SEQ_LENGTH}")
    log.info(f"  FP16:            {device == 'cuda'}")
    log.info(f"  Eval steps:      {EVAL_STEPS}")
    log.info(f"  Early stopping:  patience={EARLY_STOPPING_PATIENCE}")
    log.info(f"  Techniques:      LLRD + Focal Loss")

    trainer = FocalLLRDTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
        focal_loss_fn=focal_loss_fn,
    )

    # ---- Train ------------------------------------------------------------
    log.info("=" * 60)
    log.info("STARTING V8 TRAINING")
    log.info("  Model: DeepPavlov/rubert-base-cased-sentence")
    log.info("  Techniques: LLRD (decay=0.85) + Focal Loss (gamma=2.0)")
    log.info("=" * 60)
    trainer.train()

    # ---- Evaluate ---------------------------------------------------------
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
    f1_neg = report_dict["negative"]["f1-score"]
    f1_pos = report_dict["positive"]["f1-score"]
    f1_neu = report_dict["neutral"]["f1-score"]

    # ---- Save -------------------------------------------------------------
    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Saving model to {final_dir}")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metrics = {
        "experiment": "v8_rubert_base",
        "base_model": BASE_MODEL,
        "hypothesis": (
            "Mid-size model (110M) with sentence-level pre-training hits the "
            "sweet spot -- better than tiny, faster than large."
        ),
        "techniques": ["LLRD", "FocalLoss"],
        "config": {
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "llrd_decay_factor": LLRD_DECAY_FACTOR,
            "focal_gamma": FOCAL_GAMMA,
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
        "results": {
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "f1_positive": round(f1_pos, 4),
            "f1_negative": round(f1_neg, 4),
            "f1_neutral": round(f1_neu, 4),
        },
        "baseline_f1_macro": BASELINE_F1,
        "improvement_over_baseline": round(f1_macro - BASELINE_F1, 4),
        "report": report_dict,
    }

    with open(final_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # ---- Summary ----------------------------------------------------------
    elapsed = time.time() - t_start
    log.info("=" * 60)
    log.info("V8 RESULTS vs BASELINE")
    log.info("=" * 60)
    log.info(f"")
    log.info(f"  {'Metric':<20} {'Baseline':>10} {'V8':>10} {'Delta':>10}")
    log.info(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    log.info(f"  {'F1-macro':<20} {BASELINE_F1:>10.4f} {f1_macro:>10.4f} {f1_macro - BASELINE_F1:>+10.4f}")
    log.info(f"  {'F1-positive':<20} {'':>10} {f1_pos:>10.4f} {'':>10}")
    log.info(f"  {'F1-negative':<20} {'':>10} {f1_neg:>10.4f} {'':>10}")
    log.info(f"  {'F1-neutral':<20} {'':>10} {f1_neu:>10.4f} {'':>10}")
    log.info(f"  {'F1-weighted':<20} {'':>10} {f1_weighted:>10.4f} {'':>10}")
    log.info(f"")

    if f1_macro >= BASELINE_F1:
        log.info(f"  >>> MATCHES/BEATS BASELINE ({BASELINE_F1:.3f}) <<<")
    elif f1_macro >= 0.85:
        log.info(f"  >>> STRONG RESULT (F1 >= 0.85) but below baseline by {BASELINE_F1 - f1_macro:.4f} <<<")
    elif f1_macro >= 0.80:
        log.info(f"  >>> DECENT RESULT (F1 >= 0.80) -- {BASELINE_F1 - f1_macro:.4f} below baseline <<<")
    else:
        log.info(f"  >>> BELOW TARGET (F1 < 0.80) <<<")

    log.info(f"")
    log.info(f"  Techniques applied:")
    log.info(f"    1. LLRD (decay={LLRD_DECAY_FACTOR}, {model.config.num_hidden_layers} layers)")
    log.info(f"    2. Focal Loss (gamma={FOCAL_GAMMA}, balanced class weights)")
    log.info(f"")
    log.info(f"  Training time: {elapsed / 60:.1f} minutes")
    log.info(f"  Model saved to: {final_dir}")
    log.info(f"  Metrics saved to: {final_dir / 'metrics.json'}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
