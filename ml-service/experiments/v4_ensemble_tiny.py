"""
Variant 4: Ensemble of 3 rubert-tiny2 models with different seeds.

Hypothesis:
    Averaging predictions from 3 diverse models reduces variance and improves
    reliability, while keeping inference fast (3x tiny is still faster than
    1x large).

Each model is trained with:
    - A different random seed (42, 123, 456)
    - A different dropout rate (0.1, 0.15, 0.2)
    - Balanced class weights (weighted cross-entropy)
    - batch_size=64, epochs=5, lr=2e-5, max_seq_len=128, fp16

Ensemble strategy:
    Average softmax probabilities from all 3 models, then argmax.

Run:
    cd ml-service && python3 experiments/v4_ensemble_tiny.py
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
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
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
OUTPUT_DIR = Path("models/v4_ensemble_tiny")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "cointegrated/rubert-tiny2"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 2e-5
EVAL_STEPS = 500
EARLY_STOPPING_PATIENCE = 3

SENTIMENT_LABELS = ["positive", "negative", "neutral"]

# Ensemble configuration: 3 models with different seeds and dropout rates
ENSEMBLE_CONFIGS = [
    {"seed": 42,  "dropout": 0.1},
    {"seed": 123, "dropout": 0.15},
    {"seed": 456, "dropout": 0.2},
]


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
# Weighted Trainer (balanced class weights via cross-entropy)
# ---------------------------------------------------------------------------
class WeightedTrainer(Trainer):
    """Trainer with class-weight-balanced cross-entropy loss."""

    def __init__(self, *args, class_weights_tensor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights_tensor = class_weights_tensor

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = F.cross_entropy(
            outputs.logits, labels, weight=self.class_weights_tensor
        )
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Train a single model
# ---------------------------------------------------------------------------
def train_single_model(
    model_idx: int,
    seed: int,
    dropout: float,
    tokenizer,
    train_dataset,
    val_dataset,
    class_weights_tensor,
    device: str,
) -> AutoModelForSequenceClassification:
    """Train one rubert-tiny2 model with a specific seed and dropout."""

    log.info("=" * 60)
    log.info(f"TRAINING MODEL {model_idx + 1}/3  (seed={seed}, dropout={dropout})")
    log.info("=" * 60)

    set_seed(seed)

    # Load model with custom dropout
    config = AutoConfig.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        classifier_dropout=dropout,
    )
    config.id2label = {0: "positive", 1: "negative", 2: "neutral"}
    config.label2id = {"positive": 0, "negative": 1, "neutral": 2}

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        config=config,
        ignore_mismatched_sizes=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Total params: {total_params:,}")
    log.info(f"  Trainable params: {trainable_params:,}")
    log.info(f"  Dropout: {dropout}")

    # Output directory for this model's checkpoints
    model_output_dir = OUTPUT_DIR / f"model_{model_idx}"
    model_output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(model_output_dir / "checkpoints"),
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
        seed=seed,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
        class_weights_tensor=class_weights_tensor,
    )

    trainer.train()

    # Save this model
    final_dir = model_output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    log.info(f"  Model {model_idx + 1} saved to {final_dir}")

    return model


# ---------------------------------------------------------------------------
# Evaluate a single model on test set
# ---------------------------------------------------------------------------
def evaluate_single_model(
    model,
    test_dataset,
    test_labels,
    model_idx: int,
    device: str,
) -> dict:
    """Evaluate one model; return classification report dict and raw logits."""

    log.info(f"\n--- Evaluating Model {model_idx + 1} ---")

    model.eval()
    model.to(device)

    all_logits = []
    dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False
    )

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu())

    all_logits = torch.cat(all_logits, dim=0)  # (N, 3)
    preds = torch.argmax(all_logits, dim=-1).numpy()

    f1_macro = f1_score(test_labels, preds, average="macro")
    report_str = classification_report(
        test_labels, preds, target_names=SENTIMENT_LABELS, digits=4,
    )
    report_dict = classification_report(
        test_labels, preds, target_names=SENTIMENT_LABELS, output_dict=True,
    )

    log.info(f"  Model {model_idx + 1} F1-macro: {f1_macro:.4f}")
    log.info(f"\n{report_str}")

    return {
        "logits": all_logits,
        "preds": preds,
        "f1_macro": f1_macro,
        "report": report_dict,
    }


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

    # ── Load Data ─────────────────────────────────────────────────────
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
    log.info(
        f"Train label distribution: "
        f"{{{', '.join(f'{SENTIMENT_LABELS[k]}: {v}' for k, v in sorted(train_dist.items()))}}}"
    )

    # ── Class Weights ─────────────────────────────────────────────────
    class_weights = compute_class_weight(
        "balanced",
        classes=np.array([0, 1, 2]),
        y=np.array(train_labels),
    )
    log.info(f"Class weights: {dict(zip(SENTIMENT_LABELS, class_weights))}")
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # ── Tokenize (once -- shared across all 3 models) ────────────────
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

    # ==================================================================
    # TRAIN 3 MODELS
    # ==================================================================
    log.info("\n" + "=" * 60)
    log.info("ENSEMBLE TRAINING: 3 x rubert-tiny2")
    log.info(f"  Configs: {ENSEMBLE_CONFIGS}")
    log.info("=" * 60)

    trained_models = []
    for idx, cfg in enumerate(ENSEMBLE_CONFIGS):
        model = train_single_model(
            model_idx=idx,
            seed=cfg["seed"],
            dropout=cfg["dropout"],
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            class_weights_tensor=weights_tensor,
            device=device,
        )
        trained_models.append(model)

    # ==================================================================
    # EVALUATE INDIVIDUAL MODELS
    # ==================================================================
    log.info("\n" + "=" * 60)
    log.info("INDIVIDUAL MODEL EVALUATION ON TEST SET")
    log.info("=" * 60)

    individual_results = []
    for idx, model in enumerate(trained_models):
        result = evaluate_single_model(
            model=model,
            test_dataset=test_dataset,
            test_labels=test_labels,
            model_idx=idx,
            device=device,
        )
        individual_results.append(result)

    # ==================================================================
    # ENSEMBLE PREDICTION: average softmax probabilities
    # ==================================================================
    log.info("\n" + "=" * 60)
    log.info("ENSEMBLE PREDICTION (average softmax)")
    log.info("=" * 60)

    # Collect softmax probabilities from each model
    all_probs = []
    for result in individual_results:
        probs = F.softmax(result["logits"], dim=-1)  # (N, 3)
        all_probs.append(probs)

    # Average across models: (3, N, 3) -> (N, 3)
    stacked_probs = torch.stack(all_probs, dim=0)  # (3, N, 3)
    avg_probs = stacked_probs.mean(dim=0)           # (N, 3)
    ensemble_preds = torch.argmax(avg_probs, dim=-1).numpy()

    # Ensemble classification report
    ensemble_f1_macro = f1_score(test_labels, ensemble_preds, average="macro")
    ensemble_f1_weighted = f1_score(test_labels, ensemble_preds, average="weighted")

    ensemble_report_str = classification_report(
        test_labels, ensemble_preds, target_names=SENTIMENT_LABELS, digits=4,
    )
    ensemble_report_dict = classification_report(
        test_labels, ensemble_preds, target_names=SENTIMENT_LABELS, output_dict=True,
    )

    log.info(f"\nENSEMBLE Classification Report:")
    log.info(f"\n{ensemble_report_str}")

    # ==================================================================
    # SUMMARY: compare all models
    # ==================================================================
    log.info("\n" + "=" * 60)
    log.info("RESULTS SUMMARY")
    log.info("=" * 60)

    log.info(f"\n  {'Model':<25} {'Seed':>6} {'Dropout':>8} {'F1-macro':>10}")
    log.info(f"  {'-'*25} {'-'*6} {'-'*8} {'-'*10}")
    for idx, (cfg, result) in enumerate(zip(ENSEMBLE_CONFIGS, individual_results)):
        log.info(
            f"  Model {idx + 1:<20} {cfg['seed']:>6} {cfg['dropout']:>8.2f} "
            f"{result['f1_macro']:>10.4f}"
        )
    log.info(f"  {'-'*25} {'-'*6} {'-'*8} {'-'*10}")
    log.info(f"  {'ENSEMBLE (avg softmax)':<25} {'---':>6} {'---':>8} {ensemble_f1_macro:>10.4f}")
    log.info("")

    best_individual = max(r["f1_macro"] for r in individual_results)
    worst_individual = min(r["f1_macro"] for r in individual_results)
    ensemble_gain = ensemble_f1_macro - best_individual

    log.info(f"  Best individual F1:   {best_individual:.4f}")
    log.info(f"  Worst individual F1:  {worst_individual:.4f}")
    log.info(f"  Individual spread:    {best_individual - worst_individual:.4f}")
    log.info(f"  Ensemble F1:          {ensemble_f1_macro:.4f}")
    log.info(f"  Ensemble vs best:     {ensemble_gain:+.4f}")
    log.info("")

    if ensemble_gain > 0:
        log.info("  >>> ENSEMBLE IMPROVES over best individual model <<<")
    elif ensemble_gain == 0:
        log.info("  >>> ENSEMBLE MATCHES best individual model <<<")
    else:
        log.info(
            f"  >>> ENSEMBLE is {abs(ensemble_gain):.4f} below best individual "
            f"(may still be more robust) <<<"
        )

    # ==================================================================
    # SAVE ENSEMBLE METRICS
    # ==================================================================
    metrics = {
        "variant": "v4_ensemble_tiny",
        "hypothesis": (
            "Averaging predictions from 3 diverse models reduces variance "
            "and improves reliability, while keeping inference fast."
        ),
        "base_model": BASE_MODEL,
        "ensemble_configs": ENSEMBLE_CONFIGS,
        "training": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "max_seq_length": MAX_SEQ_LENGTH,
            "fp16": device == "cuda",
            "balanced_class_weights": class_weights.tolist(),
        },
        "data": {
            "train_size": len(train_texts),
            "val_size": len(val_texts),
            "test_size": len(test_texts),
        },
        "individual_results": [
            {
                "model_idx": idx,
                "seed": cfg["seed"],
                "dropout": cfg["dropout"],
                "f1_macro": round(result["f1_macro"], 4),
                "f1_positive": round(result["report"]["positive"]["f1-score"], 4),
                "f1_negative": round(result["report"]["negative"]["f1-score"], 4),
                "f1_neutral": round(result["report"]["neutral"]["f1-score"], 4),
            }
            for idx, (cfg, result) in enumerate(zip(ENSEMBLE_CONFIGS, individual_results))
        ],
        "ensemble_results": {
            "strategy": "average_softmax",
            "f1_macro": round(ensemble_f1_macro, 4),
            "f1_weighted": round(ensemble_f1_weighted, 4),
            "f1_positive": round(ensemble_report_dict["positive"]["f1-score"], 4),
            "f1_negative": round(ensemble_report_dict["negative"]["f1-score"], 4),
            "f1_neutral": round(ensemble_report_dict["neutral"]["f1-score"], 4),
            "ensemble_gain_over_best": round(ensemble_gain, 4),
        },
        "ensemble_report": ensemble_report_dict,
    }

    metrics_path = OUTPUT_DIR / "ensemble_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    log.info(f"Ensemble metrics saved to {metrics_path}")

    # ── Timing ────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    log.info(f"\nTotal training time: {elapsed / 60:.1f} minutes")
    log.info(f"All 3 models saved to: {OUTPUT_DIR}/model_{{0,1,2}}/final/")
    log.info(f"Ensemble metrics saved to: {metrics_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
