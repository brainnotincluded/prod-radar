"""
Variant 10: Multi-task Learning -- Sentiment + Relevance Classification Jointly

Hypothesis:
    Our original dataset (Мобайл_X-Prod.xlsx) has BOTH sentiment (Тональность)
    and relevance (Редевантность) labels. Training both tasks jointly with a
    shared encoder produces better representations than single-task training.

Architecture:
    ┌──────────────────────────────┐
    │  cointegrated/rubert-tiny2   │  (shared encoder, 312-dim hidden)
    │  [CLS] pooled output         │
    └──────────┬───────────────────┘
               │
       ┌───────┴───────┐
       │               │
  ┌────▼────┐    ┌─────▼─────┐
  │ Head 1  │    │  Head 2   │
  │ Lin(3)  │    │  Lin(2)   │
  │sentiment│    │ relevance │
  └─────────┘    └───────────┘

Loss:
    L = 0.7 * CE_sentiment + 0.3 * CE_relevance

Label maps:
    Sentiment:  позитив -> 0, негатив -> 1, нейтрально -> 2
    Relevance:  Релевант -> 1, Нерелевант -> 0

Run:
    cd ~/projects/prod-radar/ml-service && python3 experiments/v10_multitask.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

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
OUTPUT_DIR = Path("models/v10_multitask")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "cointegrated/rubert-tiny2"
HIDDEN_DIM = 312  # rubert-tiny2 hidden size
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 2e-5
SEED = 42

# Task weights in combined loss
SENTIMENT_LOSS_WEIGHT = 0.7
RELEVANCE_LOSS_WEIGHT = 0.3

# Label maps
SENTIMENT_MAP = {"позитив": 0, "негатив": 1, "нейтрально": 2}
SENTIMENT_LABELS = ["positive", "negative", "neutral"]
NUM_SENTIMENT_CLASSES = 3

RELEVANCE_MAP = {"релевант": 1, "нерелевант": 0}
RELEVANCE_LABELS = ["irrelevant", "relevant"]
NUM_RELEVANCE_CLASSES = 2


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MultiTaskDataset(Dataset):
    """Dataset that returns token encodings + both task labels."""

    def __init__(self, encodings, sentiment_labels, relevance_labels):
        self.encodings = encodings
        self.sentiment_labels = torch.tensor(sentiment_labels, dtype=torch.long)
        self.relevance_labels = torch.tensor(relevance_labels, dtype=torch.long)

    def __len__(self):
        return len(self.sentiment_labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "sentiment_labels": self.sentiment_labels[idx],
            "relevance_labels": self.relevance_labels[idx],
        }


# ---------------------------------------------------------------------------
# Multi-task Model
# ---------------------------------------------------------------------------
class MultiTaskModel(nn.Module):
    """
    Shared encoder with two classification heads.

    Head 1: sentiment (3-class)
    Head 2: relevance (binary)
    """

    def __init__(self, encoder_name, num_sentiment_classes=3, num_relevance_classes=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size  # 312 for rubert-tiny2

        # Dropout before heads (same as BERT default)
        self.dropout = nn.Dropout(0.1)

        # Task-specific classification heads
        self.sentiment_head = nn.Linear(hidden_size, num_sentiment_classes)
        self.relevance_head = nn.Linear(hidden_size, num_relevance_classes)

        log.info(f"MultiTaskModel initialized:")
        log.info(f"  Encoder: {encoder_name} (hidden_size={hidden_size})")
        log.info(f"  Sentiment head: Linear({hidden_size}, {num_sentiment_classes})")
        log.info(f"  Relevance head: Linear({hidden_size}, {num_relevance_classes})")

    def forward(self, input_ids, attention_mask):
        """
        Returns:
            sentiment_logits: (batch_size, num_sentiment_classes)
            relevance_logits: (batch_size, num_relevance_classes)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        pooled = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
        pooled = self.dropout(pooled)

        sentiment_logits = self.sentiment_head(pooled)
        relevance_logits = self.relevance_head(pooled)

        return sentiment_logits, relevance_logits


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(path: Path):
    """
    Load Excel file and extract both sentiment and relevance labels.

    Returns a DataFrame with columns: text, sentiment_label, relevance_label
    """
    log.info(f"Loading data from {path}")
    df = pd.read_excel(path)
    log.info(f"Raw rows: {len(df)}")

    # Combine title + text
    df["text"] = (
        df["Заголовок"].fillna("") + " " + df["Текст"].fillna("")
    ).str.strip()
    df = df[df["text"].str.len() > 5].copy()

    # --- Sentiment labels ---
    df["sentiment_raw"] = df["Тональность"].str.strip().str.lower()
    df = df[df["sentiment_raw"].isin(SENTIMENT_MAP.keys())].copy()
    df["sentiment_label"] = df["sentiment_raw"].map(SENTIMENT_MAP)

    # --- Relevance labels ---
    df["relevance_raw"] = df["Редевантность"].str.strip().str.lower()
    df = df[df["relevance_raw"].isin(RELEVANCE_MAP.keys())].copy()
    df["relevance_label"] = df["relevance_raw"].map(RELEVANCE_MAP)

    result = df[["text", "sentiment_label", "relevance_label"]].copy()
    result = result.dropna().reset_index(drop=True)

    log.info(f"Cleaned: {len(result)} rows with both labels")
    return result


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, dataloader, device, prefix=""):
    """Evaluate model on both tasks. Returns dict of metrics."""
    model.eval()
    all_sent_preds, all_sent_labels = [], []
    all_rel_preds, all_rel_labels = [], []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sent_labels = batch["sentiment_labels"].to(device)
            rel_labels = batch["relevance_labels"].to(device)

            sent_logits, rel_logits = model(input_ids, attention_mask)

            # Loss
            sent_loss = F.cross_entropy(sent_logits, sent_labels)
            rel_loss = F.cross_entropy(rel_logits, rel_labels)
            loss = SENTIMENT_LOSS_WEIGHT * sent_loss + RELEVANCE_LOSS_WEIGHT * rel_loss
            total_loss += loss.item()
            num_batches += 1

            # Predictions
            all_sent_preds.extend(sent_logits.argmax(dim=-1).cpu().numpy())
            all_sent_labels.extend(sent_labels.cpu().numpy())
            all_rel_preds.extend(rel_logits.argmax(dim=-1).cpu().numpy())
            all_rel_labels.extend(rel_labels.cpu().numpy())

    all_sent_preds = np.array(all_sent_preds)
    all_sent_labels = np.array(all_sent_labels)
    all_rel_preds = np.array(all_rel_preds)
    all_rel_labels = np.array(all_rel_labels)

    # Sentiment metrics
    sent_f1_macro = f1_score(all_sent_labels, all_sent_preds, average="macro")
    sent_f1_weighted = f1_score(all_sent_labels, all_sent_preds, average="weighted")
    sent_acc = accuracy_score(all_sent_labels, all_sent_preds)

    # Relevance metrics
    rel_f1_macro = f1_score(all_rel_labels, all_rel_preds, average="macro")
    rel_f1_weighted = f1_score(all_rel_labels, all_rel_preds, average="weighted")
    rel_acc = accuracy_score(all_rel_labels, all_rel_preds)

    metrics = {
        "loss": total_loss / max(num_batches, 1),
        "sentiment_f1_macro": sent_f1_macro,
        "sentiment_f1_weighted": sent_f1_weighted,
        "sentiment_accuracy": sent_acc,
        "relevance_f1_macro": rel_f1_macro,
        "relevance_f1_weighted": rel_f1_weighted,
        "relevance_accuracy": rel_acc,
    }

    if prefix:
        log.info(f"[{prefix}] loss={metrics['loss']:.4f} | "
                 f"sent_f1={sent_f1_macro:.4f} sent_acc={sent_acc:.4f} | "
                 f"rel_f1={rel_f1_macro:.4f} rel_acc={rel_acc:.4f}")

    return metrics, {
        "sentiment": (all_sent_preds, all_sent_labels),
        "relevance": (all_rel_preds, all_rel_labels),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()

    # --- Device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        log.info("Device: CPU")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Load Data ─────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("LOADING DATA")
    log.info("=" * 60)

    df = load_data(DATA_PATH)

    # Label distributions
    log.info(f"\nSentiment distribution:")
    for name, idx in SENTIMENT_MAP.items():
        count = (df["sentiment_label"] == idx).sum()
        log.info(f"  {name} ({idx}): {count} ({100 * count / len(df):.1f}%)")

    log.info(f"\nRelevance distribution:")
    for name, idx in RELEVANCE_MAP.items():
        count = (df["relevance_label"] == idx).sum()
        log.info(f"  {name} ({idx}): {count} ({100 * count / len(df):.1f}%)")

    # ── Train/Val/Test Split ──────────────────────────────────────────
    # Stratify by sentiment (the harder task with 3 classes)
    train_df, temp_df = train_test_split(
        df, test_size=0.15, random_state=SEED, stratify=df["sentiment_label"],
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["sentiment_label"],
    )

    log.info(f"\nSplit sizes:")
    log.info(f"  Train: {len(train_df)}")
    log.info(f"  Val:   {len(val_df)}")
    log.info(f"  Test:  {len(test_df)}")

    # ── Class Weights ─────────────────────────────────────────────────
    sentiment_class_weights = compute_class_weight(
        "balanced",
        classes=np.array([0, 1, 2]),
        y=train_df["sentiment_label"].values,
    )
    relevance_class_weights = compute_class_weight(
        "balanced",
        classes=np.array([0, 1]),
        y=train_df["relevance_label"].values,
    )
    log.info(f"\nSentiment class weights: {dict(zip(SENTIMENT_LABELS, sentiment_class_weights))}")
    log.info(f"Relevance class weights: {dict(zip(RELEVANCE_LABELS, relevance_class_weights))}")

    sentiment_weights_tensor = torch.tensor(
        sentiment_class_weights, dtype=torch.float32
    ).to(device)
    relevance_weights_tensor = torch.tensor(
        relevance_class_weights, dtype=torch.float32
    ).to(device)

    # ── Tokenize ──────────────────────────────────────────────────────
    log.info(f"\nLoading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(texts):
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        )

    log.info("Tokenizing train set...")
    train_enc = tokenize(train_df["text"].tolist())
    log.info("Tokenizing val set...")
    val_enc = tokenize(val_df["text"].tolist())
    log.info("Tokenizing test set...")
    test_enc = tokenize(test_df["text"].tolist())

    train_dataset = MultiTaskDataset(
        train_enc, train_df["sentiment_label"].values, train_df["relevance_label"].values
    )
    val_dataset = MultiTaskDataset(
        val_enc, val_df["sentiment_label"].values, val_df["relevance_label"].values
    )
    test_dataset = MultiTaskDataset(
        test_enc, test_df["sentiment_label"].values, test_df["relevance_label"].values
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────
    log.info(f"\nBuilding multi-task model with encoder: {BASE_MODEL}")
    model = MultiTaskModel(
        encoder_name=BASE_MODEL,
        num_sentiment_classes=NUM_SENTIMENT_CLASSES,
        num_relevance_classes=NUM_RELEVANCE_CLASSES,
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Total params: {total_params:,}")
    log.info(f"Trainable params: {trainable_params:,}")

    # ── Optimizer & Scheduler ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # FP16 via GradScaler (CUDA only; MPS uses float32)
    use_fp16 = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None
    autocast_dtype = torch.float16 if use_fp16 else torch.float32
    log.info(f"FP16: {use_fp16}")

    # ── Training Loop ─────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("TRAINING CONFIGURATION")
    log.info("=" * 60)
    log.info(f"  Model:           {BASE_MODEL}")
    log.info(f"  Batch size:      {BATCH_SIZE}")
    log.info(f"  Epochs:          {EPOCHS}")
    log.info(f"  Learning rate:   {LEARNING_RATE}")
    log.info(f"  Max seq length:  {MAX_SEQ_LENGTH}")
    log.info(f"  Total steps:     {total_steps}")
    log.info(f"  Warmup steps:    {warmup_steps}")
    log.info(f"  Loss weights:    sentiment={SENTIMENT_LOSS_WEIGHT}, relevance={RELEVANCE_LOSS_WEIGHT}")
    log.info(f"  FP16:            {use_fp16}")

    log.info("=" * 60)
    log.info("STARTING MULTI-TASK TRAINING")
    log.info("=" * 60)

    best_val_f1 = 0.0
    best_epoch = 0
    patience = 2
    patience_counter = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        running_sent_loss = 0.0
        running_rel_loss = 0.0
        num_steps = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sent_labels = batch["sentiment_labels"].to(device)
            rel_labels = batch["relevance_labels"].to(device)

            optimizer.zero_grad()

            if use_fp16:
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    sent_logits, rel_logits = model(input_ids, attention_mask)
                    sent_loss = F.cross_entropy(
                        sent_logits, sent_labels, weight=sentiment_weights_tensor
                    )
                    rel_loss = F.cross_entropy(
                        rel_logits, rel_labels, weight=relevance_weights_tensor
                    )
                    loss = (
                        SENTIMENT_LOSS_WEIGHT * sent_loss
                        + RELEVANCE_LOSS_WEIGHT * rel_loss
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                sent_logits, rel_logits = model(input_ids, attention_mask)
                sent_loss = F.cross_entropy(
                    sent_logits, sent_labels, weight=sentiment_weights_tensor
                )
                rel_loss = F.cross_entropy(
                    rel_logits, rel_labels, weight=relevance_weights_tensor
                )
                loss = (
                    SENTIMENT_LOSS_WEIGHT * sent_loss
                    + RELEVANCE_LOSS_WEIGHT * rel_loss
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            running_loss += loss.item()
            running_sent_loss += sent_loss.item()
            running_rel_loss += rel_loss.item()
            num_steps += 1

            # Log every 100 steps
            if (step + 1) % 100 == 0:
                avg_loss = running_loss / num_steps
                avg_sent = running_sent_loss / num_steps
                avg_rel = running_rel_loss / num_steps
                lr_current = scheduler.get_last_lr()[0]
                log.info(
                    f"  Epoch {epoch}/{EPOCHS} | Step {step + 1}/{len(train_loader)} | "
                    f"loss={avg_loss:.4f} (sent={avg_sent:.4f}, rel={avg_rel:.4f}) | "
                    f"lr={lr_current:.2e}"
                )

        # --- End of epoch ---
        epoch_time = time.time() - epoch_start
        avg_train_loss = running_loss / max(num_steps, 1)

        # Validation
        val_metrics, _ = evaluate(model, val_loader, device, prefix=f"Val Epoch {epoch}")

        # Combined F1 for model selection (weighted same as loss)
        combined_f1 = (
            SENTIMENT_LOSS_WEIGHT * val_metrics["sentiment_f1_macro"]
            + RELEVANCE_LOSS_WEIGHT * val_metrics["relevance_f1_macro"]
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_sentiment_f1_macro": round(val_metrics["sentiment_f1_macro"], 4),
            "val_relevance_f1_macro": round(val_metrics["relevance_f1_macro"], 4),
            "val_combined_f1": round(combined_f1, 4),
            "epoch_time_s": round(epoch_time, 1),
        }
        history.append(epoch_record)

        log.info(
            f"Epoch {epoch}/{EPOCHS} done in {epoch_time:.0f}s | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_combined_f1={combined_f1:.4f} | "
            f"val_sent_f1={val_metrics['sentiment_f1_macro']:.4f} | "
            f"val_rel_f1={val_metrics['relevance_f1_macro']:.4f}"
        )

        # Save best model
        if combined_f1 > best_val_f1:
            best_val_f1 = combined_f1
            best_epoch = epoch
            patience_counter = 0

            best_dir = OUTPUT_DIR / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_dir / "model.pt")
            tokenizer.save_pretrained(str(best_dir))
            log.info(f"  >> New best model saved (combined_f1={combined_f1:.4f})")
        else:
            patience_counter += 1
            log.info(f"  No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                log.info(f"  Early stopping at epoch {epoch}")
                break

    # ── Load Best Model & Evaluate on Test ────────────────────────────
    log.info("=" * 60)
    log.info("EVALUATING BEST MODEL ON TEST SET")
    log.info("=" * 60)

    best_state = torch.load(
        OUTPUT_DIR / "best" / "model.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(best_state)
    log.info(f"Loaded best model from epoch {best_epoch}")

    test_metrics, test_predictions = evaluate(model, test_loader, device, prefix="TEST")

    # --- Detailed sentiment report ---
    sent_preds, sent_labels = test_predictions["sentiment"]
    log.info("\n" + "=" * 60)
    log.info("SENTIMENT CLASSIFICATION REPORT (Test Set)")
    log.info("=" * 60)
    sent_report_str = classification_report(
        sent_labels, sent_preds, target_names=SENTIMENT_LABELS, digits=4,
    )
    sent_report_dict = classification_report(
        sent_labels, sent_preds, target_names=SENTIMENT_LABELS, output_dict=True,
    )
    log.info(f"\n{sent_report_str}")

    # --- Detailed relevance report ---
    rel_preds, rel_labels = test_predictions["relevance"]
    log.info("=" * 60)
    log.info("RELEVANCE CLASSIFICATION REPORT (Test Set)")
    log.info("=" * 60)
    rel_report_str = classification_report(
        rel_labels, rel_preds, target_names=RELEVANCE_LABELS, digits=4,
    )
    rel_report_dict = classification_report(
        rel_labels, rel_preds, target_names=RELEVANCE_LABELS, output_dict=True,
    )
    log.info(f"\n{rel_report_str}")

    # ── Save Final Model ──────────────────────────────────────────────
    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Saving final model to {final_dir}")

    # Save full model state dict
    torch.save(model.state_dict(), final_dir / "model.pt")

    # Save encoder and tokenizer separately (for reuse)
    model.encoder.save_pretrained(str(final_dir / "encoder"))
    tokenizer.save_pretrained(str(final_dir / "encoder"))

    # Save head weights separately
    torch.save(model.sentiment_head.state_dict(), final_dir / "sentiment_head.pt")
    torch.save(model.relevance_head.state_dict(), final_dir / "relevance_head.pt")

    # Save model config for reconstruction
    model_config = {
        "encoder_name": BASE_MODEL,
        "hidden_dim": HIDDEN_DIM,
        "num_sentiment_classes": NUM_SENTIMENT_CLASSES,
        "num_relevance_classes": NUM_RELEVANCE_CLASSES,
        "sentiment_map": SENTIMENT_MAP,
        "relevance_map": RELEVANCE_MAP,
        "sentiment_labels": SENTIMENT_LABELS,
        "relevance_labels": RELEVANCE_LABELS,
    }
    with open(final_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2, ensure_ascii=False)

    # ── Save Metrics ──────────────────────────────────────────────────
    sent_f1_macro = sent_report_dict["macro avg"]["f1-score"]
    sent_f1_weighted = sent_report_dict["weighted avg"]["f1-score"]
    rel_f1_macro = rel_report_dict["macro avg"]["f1-score"]
    rel_f1_weighted = rel_report_dict["weighted avg"]["f1-score"]

    metrics = {
        "experiment": "v10_multitask",
        "hypothesis": "Joint sentiment+relevance training produces better shared representations",
        "base_model": BASE_MODEL,
        "architecture": {
            "encoder": BASE_MODEL,
            "hidden_dim": HIDDEN_DIM,
            "sentiment_head": f"Linear({HIDDEN_DIM}, {NUM_SENTIMENT_CLASSES})",
            "relevance_head": f"Linear({HIDDEN_DIM}, {NUM_RELEVANCE_CLASSES})",
            "loss": f"{SENTIMENT_LOSS_WEIGHT}*CE_sent + {RELEVANCE_LOSS_WEIGHT}*CE_rel",
        },
        "config": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "epochs_trained": best_epoch,
            "learning_rate": LEARNING_RATE,
            "max_seq_length": MAX_SEQ_LENGTH,
            "fp16": use_fp16,
            "sentiment_loss_weight": SENTIMENT_LOSS_WEIGHT,
            "relevance_loss_weight": RELEVANCE_LOSS_WEIGHT,
            "seed": SEED,
        },
        "data": {
            "source": str(DATA_PATH),
            "total_samples": len(df),
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
        },
        "results": {
            "sentiment": {
                "f1_macro": round(sent_f1_macro, 4),
                "f1_weighted": round(sent_f1_weighted, 4),
                "accuracy": round(test_metrics["sentiment_accuracy"], 4),
                "f1_positive": round(sent_report_dict["positive"]["f1-score"], 4),
                "f1_negative": round(sent_report_dict["negative"]["f1-score"], 4),
                "f1_neutral": round(sent_report_dict["neutral"]["f1-score"], 4),
                "report": sent_report_dict,
            },
            "relevance": {
                "f1_macro": round(rel_f1_macro, 4),
                "f1_weighted": round(rel_f1_weighted, 4),
                "accuracy": round(test_metrics["relevance_accuracy"], 4),
                "f1_irrelevant": round(rel_report_dict["irrelevant"]["f1-score"], 4),
                "f1_relevant": round(rel_report_dict["relevant"]["f1-score"], 4),
                "report": rel_report_dict,
            },
            "combined_f1": round(best_val_f1, 4),
        },
        "training_history": history,
        "baseline_sentiment_f1_macro": 0.7579,
        "sentiment_improvement": round(sent_f1_macro - 0.7579, 4),
    }

    with open(final_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    log.info("\n" + "=" * 60)
    log.info("V10 MULTI-TASK RESULTS SUMMARY")
    log.info("=" * 60)
    log.info(f"")
    log.info(f"  SENTIMENT (3-class):")
    log.info(f"    F1-macro:    {sent_f1_macro:.4f}")
    log.info(f"    F1-weighted: {sent_f1_weighted:.4f}")
    log.info(f"    Accuracy:    {test_metrics['sentiment_accuracy']:.4f}")
    log.info(f"    F1-positive: {sent_report_dict['positive']['f1-score']:.4f}")
    log.info(f"    F1-negative: {sent_report_dict['negative']['f1-score']:.4f}")
    log.info(f"    F1-neutral:  {sent_report_dict['neutral']['f1-score']:.4f}")
    log.info(f"")
    log.info(f"  RELEVANCE (binary):")
    log.info(f"    F1-macro:      {rel_f1_macro:.4f}")
    log.info(f"    F1-weighted:   {rel_f1_weighted:.4f}")
    log.info(f"    Accuracy:      {test_metrics['relevance_accuracy']:.4f}")
    log.info(f"    F1-irrelevant: {rel_report_dict['irrelevant']['f1-score']:.4f}")
    log.info(f"    F1-relevant:   {rel_report_dict['relevant']['f1-score']:.4f}")
    log.info(f"")
    log.info(f"  Baseline sentiment F1-macro: 0.7579")
    log.info(f"  Improvement:                 {sent_f1_macro - 0.7579:+.4f}")
    log.info(f"")
    log.info(f"  Best epoch: {best_epoch}")
    log.info(f"  Training time: {elapsed / 60:.1f} minutes")
    log.info(f"  Model saved to: {final_dir}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
