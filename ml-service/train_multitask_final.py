"""
Multi-task model: sentiment + relevance + similarity in one forward pass.
Outputs:
  - sentiment_label: positive/negative/neutral (3 classes)
  - relevance_score: 0.0-1.0 (probability of being relevant)
  - similarity_score: 0.0-1.0 (normalized duplicate likelihood)

Base model: cointegrated/rubert-tiny2 (29M params, fast inference)
Dataset: Мобайл_X-Prod.xlsx (188K rows)
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
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

DATA_PATH = Path("data/dataset.xlsx")
OUTPUT_DIR = Path("models/multitask-final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "cointegrated/rubert-tiny2"
MAX_SEQ_LEN = 128
BATCH_SIZE = 64
EPOCHS = 5
LR = 2e-5
SEED = 42

SENTIMENT_MAP = {"позитив": 0, "негатив": 1, "нейтрально": 2}
SENTIMENT_LABELS = ["positive", "negative", "neutral"]


# ─── Dataset ──────────────────────────────────────────────────────────

class MultiTaskDataset(Dataset):
    def __init__(self, encodings, sentiment_labels, relevance_labels, similarity_scores):
        self.encodings = encodings
        self.sentiment = torch.tensor(sentiment_labels, dtype=torch.long)
        self.relevance = torch.tensor(relevance_labels, dtype=torch.float32)
        self.similarity = torch.tensor(similarity_scores, dtype=torch.float32)

    def __len__(self):
        return len(self.sentiment)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "sentiment": self.sentiment[idx],
            "relevance": self.relevance[idx],
            "similarity": self.similarity[idx],
        }


# ─── Model ────────────────────────────────────────────────────────────

class MultiTaskSentimentModel(nn.Module):
    """Three-headed model: sentiment + relevance + similarity."""

    def __init__(self, model_name, num_sentiment_classes=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size  # 312 for rubert-tiny2

        self.dropout = nn.Dropout(0.1)

        # Head 1: Sentiment (3-class classification)
        self.sentiment_head = nn.Linear(hidden, num_sentiment_classes)

        # Head 2: Relevance (binary, sigmoid output)
        self.relevance_head = nn.Linear(hidden, 1)

        # Head 3: Similarity/duplicate score (regression, sigmoid output)
        self.similarity_head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)

        sentiment_logits = self.sentiment_head(cls_output)
        relevance_logit = self.relevance_head(cls_output).squeeze(-1)
        similarity_logit = self.similarity_head(cls_output).squeeze(-1)

        return sentiment_logits, relevance_logit, similarity_logit


# ─── Data Loading ─────────────────────────────────────────────────────

def load_data(path: Path):
    log.info(f"Loading {path}...")
    df = pd.read_excel(path)
    log.info(f"Raw: {len(df)} rows")

    # Text
    df["text"] = (df["Заголовок"].fillna("") + " " + df["Текст"].fillna("")).str.strip()
    df = df[df["text"].str.len() > 5].copy()

    # Sentiment (3-class)
    df["sentiment_raw"] = df["Тональность"].str.strip().str.lower()
    df = df[df["sentiment_raw"].isin(SENTIMENT_MAP.keys())].copy()
    df["sentiment_label"] = df["sentiment_raw"].map(SENTIMENT_MAP)

    # Relevance (binary: 1=relevant, 0=irrelevant)
    df["relevance_raw"] = df["Редевантность"].str.strip().str.lower()
    df["relevance_label"] = (df["relevance_raw"] == "релевант").astype(float)

    # Similarity (normalized duplicate count → 0-1 score)
    # Higher = more duplicates = more "common" content
    max_dupes = df["Дублей"].quantile(0.99)  # cap at 99th percentile
    df["similarity_score"] = (df["Дублей"].clip(0, max_dupes) / max(max_dupes, 1)).astype(float)

    log.info(f"Cleaned: {len(df)} rows")
    log.info(f"Sentiment: {df['sentiment_raw'].value_counts().to_dict()}")
    log.info(f"Relevance: relevant={df['relevance_label'].sum():.0f}, irrelevant={len(df)-df['relevance_label'].sum():.0f}")
    log.info(f"Similarity: mean={df['similarity_score'].mean():.3f}, >0: {(df['similarity_score']>0).sum()}")

    return df


# ─── Training ─────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load data
    df = load_data(DATA_PATH)

    # Split
    train_df, temp_df = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df["sentiment_label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["sentiment_label"])
    log.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    log.info("Tokenizing...")
    train_enc = tokenizer(train_df["text"].tolist(), truncation=True, padding="max_length",
                          max_length=MAX_SEQ_LEN, return_tensors="pt")
    val_enc = tokenizer(val_df["text"].tolist(), truncation=True, padding="max_length",
                        max_length=MAX_SEQ_LEN, return_tensors="pt")
    test_enc = tokenizer(test_df["text"].tolist(), truncation=True, padding="max_length",
                         max_length=MAX_SEQ_LEN, return_tensors="pt")

    train_ds = MultiTaskDataset(train_enc, train_df["sentiment_label"].values,
                                train_df["relevance_label"].values, train_df["similarity_score"].values)
    val_ds = MultiTaskDataset(val_enc, val_df["sentiment_label"].values,
                              val_df["relevance_label"].values, val_df["similarity_score"].values)
    test_ds = MultiTaskDataset(test_enc, test_df["sentiment_label"].values,
                               test_df["relevance_label"].values, test_df["similarity_score"].values)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, num_workers=2)

    # Model
    model = MultiTaskSentimentModel(BASE_MODEL).to(device)
    params = sum(p.numel() for p in model.parameters())
    log.info(f"Model params: {params / 1e6:.1f}M")

    # Class weights for sentiment
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1, 2]),
                                         y=train_df["sentiment_label"].values)
    sentiment_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    log.info(f"Sentiment weights: {class_weights}")

    # Loss functions
    sentiment_loss_fn = nn.CrossEntropyLoss(weight=sentiment_weights)
    relevance_loss_fn = nn.BCEWithLogitsLoss()
    similarity_loss_fn = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=int(total_steps * 0.1))

    # Mixed precision
    scaler = torch.amp.GradScaler() if device == "cuda" else None

    # Training loop
    log.info("=" * 60)
    log.info("STARTING MULTI-TASK TRAINING")
    log.info(f"  Tasks: sentiment (3-class) + relevance (binary) + similarity (regression)")
    log.info(f"  Loss weights: sentiment=0.5, relevance=0.35, similarity=0.15")
    log.info("=" * 60)

    best_f1 = 0.0
    patience = 3
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        step = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sentiment_target = batch["sentiment"].to(device)
            relevance_target = batch["relevance"].to(device)
            similarity_target = batch["similarity"].to(device)

            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast(device_type="cuda"):
                    sent_logits, rel_score, sim_score = model(input_ids, attention_mask)
                    loss_sent = sentiment_loss_fn(sent_logits, sentiment_target)
                    loss_rel = relevance_loss_fn(rel_score, relevance_target)
                    loss_sim = similarity_loss_fn(sim_score, similarity_target)
                    loss = 0.5 * loss_sent + 0.35 * loss_rel + 0.15 * loss_sim

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                sent_logits, rel_score, sim_score = model(input_ids, attention_mask)
                loss_sent = sentiment_loss_fn(sent_logits, sentiment_target)
                loss_rel = relevance_loss_fn(rel_score, relevance_target)
                loss_sim = similarity_loss_fn(sim_score, similarity_target)
                loss = 0.5 * loss_sent + 0.35 * loss_rel + 0.15 * loss_sim

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()
            step += 1

            if step % 200 == 0:
                log.info(f"  Epoch {epoch+1} step {step}/{len(train_loader)} loss={loss.item():.4f}")

        avg_loss = total_loss / step
        log.info(f"Epoch {epoch+1}/{EPOCHS} avg_loss={avg_loss:.4f}")

        # Validation
        model.eval()
        all_sent_preds, all_sent_labels = [], []
        all_rel_preds, all_rel_labels = [], []
        all_sim_preds, all_sim_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                sent_logits, rel_score, sim_score = model(input_ids, attention_mask)

                all_sent_preds.extend(sent_logits.argmax(dim=-1).cpu().numpy())
                all_sent_labels.extend(batch["sentiment"].numpy())
                all_rel_preds.extend((torch.sigmoid(rel_score) > 0.5).int().cpu().numpy())
                all_rel_labels.extend(batch["relevance"].numpy().astype(int))
                all_sim_preds.extend(torch.sigmoid(sim_score).cpu().numpy())
                all_sim_labels.extend(batch["similarity"].numpy())

        sent_f1 = f1_score(all_sent_labels, all_sent_preds, average="macro")
        rel_f1 = f1_score(all_rel_labels, all_rel_preds, average="binary")
        sim_mse = np.mean((np.array(all_sim_preds) - np.array(all_sim_labels)) ** 2)

        log.info(f"  VAL: sentiment_f1={sent_f1:.4f}, relevance_f1={rel_f1:.4f}, similarity_mse={sim_mse:.4f}")

        # Combined score
        combined = 0.5 * sent_f1 + 0.35 * rel_f1 + 0.15 * (1 - sim_mse)
        log.info(f"  VAL combined={combined:.4f}")

        if combined > best_f1:
            best_f1 = combined
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), OUTPUT_DIR / "model.pt")
            tokenizer.save_pretrained(str(OUTPUT_DIR / "tokenizer"))
            log.info(f"  Saved best model (combined={combined:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info(f"  Early stopping at epoch {epoch+1}")
                break

    # Test evaluation
    log.info("=" * 60)
    log.info("EVALUATING ON TEST SET")
    log.info("=" * 60)

    model.load_state_dict(torch.load(OUTPUT_DIR / "model.pt", weights_only=True))
    model.eval()

    all_sent_preds, all_sent_labels = [], []
    all_rel_preds, all_rel_labels = [], []
    all_sim_preds, all_sim_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            sent_logits, rel_score, sim_score = model(input_ids, attention_mask)

            all_sent_preds.extend(sent_logits.argmax(dim=-1).cpu().numpy())
            all_sent_labels.extend(batch["sentiment"].numpy())
            all_rel_preds.extend((rel_score > 0.5).int().cpu().numpy())
            all_rel_labels.extend(batch["relevance"].numpy().astype(int))
            all_sim_preds.extend(sim_score.cpu().numpy())
            all_sim_labels.extend(batch["similarity"].numpy())

    # Sentiment report
    sent_report = classification_report(all_sent_labels, all_sent_preds,
                                        target_names=SENTIMENT_LABELS, output_dict=True)
    log.info(f"\nSentiment:")
    log.info(classification_report(all_sent_labels, all_sent_preds, target_names=SENTIMENT_LABELS))

    # Relevance report
    rel_report = classification_report(all_rel_labels, all_rel_preds,
                                       target_names=["irrelevant", "relevant"], output_dict=True)
    log.info(f"Relevance:")
    log.info(classification_report(all_rel_labels, all_rel_preds, target_names=["irrelevant", "relevant"]))

    # Similarity MSE
    sim_mse = float(np.mean((np.array(all_sim_preds) - np.array(all_sim_labels)) ** 2))
    log.info(f"Similarity MSE: {sim_mse:.4f}")

    # Save config
    config = {
        "base_model": BASE_MODEL,
        "tasks": ["sentiment", "relevance", "similarity"],
        "sentiment_f1_macro": round(sent_report["macro avg"]["f1-score"], 4),
        "relevance_f1": round(rel_report["relevant"]["f1-score"], 4),
        "similarity_mse": round(sim_mse, 4),
        "sentiment_report": sent_report,
        "relevance_report": rel_report,
        "dataset_size": len(df),
        "train_size": len(train_df),
        "label_map": {"sentiment": SENTIMENT_MAP, "relevance": {"relevant": 1, "irrelevant": 0}},
    }

    # Save model config for inference
    model_config = {
        "base_model": BASE_MODEL,
        "hidden_size": 312,
        "num_sentiment_classes": 3,
        "max_seq_len": MAX_SEQ_LEN,
    }
    with open(OUTPUT_DIR / "model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t_start
    log.info(f"\nDone in {elapsed / 60:.1f} minutes")
    log.info(f"Saved to {OUTPUT_DIR}")
    log.info(f"Sentiment F1: {config['sentiment_f1_macro']}")
    log.info(f"Relevance F1: {config['relevance_f1']}")
    log.info(f"Similarity MSE: {config['similarity_mse']}")


if __name__ == "__main__":
    main()
