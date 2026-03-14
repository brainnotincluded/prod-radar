"""
Prod Radar ML Training Pipeline
================================
Hybrid cascade architecture (consensus from 5-agent debate):
  Stage 0: TF-IDF + XGBoost for relevance classification
  Stage 1: rubert-tiny2 fine-tuned for sentiment + risk detection

Dataset: 188K Russian social media mentions about "Мобайл"
Target labels:
  - Тональность (Sentiment): позитив / негатив / нейтрально
  - Редевантность (Relevance): Релевант / Нерелевант
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────

DATA_PATH = Path("data/dataset.xlsx")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

SENTIMENT_MAP = {"позитив": 0, "негатив": 1, "нейтрально": 2}
SENTIMENT_LABELS = ["позитив", "негатив", "нейтрально"]

RELEVANCE_MAP = {"Релевант": 1, "Нерелевант": 0}
RELEVANCE_LABELS = ["Нерелевант", "Релевант"]

BERT_MODEL = "cointegrated/rubert-tiny2"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 2e-5

RANDOM_SEED = 42


# ─── Data Loading & Preprocessing ────────────────────────────────────

def load_and_preprocess(path: Path) -> pd.DataFrame:
    log.info(f"Loading dataset from {path}...")
    df = pd.read_excel(path)
    log.info(f"Raw dataset: {df.shape[0]} rows x {df.shape[1]} columns")

    # Combine title + text into a single text field
    df["text_combined"] = (
        df["Заголовок"].fillna("") + " " + df["Текст"].fillna("")
    ).str.strip()

    # Drop rows with empty text
    df = df[df["text_combined"].str.len() > 0].copy()
    log.info(f"After removing empty text: {df.shape[0]} rows")

    # Clean sentiment labels
    df["Тональность"] = df["Тональность"].str.strip().str.lower()
    valid_sentiments = set(SENTIMENT_MAP.keys())
    df = df[df["Тональность"].isin(valid_sentiments)].copy()

    # Clean relevance labels
    df["Редевантность"] = df["Редевантность"].str.strip()
    valid_relevance = set(RELEVANCE_MAP.keys())
    df = df[df["Редевантность"].isin(valid_relevance)].copy()

    # Map labels to integers
    df["sentiment_label"] = df["Тональность"].map(SENTIMENT_MAP)
    df["relevance_label"] = df["Редевантность"].map(RELEVANCE_MAP)

    log.info(f"Final dataset: {df.shape[0]} rows")
    log.info(f"Sentiment distribution:\n{df['Тональность'].value_counts()}")
    log.info(f"Relevance distribution:\n{df['Редевантность'].value_counts()}")

    return df


def split_data(df: pd.DataFrame):
    """Stratified split on sentiment (harder to balance)."""
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["sentiment_label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_df["sentiment_label"]
    )
    log.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


# ─── Stage 0: TF-IDF + XGBoost for Relevance ─────────────────────────

def train_relevance_xgboost(train_df, val_df, test_df):
    log.info("=" * 60)
    log.info("STAGE 0: Training TF-IDF + XGBoost for RELEVANCE")
    log.info("=" * 60)

    # TF-IDF with char n-grams (handles Russian morphology)
    tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=30000,
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
    )

    X_train = tfidf.fit_transform(train_df["text_combined"])
    X_val = tfidf.transform(val_df["text_combined"])
    X_test = tfidf.transform(test_df["text_combined"])

    y_train = train_df["relevance_label"].values
    y_val = val_df["relevance_label"].values
    y_test = test_df["relevance_label"].values

    # Compute class weights for imbalance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)
    log.info(f"Relevance scale_pos_weight: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        max_depth=6,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=RANDOM_SEED,
        early_stopping_rounds=30,
    )

    log.info("Training XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # Evaluate
    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred, target_names=RELEVANCE_LABELS, output_dict=True
    )
    log.info(f"\nRelevance Test Results:")
    log.info(classification_report(y_test, y_pred, target_names=RELEVANCE_LABELS))

    f1 = f1_score(y_test, y_pred, average="macro")
    log.info(f"Relevance F1-macro: {f1:.4f}")

    # Threshold tuning for high recall
    y_proba = model.predict_proba(X_test)[:, 1]
    best_threshold = 0.5
    best_recall = 0.0
    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred_t = (y_proba >= threshold).astype(int)
        recall = (y_pred_t[y_test == 1] == 1).mean() if (y_test == 1).sum() > 0 else 0
        precision = (y_test[y_pred_t == 1] == 1).mean() if (y_pred_t == 1).sum() > 0 else 0
        if recall >= 0.95 and precision > 0:
            if recall > best_recall or (recall == best_recall and precision > 0):
                best_threshold = threshold
                best_recall = recall
    log.info(f"Optimal threshold for >=95% recall: {best_threshold:.2f}")

    # Save
    out_dir = MODEL_DIR / "relevance_xgboost"
    out_dir.mkdir(exist_ok=True)
    model.save_model(str(out_dir / "model.json"))
    joblib.dump(tfidf, out_dir / "tfidf_vectorizer.joblib")
    with open(out_dir / "config.json", "w") as f:
        json.dump({
            "threshold": float(best_threshold),
            "f1_macro": float(f1),
            "report": report,
            "trained_at": datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)

    log.info(f"Relevance model saved to {out_dir}")
    return model, tfidf, report


# ─── Stage 1: rubert-tiny2 for Sentiment ──────────────────────────────

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_SEQ_LENGTH):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
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
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    return {"f1_macro": f1_macro, "f1_weighted": f1_weighted}


def train_sentiment_bert(train_df, val_df, test_df):
    log.info("=" * 60)
    log.info("STAGE 1: Fine-tuning rubert-tiny2 for SENTIMENT")
    log.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL,
        num_labels=len(SENTIMENT_MAP),
        problem_type="single_label_classification",
    )

    # Class weights for loss
    class_weights = compute_class_weight(
        "balanced",
        classes=np.array(list(SENTIMENT_MAP.values())),
        y=train_df["sentiment_label"].values,
    )
    log.info(f"Sentiment class weights: {dict(zip(SENTIMENT_LABELS, class_weights))}")

    # Create datasets
    train_texts = train_df["text_combined"].tolist()
    val_texts = val_df["text_combined"].tolist()
    test_texts = test_df["text_combined"].tolist()

    log.info("Tokenizing datasets...")
    train_dataset = SentimentDataset(train_texts, train_df["sentiment_label"].values, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_df["sentiment_label"].values, tokenizer)
    test_dataset = SentimentDataset(test_texts, test_df["sentiment_label"].values, tokenizer)

    output_dir = str(MODEL_DIR / "sentiment_rubert")

    training_args = TrainingArguments(
        output_dir=output_dir,
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
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        report_to="none",
        seed=RANDOM_SEED,
    )

    # Custom trainer with weighted loss
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            weight = torch.tensor(class_weights, dtype=torch.float32, device=logits.device)
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    log.info("Starting training...")
    trainer.train()

    # Evaluate on test set
    log.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    log.info(f"Sentiment Test Results: {test_results}")

    # Detailed classification report
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    report = classification_report(
        test_df["sentiment_label"].values,
        preds,
        target_names=SENTIMENT_LABELS,
        output_dict=True,
    )
    log.info(f"\nSentiment Classification Report:")
    log.info(classification_report(
        test_df["sentiment_label"].values,
        preds,
        target_names=SENTIMENT_LABELS,
    ))

    # Save final model
    final_dir = MODEL_DIR / "sentiment_rubert" / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    with open(final_dir / "config.json", "w") as f:
        json.dump({
            "model": BERT_MODEL,
            "f1_macro": float(test_results.get("eval_f1_macro", 0)),
            "f1_weighted": float(test_results.get("eval_f1_weighted", 0)),
            "report": report,
            "label_map": SENTIMENT_MAP,
            "trained_at": datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)

    log.info(f"Sentiment model saved to {final_dir}")
    return model, tokenizer, report


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("PROD RADAR ML TRAINING PIPELINE")
    log.info("=" * 60)

    # Load data
    df = load_and_preprocess(DATA_PATH)
    train_df, val_df, test_df = split_data(df)

    # Stage 0: Relevance (XGBoost)
    rel_model, tfidf, rel_report = train_relevance_xgboost(train_df, val_df, test_df)

    # Stage 1: Sentiment (rubert-tiny2)
    sent_model, tokenizer, sent_report = train_sentiment_bert(train_df, val_df, test_df)

    # Summary
    log.info("\n" + "=" * 60)
    log.info("TRAINING COMPLETE - SUMMARY")
    log.info("=" * 60)
    log.info(f"Relevance F1-macro:  {rel_report['macro avg']['f1-score']:.4f}")
    log.info(f"Sentiment F1-macro:  {sent_report['macro avg']['f1-score']:.4f}")
    log.info(f"Models saved to:     {MODEL_DIR.absolute()}")


if __name__ == "__main__":
    main()
