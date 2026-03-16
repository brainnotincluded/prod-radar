"""
Experiment V9: Advanced text preprocessing before training.

Hypothesis: Russian social media text has noise (URLs, @mentions, emojis,
            repeated characters, excessive punctuation) that confuses the model.
            Smart preprocessing can boost accuracy.

Approach:
  1. Preprocess ALL texts with a pipeline:
     a. Replace URLs with [URL]
     b. Replace @mentions with [USER]
     c. Replace emojis with sentiment tokens: [POSITIVE_EMOJI], [NEGATIVE_EMOJI], [SARCASM_EMOJI]
     d. Normalize repeated characters: "оооочень" -> "очень"
     e. Remove excessive punctuation: "!!!!!!" -> "!!"
     f. Normalize whitespace
     g. Keep Russian text, numbers, and basic punctuation
  2. Train cointegrated/rubert-tiny2 on preprocessed data
  3. Also train on RAW data (no preprocessing) for A/B comparison
  4. Print side-by-side results

Dataset:    data/merged_train.jsonl, data/merged_val.jsonl, data/merged_test.jsonl
            Format: {"text": "...", "label": 0|1|2}
            Labels: 0=positive, 1=negative, 2=neutral
Base model: cointegrated/rubert-tiny2 (29M params)

Run:
  cd ml-service && python3 experiments/v9_preprocessing.py
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
TRAIN_DATA = Path("data/merged_train.jsonl")
VAL_DATA = Path("data/merged_val.jsonl")
TEST_DATA = Path("data/merged_test.jsonl")
OUTPUT_DIR = Path("models/v9_preprocessing")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "cointegrated/rubert-tiny2"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 2e-5
SEED = 42

SENTIMENT_LABELS = ["positive", "negative", "neutral"]
BASELINE_F1 = 0.758  # Previous best (rubert-tiny2 + weighted CE)


# ---------------------------------------------------------------------------
# Emoji sentiment dictionaries
# ---------------------------------------------------------------------------

# Positive emojis -- express happiness, approval, love, celebration
POSITIVE_EMOJIS = set(
    "\U0001f600\U0001f601\U0001f602\U0001f603\U0001f604\U0001f605\U0001f606"  # grinning faces
    "\U0001f607\U0001f609\U0001f60a\U0001f60b\U0001f60c\U0001f60d\U0001f60e"  # smiling, heart-eyes
    "\U0001f60f\U0001f618\U0001f617\U0001f619\U0001f61a\U0001f61b\U0001f61c"  # kissing, wink
    "\U0001f61d\U0001f638\U0001f639\U0001f63a\U0001f63b\U0001f63c\U0001f63d"  # cat faces
    "\U0001f44d\U0001f44f\U0001f44c\U0001f64c\U0001f64f"  # thumbs up, clap, ok, raised hands, pray
    "\U0001f389\U0001f38a\U0001f381\U0001f496\U0001f495\U0001f493\U0001f497"  # party, hearts
    "\U0001f499\U0001f49a\U0001f49b\U0001f49c\U0001f49d\U0001f49e\U0001f49f"  # colored hearts
    "\u2764\u2665\u263a\u2705\u2b50\U0001f31f\U0001f31e"  # red heart, star, sun
    "\U0001f525\U0001f4af\U0001f4aa\U0001f929\U0001f970\U0001f973"  # fire, 100, muscle
    "\U0001f60e\U0001f917\U0001f92d\U0001f92b\U0001f972"  # sunglasses, hugs
    "\U00002764\U0000FE0F"  # red heart with variation selector
)

# Negative emojis -- express anger, sadness, disgust, frustration
NEGATIVE_EMOJIS = set(
    "\U0001f620\U0001f621\U0001f624\U0001f622\U0001f625\U0001f62d"  # angry, crying
    "\U0001f623\U0001f626\U0001f627\U0001f628\U0001f629\U0001f62a"  # anguished, fearful
    "\U0001f62b\U0001f630\U0001f631\U0001f632\U0001f633\U0001f634"  # tired, screaming
    "\U0001f635\U0001f636\U0001f637\U0001f641\U0001f616\U0001f61e"  # sick, frowning
    "\U0001f61f\U0001f615\U0001f910\U0001f912\U0001f915\U0001f922"  # confused, nauseated
    "\U0001f92e\U0001f92f\U0001f971\U0001f974\U0001f976\U0001f975"  # mind blown, bored
    "\U0001f44e\U0001f4a9\U0001f480\U00002639\U0001f614"  # thumbs down, skull, frown
    "\U0001f63e\U0001f63f\U0001f640"  # weary cat, crying cat, scared cat
    "\U0001f645\U0001f6ab\U0000274c"  # no gesture, prohibited, cross mark
)

# Sarcasm/irony emojis -- typically signal irony or passive-aggressiveness
SARCASM_EMOJIS = set(
    "\U0001f643\U0001f928\U0001f914\U0001f644\U0001f611\U0001f612"  # upside-down, raised brow, thinking, eye-roll
    "\U0001f60f\U0001f925\U0001f921\U0001f913"  # smirk, lying, clown, nerd
)


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

# Regex patterns (compiled once for performance)
_RE_URL = re.compile(
    r"https?://\S+|www\.\S+|[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}(?:/\S*)?",
    re.IGNORECASE,
)
_RE_MENTION = re.compile(r"@[\w]+")
_RE_REPEATED_CHAR = re.compile(r"(.)\1{2,}")  # 3+ repeated chars
_RE_EXCESSIVE_PUNCT = re.compile(r"([!?.])\1{2,}")  # 3+ repeated punctuation
_RE_MULTI_SPACE = re.compile(r"\s+")
# Keep: Russian letters, digits, basic punctuation, brackets, special tokens
_RE_ALLOWED = re.compile(
    r"[^"
    r"а-яА-ЯёЁ"           # Russian letters
    r"a-zA-Z"              # Latin (for special tokens like [URL])
    r"0-9"                 # digits
    r"\s"                  # whitespace
    r".,!?;:\-\(\)\[\]"   # basic punctuation and brackets
    r"\"'"                 # quotes
    r"]"
)


def replace_emojis(text: str) -> str:
    """Replace emojis with sentiment-category tokens."""
    result = []
    for char in text:
        if char in SARCASM_EMOJIS:
            result.append(" [SARCASM_EMOJI] ")
        elif char in POSITIVE_EMOJIS:
            result.append(" [POSITIVE_EMOJI] ")
        elif char in NEGATIVE_EMOJIS:
            result.append(" [NEGATIVE_EMOJI] ")
        elif unicodedata.category(char).startswith("So"):
            # Other symbols (remaining emojis not in our dictionaries)
            # Skip them -- they are noise
            continue
        else:
            result.append(char)
    return "".join(result)


def normalize_repeated_chars(text: str) -> str:
    """
    Normalize repeated characters in Russian words.
    "оооочень" -> "очень", "ааааа" -> "а", "нееет" -> "нет"

    Only applies to Cyrillic characters to avoid mangling special tokens.
    """
    # Match 3+ consecutive identical Cyrillic characters, reduce to 1
    return re.sub(r"([а-яА-ЯёЁ])\1{2,}", r"\1", text)


def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline for Russian social media text.

    Steps (order matters):
      1. Replace URLs with [URL]
      2. Replace @mentions with [USER]
      3. Replace emojis with sentiment tokens
      4. Normalize repeated characters
      5. Remove excessive punctuation (keep max 2)
      6. Remove non-allowed characters
      7. Normalize whitespace
    """
    # 1. URLs -> [URL]
    text = _RE_URL.sub(" [URL] ", text)

    # 2. @mentions -> [USER]
    text = _RE_MENTION.sub(" [USER] ", text)

    # 3. Emojis -> sentiment tokens
    text = replace_emojis(text)

    # 4. Normalize repeated characters in Russian words
    text = normalize_repeated_chars(text)

    # 5. Excessive punctuation: "!!!!" -> "!!", "???" -> "??"
    text = _RE_EXCESSIVE_PUNCT.sub(r"\1\1", text)

    # 6. Remove characters outside allowed set
    text = _RE_ALLOWED.sub("", text)

    # 7. Normalize whitespace
    text = _RE_MULTI_SPACE.sub(" ", text).strip()

    return text


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
# Training function (runs one full experiment: train + evaluate)
# ---------------------------------------------------------------------------
def train_and_evaluate(
    name: str,
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
    output_subdir: str,
    device: str,
) -> dict:
    """
    Train a rubert-tiny2 model on the given data and evaluate on test set.

    Returns a dict with all metrics.
    """
    t_start = time.time()

    log.info("=" * 60)
    log.info(f"TRAINING: {name}")
    log.info("=" * 60)

    # ── Class Weights ────────────────────────────────────────────────
    class_weights = compute_class_weight(
        "balanced",
        classes=np.array([0, 1, 2]),
        y=np.array(train_labels),
    )
    log.info(f"Class weights: {dict(zip(SENTIMENT_LABELS, class_weights.round(4)))}")
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # ── Tokenize ─────────────────────────────────────────────────────
    log.info(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Add special tokens used by preprocessing
    special_tokens = ["[URL]", "[USER]", "[POSITIVE_EMOJI]", "[NEGATIVE_EMOJI]", "[SARCASM_EMOJI]"]
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    log.info(f"Added {num_added} special tokens to tokenizer")

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

    # ── Model ────────────────────────────────────────────────────────
    log.info(f"Loading model: {BASE_MODEL}")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=3, ignore_mismatched_sizes=True,
    )
    model.config.id2label = {0: "positive", 1: "negative", 2: "neutral"}
    model.config.label2id = {"positive": 0, "negative": 1, "neutral": 2}

    # Resize embeddings to account for new special tokens
    model.resize_token_embeddings(len(tokenizer))

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Total params: {total_params:,}")

    # ── Training Args ────────────────────────────────────────────────
    run_output_dir = OUTPUT_DIR / output_subdir
    run_output_dir.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = len(train_dataset) // BATCH_SIZE
    eval_steps = max(steps_per_epoch // 4, 50)

    training_args = TrainingArguments(
        output_dir=str(run_output_dir / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
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

    # ── Weighted Trainer ─────────────────────────────────────────────
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

    # ── Train ────────────────────────────────────────────────────────
    log.info(f"  Data: {len(train_dataset)} train / {len(val_dataset)} val / {len(test_dataset)} test")
    log.info(f"  Batch: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LEARNING_RATE}")
    log.info(f"  Eval every {eval_steps} steps")

    trainer.train()

    # ── Evaluate ─────────────────────────────────────────────────────
    log.info(f"Evaluating {name} on test set...")
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)

    report_str = classification_report(
        test_labels, preds, target_names=SENTIMENT_LABELS, digits=4,
    )
    report_dict = classification_report(
        test_labels, preds, target_names=SENTIMENT_LABELS, output_dict=True,
    )
    log.info(f"\n{name} Test Results:\n{report_str}")

    elapsed = time.time() - t_start

    results = {
        "name": name,
        "f1_macro": report_dict["macro avg"]["f1-score"],
        "f1_weighted": report_dict["weighted avg"]["f1-score"],
        "f1_positive": report_dict["positive"]["f1-score"],
        "f1_negative": report_dict["negative"]["f1-score"],
        "f1_neutral": report_dict["neutral"]["f1-score"],
        "accuracy": report_dict["accuracy"],
        "training_time_min": round(elapsed / 60, 1),
        "report": report_dict,
    }

    # ── Save model (only for preprocessed variant -- the main experiment) ──
    if output_subdir == "preprocessed":
        final_dir = OUTPUT_DIR / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Saving preprocessed model to {final_dir}")
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))

    return results


# ---------------------------------------------------------------------------
# Preprocessing statistics
# ---------------------------------------------------------------------------
def print_preprocessing_stats(raw_texts: list[str], preprocessed_texts: list[str]):
    """Show what the preprocessing pipeline changed."""
    total = len(raw_texts)
    changed = sum(1 for r, p in zip(raw_texts, preprocessed_texts) if r != p)

    # Count specific replacements across all texts
    url_count = 0
    user_count = 0
    pos_emoji_count = 0
    neg_emoji_count = 0
    sarc_emoji_count = 0

    for text in preprocessed_texts:
        url_count += text.count("[URL]")
        user_count += text.count("[USER]")
        pos_emoji_count += text.count("[POSITIVE_EMOJI]")
        neg_emoji_count += text.count("[NEGATIVE_EMOJI]")
        sarc_emoji_count += text.count("[SARCASM_EMOJI]")

    # Average length change
    raw_avg_len = np.mean([len(t) for t in raw_texts])
    pre_avg_len = np.mean([len(t) for t in preprocessed_texts])

    log.info("=" * 60)
    log.info("PREPROCESSING STATISTICS")
    log.info("=" * 60)
    log.info(f"  Total texts:        {total}")
    log.info(f"  Texts changed:      {changed} ({changed / total * 100:.1f}%)")
    log.info(f"  Avg length (raw):   {raw_avg_len:.0f} chars")
    log.info(f"  Avg length (clean): {pre_avg_len:.0f} chars")
    log.info(f"  Length reduction:    {(1 - pre_avg_len / raw_avg_len) * 100:.1f}%")
    log.info(f"")
    log.info(f"  Token replacements found in preprocessed texts:")
    log.info(f"    [URL]:            {url_count}")
    log.info(f"    [USER]:           {user_count}")
    log.info(f"    [POSITIVE_EMOJI]: {pos_emoji_count}")
    log.info(f"    [NEGATIVE_EMOJI]: {neg_emoji_count}")
    log.info(f"    [SARCASM_EMOJI]:  {sarc_emoji_count}")

    # Show a few examples of changed texts
    log.info(f"")
    log.info("  Sample transformations (first 5 changed):")
    shown = 0
    for raw, pre in zip(raw_texts, preprocessed_texts):
        if raw != pre and shown < 5:
            log.info(f"    RAW:   {raw[:100]}{'...' if len(raw) > 100 else ''}")
            log.info(f"    CLEAN: {pre[:100]}{'...' if len(pre) > 100 else ''}")
            log.info(f"")
            shown += 1
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()

    # ── Device ───────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    if device == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Load Raw Data ────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("EXPERIMENT V9: ADVANCED TEXT PREPROCESSING")
    log.info("=" * 60)
    log.info(f"Hypothesis: Smart preprocessing of Russian social media noise")
    log.info(f"            (URLs, @mentions, emojis, repeated chars) boosts accuracy.")
    log.info("")

    log.info("Loading data...")
    train_texts_raw, train_labels = load_jsonl(TRAIN_DATA)
    val_texts_raw, val_labels = load_jsonl(VAL_DATA)
    test_texts_raw, test_labels = load_jsonl(TEST_DATA)

    log.info(f"Train: {len(train_texts_raw)} samples")
    log.info(f"Val:   {len(val_texts_raw)} samples")
    log.info(f"Test:  {len(test_texts_raw)} samples")

    train_dist = Counter(train_labels)
    log.info(f"Train label distribution: { {SENTIMENT_LABELS[k]: v for k, v in sorted(train_dist.items())} }")

    # ── Apply Preprocessing ──────────────────────────────────────────
    log.info("")
    log.info("Applying preprocessing pipeline...")
    t_preprocess = time.time()

    train_texts_preprocessed = [preprocess_text(t) for t in train_texts_raw]
    val_texts_preprocessed = [preprocess_text(t) for t in val_texts_raw]
    test_texts_preprocessed = [preprocess_text(t) for t in test_texts_raw]

    preprocess_time = time.time() - t_preprocess
    log.info(f"Preprocessing done in {preprocess_time:.1f}s")

    # Show preprocessing stats on training data
    print_preprocessing_stats(train_texts_raw, train_texts_preprocessed)

    # ── Experiment A: Train on RAW data (baseline control) ───────────
    log.info("")
    log.info("#" * 60)
    log.info("# EXPERIMENT A: RAW TEXT (no preprocessing)")
    log.info("#" * 60)

    results_raw = train_and_evaluate(
        name="RAW (no preprocessing)",
        train_texts=train_texts_raw,
        train_labels=train_labels,
        val_texts=val_texts_raw,
        val_labels=val_labels,
        test_texts=test_texts_raw,
        test_labels=test_labels,
        output_subdir="raw",
        device=device,
    )

    # ── Experiment B: Train on PREPROCESSED data ─────────────────────
    log.info("")
    log.info("#" * 60)
    log.info("# EXPERIMENT B: PREPROCESSED TEXT")
    log.info("#" * 60)

    results_preprocessed = train_and_evaluate(
        name="PREPROCESSED",
        train_texts=train_texts_preprocessed,
        train_labels=train_labels,
        val_texts=val_texts_preprocessed,
        val_labels=val_labels,
        test_texts=test_texts_preprocessed,
        test_labels=test_labels,
        output_subdir="preprocessed",
        device=device,
    )

    # ── Side-by-Side Comparison ──────────────────────────────────────
    total_time = time.time() - t_start

    log.info("")
    log.info("=" * 70)
    log.info("V9 EXPERIMENT RESULTS: RAW vs PREPROCESSED")
    log.info("=" * 70)
    log.info("")

    header = f"  {'Metric':<20} {'Baseline':>10} {'Raw':>10} {'Preproc':>10} {'Delta':>10}"
    sep = f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}"

    log.info(header)
    log.info(sep)

    delta = results_preprocessed["f1_macro"] - results_raw["f1_macro"]
    log.info(
        f"  {'F1-macro':<20} "
        f"{BASELINE_F1:>10.4f} "
        f"{results_raw['f1_macro']:>10.4f} "
        f"{results_preprocessed['f1_macro']:>10.4f} "
        f"{delta:>+10.4f}"
    )

    delta_w = results_preprocessed["f1_weighted"] - results_raw["f1_weighted"]
    log.info(
        f"  {'F1-weighted':<20} "
        f"{'':>10} "
        f"{results_raw['f1_weighted']:>10.4f} "
        f"{results_preprocessed['f1_weighted']:>10.4f} "
        f"{delta_w:>+10.4f}"
    )

    delta_pos = results_preprocessed["f1_positive"] - results_raw["f1_positive"]
    log.info(
        f"  {'F1-positive':<20} "
        f"{'':>10} "
        f"{results_raw['f1_positive']:>10.4f} "
        f"{results_preprocessed['f1_positive']:>10.4f} "
        f"{delta_pos:>+10.4f}"
    )

    delta_neg = results_preprocessed["f1_negative"] - results_raw["f1_negative"]
    log.info(
        f"  {'F1-negative':<20} "
        f"{'':>10} "
        f"{results_raw['f1_negative']:>10.4f} "
        f"{results_preprocessed['f1_negative']:>10.4f} "
        f"{delta_neg:>+10.4f}"
    )

    delta_neu = results_preprocessed["f1_neutral"] - results_raw["f1_neutral"]
    log.info(
        f"  {'F1-neutral':<20} "
        f"{'':>10} "
        f"{results_raw['f1_neutral']:>10.4f} "
        f"{results_preprocessed['f1_neutral']:>10.4f} "
        f"{delta_neu:>+10.4f}"
    )

    delta_acc = results_preprocessed["accuracy"] - results_raw["accuracy"]
    log.info(
        f"  {'Accuracy':<20} "
        f"{'':>10} "
        f"{results_raw['accuracy']:>10.4f} "
        f"{results_preprocessed['accuracy']:>10.4f} "
        f"{delta_acc:>+10.4f}"
    )

    log.info(sep)
    log.info(
        f"  {'Training time':<20} "
        f"{'':>10} "
        f"{results_raw['training_time_min']:>9.1f}m "
        f"{results_preprocessed['training_time_min']:>9.1f}m "
        f"{'':>10}"
    )

    log.info("")

    # ── Verdict ──────────────────────────────────────────────────────
    f1_raw = results_raw["f1_macro"]
    f1_pre = results_preprocessed["f1_macro"]

    if f1_pre > f1_raw + 0.005:
        log.info("  >>> HYPOTHESIS SUPPORTED: Preprocessing improved F1-macro <<<")
        log.info(f"      Preprocessed ({f1_pre:.4f}) > Raw ({f1_raw:.4f}) by {f1_pre - f1_raw:+.4f}")
    elif f1_raw > f1_pre + 0.005:
        log.info("  >>> HYPOTHESIS NOT SUPPORTED: Raw data performed better <<<")
        log.info(f"      Raw ({f1_raw:.4f}) > Preprocessed ({f1_pre:.4f}) by {f1_raw - f1_pre:+.4f}")
    else:
        log.info("  >>> INCONCLUSIVE: No significant difference (<0.005) <<<")
        log.info(f"      Raw: {f1_raw:.4f}, Preprocessed: {f1_pre:.4f}")

    if max(f1_raw, f1_pre) > BASELINE_F1:
        best_name = "Preprocessed" if f1_pre >= f1_raw else "Raw"
        best_f1 = max(f1_raw, f1_pre)
        log.info(f"      Best ({best_name}) beats baseline: {best_f1:.4f} > {BASELINE_F1:.4f}")
    else:
        log.info(f"      Neither variant beats baseline ({BASELINE_F1:.4f})")

    log.info("")
    log.info(f"  Preprocessing pipeline:")
    log.info(f"    1. URLs -> [URL]")
    log.info(f"    2. @mentions -> [USER]")
    log.info(f"    3. Emojis -> [POSITIVE_EMOJI] / [NEGATIVE_EMOJI] / [SARCASM_EMOJI]")
    log.info(f"    4. Repeated chars normalized (Cyrillic only)")
    log.info(f"    5. Excessive punctuation reduced (max 2)")
    log.info(f"    6. Non-Russian/non-punctuation chars removed")
    log.info(f"    7. Whitespace normalized")
    log.info("")
    log.info(f"  Total experiment time: {total_time / 60:.1f} minutes")
    log.info(f"  Preprocessed model saved to: {OUTPUT_DIR / 'final'}")
    log.info("=" * 70)

    # ── Save comprehensive metrics ───────────────────────────────────
    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "experiment": "v9_preprocessing",
        "hypothesis": "Smart preprocessing of Russian social media noise boosts accuracy",
        "base_model": BASE_MODEL,
        "preprocessing_steps": [
            "URLs -> [URL]",
            "@mentions -> [USER]",
            "Emojis -> [POSITIVE_EMOJI] / [NEGATIVE_EMOJI] / [SARCASM_EMOJI]",
            "Repeated chars normalized (Cyrillic only)",
            "Excessive punctuation reduced (max 2)",
            "Non-allowed chars removed",
            "Whitespace normalized",
        ],
        "config": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "max_seq_length": MAX_SEQ_LENGTH,
            "loss": "weighted_cross_entropy",
            "early_stopping_patience": 3,
            "fp16": device == "cuda",
            "seed": SEED,
        },
        "data": {
            "train_size": len(train_texts_raw),
            "val_size": len(val_texts_raw),
            "test_size": len(test_texts_raw),
        },
        "results_raw": {
            "f1_macro": round(results_raw["f1_macro"], 4),
            "f1_weighted": round(results_raw["f1_weighted"], 4),
            "f1_positive": round(results_raw["f1_positive"], 4),
            "f1_negative": round(results_raw["f1_negative"], 4),
            "f1_neutral": round(results_raw["f1_neutral"], 4),
            "accuracy": round(results_raw["accuracy"], 4),
            "training_time_min": results_raw["training_time_min"],
        },
        "results_preprocessed": {
            "f1_macro": round(results_preprocessed["f1_macro"], 4),
            "f1_weighted": round(results_preprocessed["f1_weighted"], 4),
            "f1_positive": round(results_preprocessed["f1_positive"], 4),
            "f1_negative": round(results_preprocessed["f1_negative"], 4),
            "f1_neutral": round(results_preprocessed["f1_neutral"], 4),
            "accuracy": round(results_preprocessed["accuracy"], 4),
            "training_time_min": results_preprocessed["training_time_min"],
        },
        "delta_preprocessed_minus_raw": {
            "f1_macro": round(f1_pre - f1_raw, 4),
            "f1_weighted": round(delta_w, 4),
            "f1_positive": round(delta_pos, 4),
            "f1_negative": round(delta_neg, 4),
            "f1_neutral": round(delta_neu, 4),
            "accuracy": round(delta_acc, 4),
        },
        "baseline_f1_macro": BASELINE_F1,
        "best_f1_macro": round(max(f1_raw, f1_pre), 4),
        "best_variant": "preprocessed" if f1_pre >= f1_raw else "raw",
        "hypothesis_supported": f1_pre > f1_raw + 0.005,
        "total_time_min": round(total_time / 60, 1),
        "report_raw": results_raw["report"],
        "report_preprocessed": results_preprocessed["report"],
    }

    metrics_path = final_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    log.info(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
