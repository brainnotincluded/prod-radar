"""
Download and merge open-source Russian sentiment datasets from HuggingFace
into a unified training file compatible with finetune.py / finetune_v2.py.

Label convention (matches existing model):
    0 = positive
    1 = negative
    2 = neutral

Usage:
    cd ml-service && python3 download_datasets.py

Output files (in data/):
    merged_train.jsonl   -- 85 % of data
    merged_val.jsonl     -- 7.5 %
    merged_test.jsonl    -- 7.5 %
"""

import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────

SEED = 42
MIN_TEXT_LEN = 10
MAX_TEXT_LEN = 2000
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Our label convention (same as finetune.py)
LABEL_NAMES = {0: "positive", 1: "negative", 2: "neutral"}

# Original dataset label map (Russian strings -> int)
SENTIMENT_MAP_RU = {"позитив": 0, "негатив": 1, "нейтрально": 2}


# ─── HuggingFace dataset loaders ─────────────────────────────────────

def load_monohime() -> list[dict]:
    """
    MonoHime/ru_sentiment_dataset  (~211K rows)
    Original labels: 0=neutral, 1=positive, 2=negative
    Target labels:   0=positive, 1=negative, 2=neutral
    """
    log.info("Downloading MonoHime/ru_sentiment_dataset ...")
    ds = load_dataset("MonoHime/ru_sentiment_dataset", split="train")

    # Remap: original -> our convention
    remap = {
        0: 2,  # neutral -> 2
        1: 0,  # positive -> 0
        2: 1,  # negative -> 1
    }

    rows = []
    for item in ds:
        text = str(item.get("text") or "").strip()
        orig_label = item.get("label")
        if orig_label is None or orig_label not in remap:
            continue
        rows.append({"text": text, "label": remap[orig_label]})

    log.info(f"  MonoHime: {len(rows)} rows loaded")
    return rows


def load_sharonov() -> list[dict]:
    """
    DmitrySharonov/ru_sentiment_neg_pos_neutral  (~257K rows)
    Labels are string: "negative" / "positive" / "neutral"
    """
    log.info("Downloading DmitrySharonov/ru_sentiment_neg_pos_neutral ...")
    ds = load_dataset("DmitrySharonov/ru_sentiment_neg_pos_neutral", split="train")

    str_to_label = {
        "positive": 0,
        "negative": 1,
        "neutral": 2,
    }

    rows = []
    for item in ds:
        text = str(item.get("text") or "").strip()
        label_str = str(item.get("label") or "").strip().lower()
        if label_str not in str_to_label:
            continue
        rows.append({"text": text, "label": str_to_label[label_str]})

    log.info(f"  Sharonov: {len(rows)} rows loaded")
    return rows


def load_sentirueval() -> list[dict]:
    """
    mteb/SentiRuEval2016  (~6K rows)
    Inspect label format at runtime and map accordingly.
    """
    log.info("Downloading mteb/SentiRuEval2016 ...")

    rows = []
    try:
        ds = load_dataset("mteb/SentiRuEval2016", split="test")
    except Exception:
        # Some MTEB datasets have non-standard splits
        try:
            ds = load_dataset("mteb/SentiRuEval2016")
            # Try to get any available split
            split_name = list(ds.keys())[0]
            ds = ds[split_name]
            log.info(f"  Using split: {split_name}")
        except Exception as e:
            log.warning(f"  Could not load mteb/SentiRuEval2016: {e}")
            return rows

    # Inspect first few items to understand label format
    sample = ds[0] if len(ds) > 0 else {}
    log.info(f"  SentiRuEval sample keys: {list(sample.keys())}")
    log.info(f"  SentiRuEval sample: {sample}")

    # Determine text and label column names
    text_col = None
    label_col = None
    for col in ["text", "sentence", "review"]:
        if col in ds.column_names:
            text_col = col
            break
    for col in ["label", "sentiment", "labels"]:
        if col in ds.column_names:
            label_col = col
            break

    if text_col is None or label_col is None:
        log.warning(f"  Cannot find text/label columns in: {ds.column_names}")
        return rows

    # Detect label type from first non-null entry
    first_label = sample.get(label_col)
    log.info(f"  Label column: '{label_col}', sample value: {first_label} (type: {type(first_label).__name__})")

    # Build mapping based on observed labels
    if isinstance(first_label, str):
        str_map = {"positive": 0, "negative": 1, "neutral": 2}
        for item in ds:
            text = str(item.get(text_col) or "").strip()
            lbl = str(item.get(label_col) or "").strip().lower()
            if lbl in str_map:
                rows.append({"text": text, "label": str_map[lbl]})
    elif isinstance(first_label, (int, float)):
        # Collect unique labels to understand the scheme
        unique_labels = set()
        for item in ds:
            unique_labels.add(item.get(label_col))
            if len(unique_labels) > 10:
                break
        log.info(f"  Unique labels (sample): {sorted(unique_labels)}")

        # Common integer schemes:
        #   {0, 1, 2} -- need to figure out which is which
        #   {-1, 0, 1} -- negative/neutral/positive (SentiRuEval standard)
        if unique_labels <= {-1, 0, 1}:
            int_map = {1: 0, -1: 1, 0: 2}  # positive/negative/neutral
        elif unique_labels <= {0, 1, 2}:
            # Assume same as MonoHime: 0=neutral, 1=positive, 2=negative
            int_map = {0: 2, 1: 0, 2: 1}
        else:
            log.warning(f"  Unknown label scheme: {unique_labels}")
            return rows

        for item in ds:
            text = str(item.get(text_col) or "").strip()
            lbl = item.get(label_col)
            if lbl in int_map:
                rows.append({"text": text, "label": int_map[lbl]})
    elif isinstance(first_label, list):
        # MTEB datasets sometimes store labels as list of ints
        # For classification, take the first element
        log.info("  Labels are lists -- taking first element")
        unique_labels = set()
        for item in ds:
            lbl_list = item.get(label_col, [])
            if lbl_list:
                unique_labels.add(lbl_list[0])
            if len(unique_labels) > 10:
                break
        log.info(f"  Unique label[0] values (sample): {sorted(unique_labels)}")

        if unique_labels <= {-1, 0, 1}:
            int_map = {1: 0, -1: 1, 0: 2}
        elif unique_labels <= {0, 1, 2}:
            int_map = {0: 2, 1: 0, 2: 1}
        else:
            log.warning(f"  Unknown label scheme: {unique_labels}")
            return rows

        for item in ds:
            text = str(item.get(text_col) or "").strip()
            lbl_list = item.get(label_col, [])
            if lbl_list and lbl_list[0] in int_map:
                rows.append({"text": text, "label": int_map[lbl_list[0]]})
    else:
        log.warning(f"  Unrecognized label type: {type(first_label)}")

    log.info(f"  SentiRuEval: {len(rows)} rows loaded")
    return rows


# ─── Local data loaders ──────────────────────────────────────────────

def load_original_xlsx() -> list[dict]:
    """Load data/dataset.xlsx (same format as finetune.py)."""
    path = DATA_DIR / "dataset.xlsx"
    if not path.exists():
        log.info(f"  {path} not found -- skipping")
        return []

    log.info(f"Loading original dataset: {path}")
    df = pd.read_excel(path)
    df["text"] = (
        df["Заголовок"].fillna("") + " " + df["Текст"].fillna("")
    ).str.strip()
    df["sentiment_raw"] = df["Тональность"].str.strip().str.lower()
    df = df[df["sentiment_raw"].isin(SENTIMENT_MAP_RU.keys())].copy()
    df["label"] = df["sentiment_raw"].map(SENTIMENT_MAP_RU)

    rows = [{"text": r["text"], "label": int(r["label"])} for _, r in df.iterrows()]
    log.info(f"  Original xlsx: {len(rows)} rows")
    return rows


def load_augmented_jsonl() -> list[dict]:
    """Load data/augmented_merged.jsonl (sentiment field is Russian string)."""
    path = DATA_DIR / "augmented_merged.jsonl"
    if not path.exists():
        log.info(f"  {path} not found -- skipping")
        return []

    log.info(f"Loading augmented data: {path}")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sentiment = item.get("sentiment", "").strip().lower()
            text = item.get("text", "").strip()
            if sentiment in SENTIMENT_MAP_RU and text:
                rows.append({"text": text, "label": SENTIMENT_MAP_RU[sentiment]})
    log.info(f"  Augmented: {len(rows)} rows")
    return rows


# ─── Filtering & deduplication ───────────────────────────────────────

def filter_by_length(rows: list[dict]) -> list[dict]:
    """Keep only texts with MIN_TEXT_LEN < len(text) < MAX_TEXT_LEN."""
    filtered = [
        r for r in rows
        if MIN_TEXT_LEN < len(r["text"]) < MAX_TEXT_LEN
    ]
    return filtered


def deduplicate(rows: list[dict]) -> list[dict]:
    """Remove duplicate texts, keeping first occurrence."""
    seen = set()
    unique = []
    for r in rows:
        if r["text"] not in seen:
            seen.add(r["text"])
            unique.append(r)
    return unique


# ─── Statistics ──────────────────────────────────────────────────────

def print_stats(rows: list[dict], title: str = "Dataset"):
    """Print total count and per-class distribution."""
    total = len(rows)
    counts = Counter(r["label"] for r in rows)
    log.info(f"\n{'='*60}")
    log.info(f"{title}: {total} samples")
    log.info(f"{'='*60}")
    for label_id in sorted(counts.keys()):
        name = LABEL_NAMES.get(label_id, f"unknown_{label_id}")
        cnt = counts[label_id]
        pct = 100.0 * cnt / total if total > 0 else 0
        log.info(f"  {label_id} ({name:>8s}): {cnt:>8d}  ({pct:5.1f}%)")


def save_jsonl(rows: list[dict], path: Path):
    """Save list of {"text": ..., "label": ...} to JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"text": r["text"], "label": r["label"]}, ensure_ascii=False) + "\n")
    log.info(f"  Saved {len(rows)} rows -> {path}")


# ─── Main ────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    # Collect data from all sources with per-source tracking
    sources: dict[str, list[dict]] = {}

    # 1) HuggingFace datasets
    sources["MonoHime/ru_sentiment_dataset"] = load_monohime()
    sources["DmitrySharonov/ru_sentiment_neg_pos_neutral"] = load_sharonov()
    sources["mteb/SentiRuEval2016"] = load_sentirueval()

    # 2) Local datasets
    sources["original_xlsx"] = load_original_xlsx()
    sources["augmented_jsonl"] = load_augmented_jsonl()

    # Merge all into one list
    all_rows = []
    for rows in sources.values():
        all_rows.extend(rows)
    log.info(f"\nTotal raw (all sources): {len(all_rows)}")

    # Filter by text length
    all_rows = filter_by_length(all_rows)
    log.info(f"After length filter ({MIN_TEXT_LEN} < len < {MAX_TEXT_LEN}): {len(all_rows)}")

    # Deduplicate
    all_rows = deduplicate(all_rows)
    log.info(f"After deduplication: {len(all_rows)}")

    # Per-source stats (after filtering, re-count per source for reporting)
    # We track source counts before merge for reporting
    log.info(f"\n{'='*60}")
    log.info("PER-SOURCE COUNTS (before merge / dedup):")
    log.info(f"{'='*60}")
    for src_name, src_rows in sources.items():
        filtered = filter_by_length(src_rows)
        log.info(f"  {src_name}: {len(src_rows)} raw -> {len(filtered)} after length filter")

    # Overall stats
    print_stats(all_rows, "MERGED DATASET (deduplicated)")

    # Validate labels
    valid_labels = {0, 1, 2}
    invalid = [r for r in all_rows if r["label"] not in valid_labels]
    if invalid:
        log.warning(f"Dropping {len(invalid)} rows with invalid labels")
        all_rows = [r for r in all_rows if r["label"] in valid_labels]

    # Resample to target class balance: 40% neutral, 30% positive, 30% negative
    # This compensates for external datasets diluting the neutral class
    import random
    random.seed(SEED)
    by_label: dict[int, list[dict]] = {0: [], 1: [], 2: []}
    for r in all_rows:
        by_label[r["label"]].append(r)

    # Use the smallest of (pos, neg) as reference, then scale neutral up
    n_pos = len(by_label[0])
    n_neg = len(by_label[1])
    n_neu = len(by_label[2])
    log.info(f"\nBefore resampling: pos={n_pos}, neg={n_neg}, neu={n_neu}")

    # Target: neutral = 40%, pos = 30%, neg = 30%
    # Keep all neutral (it's underrepresented), downsample pos/neg
    target_neu = n_neu  # keep all neutral samples
    target_pos = int(target_neu * 0.75)  # 30/40 ratio
    target_neg = int(target_neu * 0.75)

    # Don't upsample — only downsample overrepresented classes
    target_pos = min(target_pos, n_pos)
    target_neg = min(target_neg, n_neg)

    resampled = []
    resampled.extend(by_label[2])  # all neutral
    resampled.extend(random.sample(by_label[0], target_pos))
    resampled.extend(random.sample(by_label[1], target_neg))
    random.shuffle(resampled)
    all_rows = resampled

    log.info(f"After resampling: pos={target_pos}, neg={target_neg}, neu={target_neu}")
    print_stats(all_rows, "RESAMPLED DATASET")

    # Stratified split: 85% train, 7.5% val, 7.5% test
    labels = [r["label"] for r in all_rows]

    train_rows, temp_rows, train_labels, temp_labels = train_test_split(
        all_rows, labels,
        test_size=0.15,
        random_state=SEED,
        stratify=labels,
    )
    val_rows, test_rows, _, _ = train_test_split(
        temp_rows, temp_labels,
        test_size=0.5,
        random_state=SEED,
        stratify=temp_labels,
    )

    log.info(f"\nSplit sizes:")
    log.info(f"  Train: {len(train_rows)}")
    log.info(f"  Val:   {len(val_rows)}")
    log.info(f"  Test:  {len(test_rows)}")

    print_stats(train_rows, "TRAIN")
    print_stats(val_rows, "VAL")
    print_stats(test_rows, "TEST")

    # Save
    log.info(f"\nSaving splits to {DATA_DIR}/")
    save_jsonl(train_rows, DATA_DIR / "merged_train.jsonl")
    save_jsonl(val_rows, DATA_DIR / "merged_val.jsonl")
    save_jsonl(test_rows, DATA_DIR / "merged_test.jsonl")

    elapsed = time.time() - t_start
    log.info(f"\nDone in {elapsed:.1f}s")
    log.info("Files ready for finetune_v2.py")


if __name__ == "__main__":
    main()
