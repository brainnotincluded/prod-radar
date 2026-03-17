#!/usr/bin/env python3
"""
Local inference script for ProdRadar sentiment model.
Processes news from news.json with real-time output.

Usage:
    python3 run_inference.py
    python3 run_inference.py --limit 50
    python3 run_inference.py --file /path/to/news.json
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import pipeline

# ─── Config ───────────────────────────────────────────────────────────

MODEL_PATH = Path(__file__).parent / "models" / "phase1-ruroberta" / "final"
MODEL_FALLBACK = "Daniil125/prodradar-sentiment-ru"
DEFAULT_NEWS = Path.home() / "Downloads" / "news.json"

SENTIMENT_LABELS = {0: "positive", 1: "negative", 2: "neutral"}
LABEL_MAP = {
    "positive": "positive", "negative": "negative", "neutral": "neutral",
    "POSITIVE": "positive", "NEGATIVE": "negative", "NEUTRAL": "neutral",
    "LABEL_0": "positive", "LABEL_1": "negative", "LABEL_2": "neutral",
}

COLORS = {
    "positive": "\033[92m",  # green
    "negative": "\033[91m",  # red
    "neutral": "\033[90m",   # gray
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "yellow": "\033[93m",
    "cyan": "\033[96m",
}

EMOJI = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}


# ─── Sentence splitting ──────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    text = re.sub(r'(\d)\.\s*(\d)', r'\1[DOT]\2', text)
    for abbr in ["г.", "гг.", "т.д.", "т.п.", "т.е.", "др.", "пр.", "руб.", "коп.",
                  "млн.", "млрд.", "трлн.", "тыс.", "ул.", "д.", "стр.", "корп."]:
        text = text.replace(abbr, abbr.replace(".", "[DOT]"))
    parts = re.split(r'(?<=[.!?])\s+', text)
    sentences = []
    for p in parts:
        p = p.replace("[DOT]", ".").strip()
        if len(p) > 5:
            sentences.append(p)
    return sentences if sentences else [text]


# ─── Sentiment mapping ───────────────────────────────────────────────

def map_sentiment(raw_results):
    if not raw_results:
        return "neutral", 0.5
    scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for item in raw_results:
        mapped = LABEL_MAP.get(item["label"], "neutral")
        scores[mapped] = max(scores[mapped], item["score"])
    best = max(scores, key=lambda k: scores[k])
    return best, scores[best]


def aggregate_sentences(pipe, text: str):
    """Sentence-level aggregation with emotional ratio scoring."""
    sentences = split_sentences(text)

    if len(sentences) <= 1 or len(text) < 300:
        results = pipe(text, top_k=None)
        label, score = map_sentiment(results)
        return label, score, [(text, label, score)]

    # Classify all sentences
    results = pipe(sentences, top_k=None, batch_size=32)
    sent_results = []
    pos_total = neg_total = 0.0
    pos_count = neg_count = 0

    for sent, raw in zip(sentences, results):
        label, score = map_sentiment(raw)
        sent_results.append((sent, label, score))
        if label == "positive":
            pos_total += score
            pos_count += 1
        elif label == "negative":
            neg_total += score
            neg_count += 1

    n = len(sentences)
    emotional_count = pos_count + neg_count
    emotional_ratio = emotional_count / n if n > 0 else 0

    if emotional_ratio >= 0.15 and (pos_total > 0 or neg_total > 0):
        if neg_total > pos_total:
            best_label = "negative"
            best_score = neg_total / (neg_total + pos_total)
        elif pos_total > neg_total:
            best_label = "positive"
            best_score = pos_total / (neg_total + pos_total)
        else:
            best_label = "neutral"
            best_score = 0.5
    else:
        best_label = "neutral"
        best_score = 0.8

    return best_label, round(best_score, 4), sent_results


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ProdRadar Local Inference")
    parser.add_argument("--file", default=str(DEFAULT_NEWS), help="Path to news.json")
    parser.add_argument("--limit", type=int, default=30, help="Number of news to process")
    parser.add_argument("--sentences", action="store_true", help="Show per-sentence breakdown")
    args = parser.parse_args()

    c = COLORS

    # Load model
    model_path = str(MODEL_PATH) if MODEL_PATH.exists() else MODEL_FALLBACK
    print(f"\n{c['bold']}🧠 Loading model: {model_path}{c['reset']}")
    print(f"{c['dim']}   Device: {'mps' if torch.backends.mps.is_available() else 'cpu'}{c['reset']}")

    t0 = time.time()
    device = -1  # CPU (MPS has issues with some ops)
    pipe = pipeline("text-classification", model=model_path, device=device,
                    truncation=True, max_length=512, top_k=None)
    print(f"{c['dim']}   Loaded in {time.time() - t0:.1f}s{c['reset']}\n")

    # Load news
    print(f"{c['bold']}📰 Loading news from {args.file}{c['reset']}")
    with open(args.file, encoding="utf-8") as f:
        news = json.load(f)
    print(f"{c['dim']}   Total: {len(news)} articles, processing {args.limit}{c['reset']}\n")

    print(f"{c['bold']}{'─' * 90}{c['reset']}")
    print(f"{c['bold']}{'#':>3}  {'Sentiment':>10}  {'Score':>5}  {'Len':>5}  {'Sents':>5}  Title{c['reset']}")
    print(f"{c['bold']}{'─' * 90}{c['reset']}")

    counts = {"positive": 0, "negative": 0, "neutral": 0}
    total_time = 0

    for i, article in enumerate(news[:args.limit], 1):
        text = article.get("text", article.get("title", ""))
        title = article.get("title", "")[:60]

        if not text or len(text) < 10:
            continue

        t1 = time.time()
        label, score, sent_details = aggregate_sentences(pipe, text)
        elapsed = time.time() - t1
        total_time += elapsed

        counts[label] = counts.get(label, 0) + 1
        emoji = EMOJI.get(label, "?")
        color = c.get(label, "")

        n_sents = len(sent_details)
        text_len = len(text)

        # Real-time print
        print(f"{i:>3}  {emoji} {color}{label:>8}{c['reset']}  {score:.2f}  {text_len:>5}  {n_sents:>5}  {title}")

        # Show sentence breakdown if requested
        if args.sentences and n_sents > 1:
            for sent_text, sent_label, sent_score in sent_details:
                sc = c.get(sent_label, "")
                dot = EMOJI.get(sent_label, "·")
                print(f"     {dot} {sc}{sent_label:>8}{c['reset']} {sent_score:.2f}  {c['dim']}{sent_text[:75]}{c['reset']}")
            print()

        sys.stdout.flush()

    # Summary
    total = sum(counts.values())
    print(f"\n{c['bold']}{'─' * 90}{c['reset']}")
    print(f"\n{c['bold']}📊 ИТОГО ({total} новостей, {total_time:.1f}s, {total_time/max(total,1):.2f}s/article):{c['reset']}\n")

    for label in ["positive", "negative", "neutral"]:
        cnt = counts.get(label, 0)
        pct = 100 * cnt / total if total > 0 else 0
        bar = "█" * int(pct / 2)
        color = c.get(label, "")
        emoji = EMOJI.get(label, "")
        print(f"  {emoji} {color}{label:>8}{c['reset']}: {cnt:>3} ({pct:5.1f}%)  {color}{bar}{c['reset']}")

    print()


if __name__ == "__main__":
    main()
