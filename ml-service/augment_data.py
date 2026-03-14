"""
Data augmentation via Kimi API (Moonshot).
Generates synthetic Russian social media mentions to balance the dataset,
focusing on negative and positive classes.
"""

import json
import logging
import sys
import time
from pathlib import Path

import requests
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

KIMI_API_KEY = "sk-kimi-ulvg9YLRnu3j7p9d7g7Gfbneb1PIs8nAu29VUB7rJiT10TfiIPjkIMpp1MNTvA6T"
KIMI_URL = "https://api.moonshot.cn/v1/chat/completions"

DATA_PATH = Path("data/dataset.xlsx")
OUTPUT_PATH = Path("data/augmented.jsonl")

BATCH_SIZE = 20  # texts per API call
TARGET_PER_CLASS = 1000  # new synthetic samples per sentiment class


def call_kimi(prompt: str, max_tokens: int = 4096) -> str:
    """Call Kimi API and return the response text."""
    headers = {
        "Authorization": f"Bearer {KIMI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "moonshot-v1-8k",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.9,
        "max_tokens": max_tokens,
    }

    for attempt in range(3):
        try:
            resp = requests.post(KIMI_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            log.warning(f"Kimi API attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)

    raise RuntimeError("Kimi API failed after 3 attempts")


def load_examples(df: pd.DataFrame, sentiment: str, n: int = 10) -> list[str]:
    """Load random real examples of a given sentiment for few-shot prompting."""
    subset = df[df["Тональность"].str.strip().str.lower() == sentiment]
    texts = (subset["Заголовок"].fillna("") + " " + subset["Текст"].fillna("")).str.strip()
    texts = texts[texts.str.len() > 20]
    samples = texts.sample(min(n, len(texts)), random_state=42).tolist()
    return [t[:300] for t in samples]  # truncate long texts


def generate_batch(sentiment: str, examples: list[str], count: int = 20) -> list[dict]:
    """Generate synthetic social media mentions via Kimi."""
    sentiment_ru = {
        "позитив": "позитивной (похвала, благодарность, радость)",
        "негатив": "негативной (жалобы, проблемы, критика, гнев)",
        "нейтрально": "нейтральной (факты, новости, информация без эмоций)",
    }

    examples_text = "\n".join(f"- {ex}" for ex in examples[:5])

    prompt = f"""Сгенерируй {count} уникальных коротких сообщений из социальных сетей на русском языке.

Тематика: упоминания мобильного оператора, банка или технологической компании.
Тональность: {sentiment_ru[sentiment]}

Вот примеры реальных сообщений с такой тональностью:
{examples_text}

Правила:
1. Каждое сообщение — 1-3 предложения, как в Telegram/VK/Twitter
2. Используй разговорный стиль, допустимы сленг и эмоджи
3. Разнообразие тем: связь, тарифы, поддержка, приложение, переводы, карты
4. Каждое сообщение на отдельной строке
5. Не нумеруй сообщения
6. Не добавляй пояснений, только сами сообщения

Сгенерируй ровно {count} сообщений:"""

    response = call_kimi(prompt)

    # Parse response into individual messages
    lines = [
        line.strip().lstrip("0123456789.-) ")
        for line in response.strip().split("\n")
        if line.strip() and len(line.strip()) > 10
    ]

    results = []
    for text in lines[:count]:
        results.append({
            "text": text,
            "sentiment": sentiment,
            "source": "kimi_augmented",
        })

    return results


def main():
    log.info("Loading original dataset for examples...")
    df = pd.read_excel(DATA_PATH)
    log.info(f"Original dataset: {len(df)} rows")

    all_generated = []

    for sentiment in ["позитив", "негатив", "нейтрально"]:
        log.info(f"\n{'='*50}")
        log.info(f"Generating {TARGET_PER_CLASS} {sentiment} samples...")
        examples = load_examples(df, sentiment, n=15)
        log.info(f"Loaded {len(examples)} real examples for few-shot")

        generated_count = 0
        batch_num = 0

        while generated_count < TARGET_PER_CLASS:
            batch_num += 1
            remaining = TARGET_PER_CLASS - generated_count
            batch_count = min(BATCH_SIZE, remaining)

            log.info(f"  Batch {batch_num}: requesting {batch_count} samples...")
            try:
                batch = generate_batch(sentiment, examples, batch_count)
                all_generated.extend(batch)
                generated_count += len(batch)
                log.info(f"  Got {len(batch)} samples (total: {generated_count}/{TARGET_PER_CLASS})")
            except Exception as e:
                log.error(f"  Failed: {e}")
                time.sleep(5)
                continue

            # Rate limiting
            time.sleep(1)

    # Save to JSONL
    log.info(f"\nSaving {len(all_generated)} augmented samples to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in all_generated:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Summary
    from collections import Counter
    counts = Counter(item["sentiment"] for item in all_generated)
    log.info(f"Generated: {dict(counts)}")
    log.info(f"Total: {len(all_generated)} synthetic samples")
    log.info(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
