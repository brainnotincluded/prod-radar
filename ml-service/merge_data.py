"""Merge all generated JSON data files into a single JSONL for training."""
import json
from pathlib import Path

SENTIMENT_NORMALIZE = {
    "positive": "позитив",
    "negative": "негатив",
    "neutral": "нейтрально",
}

all_items = []
sources = []

# Collect from gen/ directory
gen_dir = Path("gen")
if gen_dir.exists():
    for f in sorted(gen_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            for item in data:
                sentiment = item.get("sentiment", "neutral")
                text = item.get("text", "").strip()
                if text and len(text) > 3:
                    all_items.append({
                        "text": text,
                        "sentiment": SENTIMENT_NORMALIZE.get(sentiment, sentiment),
                        "source_file": f.name,
                    })
            sources.append(f"{f.name}: {len(data)} items")
        except Exception as e:
            print(f"Error reading {f}: {e}")

# Collect from data_*.json files
for f in sorted(Path(".").glob("data_*.json")):
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        for item in data:
            sentiment = item.get("sentiment", "neutral")
            text = item.get("text", "").strip()
            if text and len(text) > 3:
                all_items.append({
                    "text": text,
                    "sentiment": SENTIMENT_NORMALIZE.get(sentiment, sentiment),
                    "source_file": f.name,
                })
        sources.append(f"{f.name}: {len(data)} items")
    except Exception as e:
        print(f"Error reading {f}: {e}")

# Deduplicate by text
seen = set()
unique = []
for item in all_items:
    if item["text"] not in seen:
        seen.add(item["text"])
        unique.append(item)

# Save
output = Path("data/augmented_merged.jsonl")
output.parent.mkdir(exist_ok=True)
with open(output, "w", encoding="utf-8") as f:
    for item in unique:
        f.write(json.dumps({"text": item["text"], "sentiment": item["sentiment"]}, ensure_ascii=False) + "\n")

# Stats
from collections import Counter
counts = Counter(item["sentiment"] for item in unique)
print(f"\nSource files: {len(sources)}")
for s in sources:
    print(f"  {s}")
print(f"\nTotal raw: {len(all_items)}")
print(f"Unique (deduplicated): {len(unique)}")
print(f"Sentiment distribution:")
for k, v in sorted(counts.items()):
    print(f"  {k}: {v}")
print(f"\nSaved to {output}")
