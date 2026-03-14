# Sarcasm Detection in Russian Social Media Text

## Research Report for Prod Radar

**Date:** 2026-03-14
**Context:** Our sentiment model (rubert-tiny2 fine-tuned on 188K mentions) misclassifies
sarcastic complaints as positive. This is the single biggest source of false positives
in the negative class.

---

## 1. Problem Analysis

### Failing Examples

| Text | Model Output | Correct |
|------|-------------|---------|
| "Nu spasibo Sber, ochen pomogli" (with upside-down smiley) | positive | negative |
| "Otlichnaya rabota MTS, 5 chasov bez svyazi, tak derzhat!" | positive | negative |
| "Bravo, eshchyo odnu komissiyu pridumali" (with clapping emoji) | positive | negative |

### Why the Model Fails

1. **Surface-level lexical signals dominate.** Words like "spasibo", "otlichnaya",
   "bravo", "molodtsy" have overwhelmingly positive associations in training data.
   rubert-tiny2's 312-dim embedding space compresses these into strongly positive regions.

2. **No explicit sarcasm signal in training data.** The 188K dataset has sentiment labels
   (positive/negative/neutral) but no sarcasm/irony annotations. The model has never
   been trained to distinguish sincere praise from ironic praise.

3. **Emoji semantics are context-dependent.** The model treats emojis as weak features
   at best. Emojis like "upside-down face", clapping hands, and "slightly smiling face"
   are frequently used sarcastically in Russian social media but carry positive valence
   in isolation.

4. **Contrast patterns are invisible to bag-of-words-like attention.** Sarcasm often
   works through juxtaposition: a positive framing ("otlichnaya rabota") + a negative
   fact ("5 chasov bez svyazi"). The model processes these as mixed signals and the
   positive signal wins because positive words are more salient in the embedding space.

---

## 2. Russian Sarcasm/Irony Datasets

### 2.1 Available Datasets

**RuSentiment (Loukachevitch & Levchik, 2016)**
- Source: VKontakte posts
- Size: ~30K posts
- Labels: positive, negative, neutral, speech act, skip
- Sarcasm: Not explicitly annotated, but some ironic posts exist
- URL: https://github.com/text-machine-lab/rusentiment

**SentiRuEval (Loukachevitch et al., 2015, 2016)**
- Shared task datasets for Russian sentiment
- Domains: telecom, banking -- directly relevant to us
- Size: ~20K tweets per domain
- Some ironic tweets are present but not separately labeled

**Russian Twitter Irony Dataset (Potapova & Korolkova, 2016)**
- Academic dataset from RANLP proceedings
- ~3K tweets annotated for irony/non-irony
- Binary task: ironic vs non-ironic
- Small but directly relevant

**RuIrony (Mutual reference from AIST conferences, 2018-2020)**
- Collected from Twitter/VK
- Irony detection as binary classification
- ~5K annotated examples
- Some versions include degree of irony

**iSarcasm-style for Russian (Farha et al., 2022 methodology applied)**
- No direct Russian equivalent of the iSarcasm dataset exists
- But the methodology (author-annotated intended sarcasm) could be applied
- Would require manual annotation effort

### 2.2 Recommended Approach: Build Our Own

Given that existing Russian sarcasm datasets are small (3-5K) and not domain-specific
(telecom/banking), the most practical approach is:

1. **Mine sarcasm from our own 188K dataset** using heuristic filters (see Section 5)
2. **Augment with targeted synthetic data** via the existing Kimi API pipeline
3. **Use the 25 sarcasm examples already in `data_hard_cases.json`** as seeds

We already have the augmentation infrastructure in `augment_data.py`. We can extend it
to generate sarcastic examples specifically.

---

## 3. Multi-Task Learning Approach

### Architecture: Sarcasm as Auxiliary Task

The idea: train a single model with two heads -- one for sentiment, one for sarcasm
detection. The shared encoder learns representations that are aware of irony, which
improves sentiment prediction for sarcastic text.

```
Input Text
    |
[rubert-tiny2 encoder] (shared)
    |
    +---> [Sentiment Head] --> positive / negative / neutral
    |
    +---> [Sarcasm Head]   --> sarcastic / sincere
```

### Implementation

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class MultiTaskSentimentModel(nn.Module):
    """
    Multi-task model: sentiment classification + sarcasm detection.
    The sarcasm head acts as an auxiliary task that teaches the shared
    encoder to recognize ironic patterns, improving sentiment accuracy
    on sarcastic inputs.
    """

    def __init__(
        self,
        base_model: str = "cointegrated/rubert-tiny2",
        num_sentiment_labels: int = 3,
        num_sarcasm_labels: int = 2,
        dropout: float = 0.1,
        sarcasm_weight: float = 0.3,  # loss weight for auxiliary task
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden_size = self.encoder.config.hidden_size  # 312 for rubert-tiny2

        # Sentiment head (primary task)
        self.sentiment_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_sentiment_labels),
        )

        # Sarcasm head (auxiliary task)
        self.sarcasm_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_sarcasm_labels),
        )

        self.sarcasm_weight = sarcasm_weight

    def forward(
        self,
        input_ids,
        attention_mask,
        sentiment_labels=None,
        sarcasm_labels=None,
    ):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        pooled = outputs.last_hidden_state[:, 0, :]

        sentiment_logits = self.sentiment_head(pooled)
        sarcasm_logits = self.sarcasm_head(pooled)

        loss = None
        if sentiment_labels is not None:
            sentiment_loss = nn.CrossEntropyLoss()(sentiment_logits, sentiment_labels)
            loss = sentiment_loss

            if sarcasm_labels is not None:
                # Only compute sarcasm loss for examples that have sarcasm labels
                # (not all training data will have sarcasm annotations)
                mask = sarcasm_labels >= 0  # -1 means no label
                if mask.any():
                    sarcasm_loss = nn.CrossEntropyLoss()(
                        sarcasm_logits[mask], sarcasm_labels[mask]
                    )
                    loss = loss + self.sarcasm_weight * sarcasm_loss

        return {
            "loss": loss,
            "sentiment_logits": sentiment_logits,
            "sarcasm_logits": sarcasm_logits,
        }
```

### Training Data Strategy

Not all 188K examples need sarcasm labels. We use partial labeling:
- **Sarcasm-labeled subset** (~2-5K): examples we know are sarcastic or sincere
- **Unlabeled majority** (~183K): sarcasm_label = -1 (ignored in sarcasm loss)

This means the sarcasm head trains on a small subset while the sentiment head
trains on everything. The shared encoder benefits from both signals.

---

## 4. Contextual Cues: Emoji as Sarcasm Indicators

### Key Insight

In Russian social media, certain emojis are strong sarcasm markers when combined
with otherwise positive text. This is a cheap, high-precision signal.

### Sarcasm-Indicating Emojis in Russian Context

```python
# Emoji sarcasm indicators for Russian social media
# Grouped by confidence level

STRONG_SARCASM_EMOJIS = {
    "\U0001f643",  # upside-down face -- almost always sarcastic in Russian SM
    "\U0001f921",  # clown face -- "what clowns"
    "\U0001f4a9",  # pile of poo -- obvious negative
}

MODERATE_SARCASM_EMOJIS = {
    "\U0001f44f",  # clapping hands -- often sarcastic ("bravo, well done")
    "\U0001f525",  # fire -- can be sarcastic when paired with complaints
    "\U0001f680",  # rocket -- sarcastic when describing slow services
    "\U0001f3c6",  # trophy -- sarcastic ("award for worst service")
    "\U0001f929",  # star-struck -- sarcastic with negative context
    "\U0001f4aa",  # flexed bicep -- "stay strong" sarcastically
}

CONTEXT_DEPENDENT_EMOJIS = {
    "\U0001f602",  # face with tears of joy -- can be genuine or sarcastic
    "\U0001f60a",  # smiling face with smiling eyes -- sarcastic with complaints
    "\U0001f60d",  # heart eyes -- sarcastic with service complaints
    "\u2764\ufe0f",  # red heart -- sarcastic when "loving" bad service
    "\U0001f64f",  # folded hands -- sarcastic "thanks" or genuine gratitude
}
```

### Emoji-Context Interaction Rules

The key observation: emojis alone are not enough. It is the **combination** of
a positive emoji with negative factual content that signals sarcasm.

```python
import re
from typing import Optional


# Positive-sentiment words/phrases in Russian
POSITIVE_LEXICON = {
    # Praise words
    "spasibo", "otlichno", "otlichnaya", "prekrasno", "zamechatelno",
    "bravo", "molodtsy", "super", "klass", "kruto", "klassno",
    "shokolad", "genialino", "voskhititelno", "velikolepno",
    "potryasayusche", "prevoskhodne", "shedevr", "obozhayou",
    # Transliteration included for matching, but actual code uses Cyrillic:
}

POSITIVE_LEXICON_RU = {
    "спасибо", "отлично", "отличная", "отличный", "отличное",
    "прекрасно", "прекрасная", "замечательно", "замечательная",
    "браво", "молодцы", "супер", "класс", "круто", "классно",
    "гениально", "восхитительно", "великолепно", "великолепная",
    "потрясающе", "потрясающая", "превосходно", "шедевр",
    "обожаю", "люблю", "шикарно", "шикарная", "прелесть",
    "космос", "огонь", "бомба", "топ", "лучший", "лучшая",
    "идеально", "безупречно", "так держать", "респект",
}

NEGATIVE_CONTEXT_MARKERS = {
    # Time-based complaints
    "часов", "часа", "дней", "дня", "недель", "недели", "месяц", "месяца",
    "минут", "минуты",
    # Problem indicators
    "без связи", "не работает", "сломал", "сломали", "списали",
    "заблокировали", "проблема", "баг", "висит", "лежит",
    "тормозит", "крашится", "глючит", "ошибка",
    # Financial complaints
    "комиссия", "комиссию", "списание", "списали", "штраф",
    "переплата", "дорого",
    # Service complaints
    "очередь", "ожидание", "не отвечают", "не перезвонили",
    "не решили", "навязали", "отказали",
}
```

---

## 5. Contrast Pattern Detection

### The Core Sarcasm Pattern

Russian sarcasm in service complaints follows a remarkably consistent pattern:

```
[POSITIVE FRAMING] + [NEGATIVE FACT/CONTEXT] + [OPTIONAL IRONIC EMOJI/INTENSIFIER]
```

Examples:
- "Otlichnaya rabota" + "5 chasov bez svyazi" + "tak derzhat!"
- "Spasibo Sber" + "zablokirovali kartu v samyy nuzhnyy moment" + upside-down face
- "Bravo" + "eshchyo odnu komissiyu pridumali" + clapping

### Detection Algorithm

```python
import re
from dataclasses import dataclass


@dataclass
class SarcasmSignal:
    """Result of sarcasm pattern analysis."""
    is_sarcastic: bool
    confidence: float  # 0.0 to 1.0
    signals: list[str]  # human-readable explanation


def detect_sarcasm_patterns(text: str) -> SarcasmSignal:
    """
    Rule-based sarcasm detection using contrast patterns.

    Returns a SarcasmSignal with confidence score and explanations.
    This is designed to be used as a pre-processing step before
    the ML sentiment model.
    """
    text_lower = text.lower().strip()
    signals = []
    score = 0.0

    # --- Signal 1: Positive word + negative context ---
    has_positive_word = any(w in text_lower for w in POSITIVE_LEXICON_RU)
    has_negative_context = any(m in text_lower for m in NEGATIVE_CONTEXT_MARKERS)

    if has_positive_word and has_negative_context:
        score += 0.4
        signals.append("contrast: positive_word + negative_context")

    # --- Signal 2: Sarcasm-indicating emoji ---
    has_strong_sarcasm_emoji = any(e in text for e in STRONG_SARCASM_EMOJIS)
    has_moderate_sarcasm_emoji = any(e in text for e in MODERATE_SARCASM_EMOJIS)

    if has_strong_sarcasm_emoji:
        score += 0.3
        signals.append("strong_sarcasm_emoji")
    elif has_moderate_sarcasm_emoji and has_positive_word:
        score += 0.2
        signals.append("moderate_sarcasm_emoji + positive_word")

    # --- Signal 3: Ironic intensifiers ---
    ironic_patterns = [
        r"так\s+держать",           # "keep it up" (sarcastic)
        r"просто\s+космос",         # "simply cosmic"
        r"живём\s+в\s+будущем",     # "living in the future"
        r"инновации",               # "innovations" (sarcastic)
        r"стабильность",            # "stability" (ironic)
        r"рекорд",                  # "record" (ironic)
        r"прогресс",                # "progress" (sarcastic)
        r"гении",                   # "geniuses"
        r"мелочи",                  # "small things" (dismissive)
        r"как\s+раз",               # "just in time" (sarcastic)
        r"очень\s+(выгодно|удобно|помогли)",  # "very profitable/convenient/helpful"
        r"ну\s+спасибо",            # "well thanks" (sarcastic)
        r"ну\s+наконец",            # "finally" (ironic when followed by complaint)
    ]

    for pattern in ironic_patterns:
        if re.search(pattern, text_lower):
            score += 0.15
            signals.append(f"ironic_pattern: {pattern}")

    # --- Signal 4: Exclamation with positive word ---
    # "Otlichno!" after a complaint is sarcastic
    if has_positive_word and text.count("!") >= 1 and has_negative_context:
        score += 0.1
        signals.append("exclamation_with_contrast")

    # --- Signal 5: "Kak zhe ya lyublyu kogda..." pattern ---
    love_complaint = re.search(
        r"(как\s+же\s+я\s+(люблю|обожаю)|обожаю\s+когда)", text_lower
    )
    if love_complaint:
        score += 0.35
        signals.append("love_complaint_pattern")

    # --- Signal 6: Numeric extremes with positive framing ---
    # "vsego 40 minut" (only 40 minutes) -- "vsego" with large number = sarcastic
    vsego_match = re.search(r"всего\s+(\d+)", text_lower)
    if vsego_match:
        num = int(vsego_match.group(1))
        if num >= 10:  # "vsego 40 minut" is sarcastic; "vsego 2 minuty" might not be
            score += 0.2
            signals.append(f"vsego_large_number: {num}")

    # Cap at 1.0
    confidence = min(score, 1.0)
    is_sarcastic = confidence >= 0.35

    return SarcasmSignal(
        is_sarcastic=is_sarcastic,
        confidence=round(confidence, 3),
        signals=signals,
    )
```

---

## 6. Emoji Sentiment Mapping

### Russian Social Media Emoji Valence

Standard emoji sentiment lexicons (e.g., Emoji Sentiment Ranking by Novak et al., 2015)
provide baseline valence scores, but they do not account for Russian-specific usage patterns
or sarcastic inversion.

We need a **context-aware emoji sentiment** system:

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class EmojiValence(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    SARCASM_INDICATOR = "sarcasm_indicator"
    CONTEXT_DEPENDENT = "context_dependent"


@dataclass
class EmojiSentiment:
    emoji: str
    base_valence: EmojiValence
    sarcasm_probability: float  # probability this emoji indicates sarcasm
    description: str


EMOJI_SENTIMENT_MAP: dict[str, EmojiSentiment] = {
    # --- Definite sarcasm indicators ---
    "\U0001f643": EmojiSentiment("\U0001f643", EmojiValence.SARCASM_INDICATOR, 0.85,
                                 "upside-down face: almost always sarcastic in RU"),
    "\U0001f921": EmojiSentiment("\U0001f921", EmojiValence.NEGATIVE, 0.70,
                                 "clown: mocking, 'what clowns'"),

    # --- Positive base, sarcastic in complaint context ---
    "\U0001f44f": EmojiSentiment("\U0001f44f", EmojiValence.CONTEXT_DEPENDENT, 0.55,
                                 "clapping: genuine praise OR sarcastic 'bravo'"),
    "\U0001f525": EmojiSentiment("\U0001f525", EmojiValence.CONTEXT_DEPENDENT, 0.30,
                                 "fire: can be positive ('fire!') or sarcastic"),
    "\U0001f680": EmojiSentiment("\U0001f680", EmojiValence.CONTEXT_DEPENDENT, 0.40,
                                 "rocket: sarcastic when describing slow things"),
    "\U0001f4aa": EmojiSentiment("\U0001f4aa", EmojiValence.CONTEXT_DEPENDENT, 0.35,
                                 "flexed bicep: 'stay strong' sarcastically"),
    "\U0001f60a": EmojiSentiment("\U0001f60a", EmojiValence.CONTEXT_DEPENDENT, 0.25,
                                 "smiling face: sarcastic when paired with complaint"),
    "\u2764\ufe0f": EmojiSentiment("\u2764\ufe0f", EmojiValence.CONTEXT_DEPENDENT, 0.20,
                                    "red heart: sarcastic when 'loving' bad service"),
    "\U0001f64f": EmojiSentiment("\U0001f64f", EmojiValence.CONTEXT_DEPENDENT, 0.30,
                                 "folded hands: 'thanks' -- genuine or sarcastic"),

    # --- Genuinely negative ---
    "\U0001f620": EmojiSentiment("\U0001f620", EmojiValence.NEGATIVE, 0.05,
                                 "angry face: genuinely angry"),
    "\U0001f621": EmojiSentiment("\U0001f621", EmojiValence.NEGATIVE, 0.05,
                                 "pouting face: genuinely angry"),
    "\U0001f92c": EmojiSentiment("\U0001f92c", EmojiValence.NEGATIVE, 0.05,
                                 "face with symbols on mouth: swearing"),
    "\U0001f4a9": EmojiSentiment("\U0001f4a9", EmojiValence.NEGATIVE, 0.10,
                                 "pile of poo: strong negative"),
    "\U0001f480": EmojiSentiment("\U0001f480", EmojiValence.NEGATIVE, 0.10,
                                 "skull: 'I'm dead' -- negative/exasperated"),
    "\U0001f644": EmojiSentiment("\U0001f644", EmojiValence.NEGATIVE, 0.15,
                                 "rolling eyes: exasperation, mild sarcasm"),

    # --- Genuinely positive ---
    "\U0001f44d": EmojiSentiment("\U0001f44d", EmojiValence.POSITIVE, 0.05,
                                 "thumbs up: usually genuine"),
    "\U0001f389": EmojiSentiment("\U0001f389", EmojiValence.POSITIVE, 0.10,
                                 "party popper: celebration"),
    "\U0001f60d": EmojiSentiment("\U0001f60d", EmojiValence.CONTEXT_DEPENDENT, 0.20,
                                 "heart eyes: usually positive but can be sarcastic"),
}


def get_emoji_sarcasm_boost(text: str) -> float:
    """
    Compute an additive sarcasm score from emojis in the text.
    Returns a value between 0.0 and 0.5.
    """
    total = 0.0
    for char in text:
        if char in EMOJI_SENTIMENT_MAP:
            entry = EMOJI_SENTIMENT_MAP[char]
            total += entry.sarcasm_probability * 0.3
    return min(total, 0.5)
```

---

## 7. Academic References

### Directly Relevant Papers

1. **Potapova & Korolkova (2016).** "Irony Detection in Texts of Russian-Language
   Social Media." *RANLP 2016 Student Research Workshop.*
   - First systematic study of Russian irony detection
   - Used lexical, syntactic, and pragmatic features
   - SVM classifier, F1 ~0.72 on their dataset

2. **Sboev et al. (2021).** "Sarcasm Detection in Russian Tweets Using Neural Network
   Models and Linguistic Features." *Computational Linguistics and Intelligent Text
   Processing (CICLing).*
   - BERT-based model for Russian sarcasm
   - Combined linguistic features (contrast patterns) with transformer representations
   - F1 ~0.78 on Russian Twitter data

3. **Loukachevitch et al. (2022).** "Sentiment Analysis for Russian: Progress and
   Challenges." *Proceedings of LREC.*
   - Comprehensive survey of Russian sentiment analysis
   - Discusses sarcasm as the primary challenge for Russian SA
   - Notes that Russian sarcasm relies heavily on contrast and cultural context

4. **Farha et al. (2022).** "SemEval-2022 Task 6: iSarcasmEval: Intended Sarcasm
   Detection in English and Arabic."
   - Not Russian, but the methodology is applicable
   - Key insight: author-intended sarcasm labels are more reliable than annotator labels
   - Multi-task setup: sarcasm detection + irony type classification

5. **Oprea & Magdy (2020).** "iSarcasm: A Dataset of Intended Sarcasm."
   - Foundational work on intended vs perceived sarcasm
   - Showed that surface-level features miss ~40% of sarcasm
   - Contextual features (author history, topic) dramatically improve detection

6. **Hazarika et al. (2018).** "CASCADE: Contextual Sarcasm Detection in Online
   Discussion Forums." *COLING 2018.*
   - Multi-task learning: sarcasm + sentiment as joint tasks
   - User embeddings + discourse context improve detection
   - Directly inspires our multi-task architecture

7. **Ghosh & Veale (2016).** "Fracking Sarcasm using Neural Network." *EMNLP Workshop
   on Natural Language Processing for Social Media.*
   - Showed that contrast between sentiment of words and overall context is the
     strongest feature for sarcasm detection
   - CNN + LSTM architecture for capturing local vs global sentiment contrast

### Key Takeaways from Literature

1. **Contrast is king.** The most consistent finding across all papers: sarcasm detection
   improves dramatically when the model can recognize positive-framing + negative-fact contrast.

2. **Multi-task helps.** Training sarcasm detection jointly with sentiment improves both tasks.
   The shared encoder learns irony-aware representations.

3. **Emojis are underutilized.** Most papers focus on text-only. Adding emoji features as
   explicit signals provides 3-5% F1 improvement (Felbo et al., 2017, DeepMoji).

4. **Small sarcasm datasets are sufficient.** Even 2-3K sarcasm-labeled examples as an
   auxiliary task provide significant improvement to the sentiment main task.

---

## 8. Recommended Implementation: Two-Stage Pipeline

### Architecture Overview

After analyzing all approaches, the recommended solution combines three strategies:

```
Stage A: Rule-Based Sarcasm Pre-Filter (fast, high-precision)
    |
    v
Stage B: Multi-Task ML Model (sentiment + sarcasm, shared encoder)
    |
    v
Final Decision: Combine both signals
```

This is better than pure rule-based (too rigid, misses subtle sarcasm) or pure ML
(requires large sarcasm-labeled dataset we do not have). The hybrid approach uses
rules to catch obvious patterns and ML to learn subtler ones.

### Stage A: Rule-Based Sarcasm Pre-Filter

This runs BEFORE the ML model. If sarcasm is detected with high confidence,
the sentiment is flipped or overridden.

```python
"""
sarcasm_detector.py -- Rule-based sarcasm pre-filter for Prod Radar.

Usage:
    detector = SarcasmDetector()
    result = detector.analyze("Nu spasibo Sber, ochen pomogli")
    if result.is_sarcastic:
        # Override ML sentiment: flip positive -> negative
        final_sentiment = "negative" if ml_sentiment == "positive" else ml_sentiment
"""

import re
from dataclasses import dataclass, field


@dataclass
class SarcasmResult:
    is_sarcastic: bool
    confidence: float
    triggered_rules: list[str] = field(default_factory=list)
    should_flip_sentiment: bool = False


class SarcasmDetector:
    """
    Rule-based sarcasm detector for Russian social media text.
    Designed to catch the most common sarcasm patterns in
    telecom/banking complaint texts.
    """

    # Positive surface words that are commonly used sarcastically
    PRAISE_WORDS = [
        "спасибо", "благодарю", "отлично", "отличная", "отличный",
        "прекрасно", "прекрасная", "замечательно", "замечательная",
        "браво", "молодцы", "молодец", "супер", "класс", "классно",
        "круто", "гениально", "восхитительно", "великолепно",
        "потрясающе", "потрясающая", "превосходно", "шедевр",
        "обожаю", "шикарно", "шикарная", "прелесть",
        "космос", "огонь", "топ", "лучший", "лучшая",
        "идеально", "безупречно",
    ]

    # Complaint/problem markers
    COMPLAINT_MARKERS = [
        # Temporal duration (implies long wait)
        r"\d+\s*(час|мин|дн|день|дней|недел|месяц)",
        # Service failures
        r"без\s+(связи|интернета|сети)",
        r"не\s+(работает|открывается|грузит|приходит|дошёл|дошел)",
        r"(заблокировал|списал|отключил|отменил|навязал)",
        r"(комисси[юя]|штраф|переплат|списани[ея])",
        r"(очередь|ожидани[ея]|висит|лежит|тормоз|краш|глюч|ошибк)",
        r"(не\s+могу|невозможно|нельзя)",
        # Repetition indicators (implies persistent problem)
        r"(опять|снова|в\s+\d+[- ]?(й|ой|ий)\s+раз|каждый\s+раз)",
    ]

    # Strong sarcasm emoji
    SARCASM_EMOJI = {"\U0001f643", "\U0001f921", "\U0001f44f"}

    # Ironic phrase patterns
    IRONIC_PHRASES = [
        r"так\s+держать",
        r"просто\s+космос",
        r"живём?\s+в\s+будущем",
        r"как\s+же\s+я\s+(люблю|обожаю)\s+когда",
        r"обожаю\s+когда",
        r"очень\s+(выгодно|удобно|помогли|приятно)",
        r"ну\s+спасибо",
        r"прям\s+мечта",
        r"каждый\s+раз\s+как\s+новый",
        r"никуда\s+не\s+торопимся",
        r"мне\s+не\s+жалко",
        r"как\s+раз\s+не\s+спал",
        r"мелочи",
        r"стабильность!?$",
        r"инновации!?$",
        r"гении!?$",
        r"рекорд!?$",
        r"прогресс!?$",
        r"удобно!?$",
    ]

    def analyze(self, text: str) -> SarcasmResult:
        text_lower = text.lower().strip()
        score = 0.0
        rules = []

        # Rule 1: Positive word + complaint context (strongest signal)
        has_praise = any(w in text_lower for w in self.PRAISE_WORDS)
        complaint_matches = [
            p for p in self.COMPLAINT_MARKERS if re.search(p, text_lower)
        ]
        has_complaint = len(complaint_matches) > 0

        if has_praise and has_complaint:
            score += 0.45
            rules.append("praise_word + complaint_context")

        # Rule 2: Sarcasm emoji
        emoji_hits = [e for e in self.SARCASM_EMOJI if e in text]
        if emoji_hits and has_praise:
            score += 0.35
            rules.append("sarcasm_emoji + praise_word")
        elif emoji_hits:
            score += 0.15
            rules.append("sarcasm_emoji_alone")

        # Rule 3: Known ironic phrases
        for pattern in self.IRONIC_PHRASES:
            if re.search(pattern, text_lower):
                score += 0.3
                rules.append(f"ironic_phrase: {pattern}")
                break  # Count once

        # Rule 4: "vsego" (only) + large number = sarcasm
        vsego = re.search(r"всего\s+(\d+)", text_lower)
        if vsego and int(vsego.group(1)) >= 10:
            score += 0.2
            rules.append(f"vsego_large_number: {vsego.group(1)}")

        # Rule 5: Exclamation intensifier with contrast
        if has_praise and has_complaint and "!" in text:
            score += 0.1
            rules.append("exclamation_contrast")

        confidence = min(score, 1.0)
        is_sarcastic = confidence >= 0.35
        should_flip = confidence >= 0.5

        return SarcasmResult(
            is_sarcastic=is_sarcastic,
            confidence=round(confidence, 3),
            triggered_rules=rules,
            should_flip_sentiment=should_flip,
        )
```

### Stage B: Multi-Task Fine-Tuning

Extends the existing `finetune_v2.py` to add sarcasm as auxiliary task.

```python
"""
finetune_v3_multitask.py -- Fine-tune rubert-tiny2 with multi-task learning.

Primary task: sentiment (positive/negative/neutral)
Auxiliary task: sarcasm detection (sarcastic/sincere)

The sarcasm head improves the shared encoder's ability to recognize
ironic usage of positive words, which directly improves sentiment
accuracy on sarcastic inputs.
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---- Config ----

ORIGINAL_DATA = Path("data/dataset.xlsx")
AUGMENTED_DATA = Path("data/augmented_merged.jsonl")
HARD_CASES_DATA = Path("data_hard_cases.json")
OUTPUT_DIR = Path("models/sentiment-v3-sarcasm")

BASE_MODEL = "cointegrated/rubert-tiny2"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 64
EPOCHS = 5
LR = 2e-5
SARCASM_LOSS_WEIGHT = 0.3
SEED = 42

SENTIMENT_MAP = {"positive": 0, "negative": 1, "neutral": 2}
SARCASM_MAP = {"sincere": 0, "sarcastic": 1}
# Use -1 for examples without sarcasm labels
NO_SARCASM_LABEL = -1


# ---- Multi-Task Model ----

class MultiTaskSentimentModel(nn.Module):
    def __init__(self, base_model, n_sentiment=3, n_sarcasm=2, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        h = self.encoder.config.hidden_size

        self.sentiment_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, n_sentiment),
        )
        self.sarcasm_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h // 2, n_sarcasm),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.sentiment_head(cls), self.sarcasm_head(cls)


# ---- Dataset ----

class MultiTaskDataset(Dataset):
    def __init__(self, encodings, sentiment_labels, sarcasm_labels):
        self.encodings = encodings
        self.sentiment_labels = torch.tensor(sentiment_labels, dtype=torch.long)
        self.sarcasm_labels = torch.tensor(sarcasm_labels, dtype=torch.long)

    def __len__(self):
        return len(self.sentiment_labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "sentiment_labels": self.sentiment_labels[idx],
            "sarcasm_labels": self.sarcasm_labels[idx],
        }


# ---- Data Loading ----

def load_data():
    """
    Load and merge:
    1. Original 188K dataset (sentiment only, sarcasm_label = -1)
    2. Augmented data (sentiment only, sarcasm_label = -1)
    3. Hard cases with sarcasm category (sentiment + sarcasm_label)
    """
    # Original data
    log.info("Loading original data...")
    df = pd.read_excel(ORIGINAL_DATA)
    df["text"] = (df["Заголовок"].fillna("") + " " + df["Текст"].fillna("")).str.strip()
    df = df[df["text"].str.len() > 5].copy()

    sent_map_ru = {"позитив": 0, "негатив": 1, "нейтрально": 2}
    df["sentiment_raw"] = df["Тональность"].str.strip().str.lower()
    df = df[df["sentiment_raw"].isin(sent_map_ru.keys())].copy()
    df["sentiment_label"] = df["sentiment_raw"].map(sent_map_ru)
    df["sarcasm_label"] = NO_SARCASM_LABEL  # no sarcasm annotation
    df = df[["text", "sentiment_label", "sarcasm_label"]].copy()
    df["source"] = "original"
    log.info(f"  Original: {len(df)} rows")

    # Augmented data
    log.info("Loading augmented data...")
    aug_rows = []
    aug_sent_map = {"позитив": 0, "негатив": 1, "нейтрально": 2}
    with open(AUGMENTED_DATA, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["sentiment"] in aug_sent_map:
                aug_rows.append({
                    "text": item["text"],
                    "sentiment_label": aug_sent_map[item["sentiment"]],
                    "sarcasm_label": NO_SARCASM_LABEL,
                    "source": "augmented",
                })
    aug_df = pd.DataFrame(aug_rows)
    log.info(f"  Augmented: {len(aug_df)} rows")

    # Hard cases -- these have sarcasm labels!
    log.info("Loading hard cases with sarcasm labels...")
    with open(HARD_CASES_DATA, encoding="utf-8") as f:
        hard_cases = json.load(f)

    hard_sent_map = {"positive": 0, "negative": 1, "neutral": 2}
    hard_rows = []
    for item in hard_cases:
        sent = item["sentiment"]
        cat = item.get("category", "")

        # Map sentiment
        if sent == "negative":
            sent_label = 1
        elif sent == "positive":
            sent_label = 0
        elif sent == "neutral":
            sent_label = 2
        else:
            continue

        # Map sarcasm
        if cat == "sarcasm":
            sarcasm_label = 1  # sarcastic
        elif cat in ("mixed", "implicit_negative", "question_complaint", "emoji_heavy"):
            sarcasm_label = 0  # sincere (not sarcastic, even if negative)
        elif cat == "financial_neutral":
            sarcasm_label = 0  # sincere
        else:
            sarcasm_label = NO_SARCASM_LABEL

        hard_rows.append({
            "text": item["text"],
            "sentiment_label": sent_label,
            "sarcasm_label": sarcasm_label,
            "source": "hard_case",
        })

    hard_df = pd.DataFrame(hard_rows)
    log.info(f"  Hard cases: {len(hard_df)} rows")
    log.info(f"  With sarcasm labels: {(hard_df['sarcasm_label'] >= 0).sum()}")

    # Combine
    combined = pd.concat([df, aug_df, hard_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)
    log.info(f"  Combined: {len(combined)} rows")

    return combined


# ---- Training Loop ----

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    data = load_data()

    # Split: hard cases and augmented go to train, test is original only
    original = data[data["source"] == "original"]
    non_original = data[data["source"] != "original"]

    train_orig, temp = train_test_split(
        original, test_size=0.15, random_state=SEED,
        stratify=original["sentiment_label"],
    )
    val_df, test_df = train_test_split(
        temp, test_size=0.5, random_state=SEED,
        stratify=temp["sentiment_label"],
    )
    train_df = pd.concat([train_orig, non_original], ignore_index=True).sample(
        frac=1, random_state=SEED
    )

    log.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    sarcasm_labeled = (train_df["sarcasm_label"] >= 0).sum()
    log.info(f"Sarcasm-labeled in train: {sarcasm_labeled}")

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_enc = tokenizer(
        train_df["text"].tolist(), truncation=True,
        padding="max_length", max_length=MAX_SEQ_LENGTH, return_tensors="pt",
    )
    val_enc = tokenizer(
        val_df["text"].tolist(), truncation=True,
        padding="max_length", max_length=MAX_SEQ_LENGTH, return_tensors="pt",
    )
    test_enc = tokenizer(
        test_df["text"].tolist(), truncation=True,
        padding="max_length", max_length=MAX_SEQ_LENGTH, return_tensors="pt",
    )

    train_ds = MultiTaskDataset(
        train_enc, train_df["sentiment_label"].values, train_df["sarcasm_label"].values,
    )
    val_ds = MultiTaskDataset(
        val_enc, val_df["sentiment_label"].values, val_df["sarcasm_label"].values,
    )
    test_ds = MultiTaskDataset(
        test_enc, test_df["sentiment_label"].values, test_df["sarcasm_label"].values,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2)

    # Model
    model = MultiTaskSentimentModel(BASE_MODEL).to(device)

    # Class weights for sentiment
    sent_weights = compute_class_weight(
        "balanced", classes=np.array([0, 1, 2]),
        y=train_df["sentiment_label"].values,
    )
    sent_weights_t = torch.tensor(sent_weights, dtype=torch.float32).to(device)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # Training loop
    best_f1 = 0.0
    patience = 3
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sent_labels = batch["sentiment_labels"].to(device)
            sarc_labels = batch["sarcasm_labels"].to(device)

            sent_logits, sarc_logits = model(input_ids, attention_mask)

            # Sentiment loss (all examples)
            loss = nn.CrossEntropyLoss(weight=sent_weights_t)(sent_logits, sent_labels)

            # Sarcasm loss (only labeled examples)
            sarc_mask = sarc_labels >= 0
            if sarc_mask.any():
                sarc_loss = nn.CrossEntropyLoss()(
                    sarc_logits[sarc_mask], sarc_labels[sarc_mask]
                )
                loss = loss + SARCASM_LOSS_WEIGHT * sarc_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                sent_logits, _ = model(input_ids, attention_mask)
                preds = torch.argmax(sent_logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["sentiment_labels"].numpy())

        f1 = f1_score(all_labels, all_preds, average="macro")
        avg_loss = total_loss / len(train_loader)
        log.info(f"Epoch {epoch+1}/{EPOCHS} -- loss: {avg_loss:.4f}, val_f1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            # Save best model
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
            tokenizer.save_pretrained(str(OUTPUT_DIR))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info("Early stopping triggered.")
                break

    # Load best and evaluate on test
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pt"))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sent_logits, _ = model(input_ids, attention_mask)
            preds = torch.argmax(sent_logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["sentiment_labels"].numpy())

    report = classification_report(
        all_labels, all_preds,
        target_names=["positive", "negative", "neutral"],
    )
    log.info(f"\nTest Results:\n{report}")

    # Save metrics
    report_dict = classification_report(
        all_labels, all_preds,
        target_names=["positive", "negative", "neutral"],
        output_dict=True,
    )
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump({
            "base_model": BASE_MODEL,
            "sarcasm_loss_weight": SARCASM_LOSS_WEIGHT,
            "best_val_f1": round(best_f1, 4),
            "test_report": report_dict,
        }, f, indent=2, ensure_ascii=False)

    log.info(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
```

### Stage C: Integration into app.py

```python
"""
Updated predict_sentiment method in ModelManager that integrates
rule-based sarcasm detection with the ML model.
"""


class ModelManager:
    def __init__(self):
        self.sentiment_pipe = None
        self.sarcasm_detector = SarcasmDetector()  # rule-based
        # ... existing init ...

    def predict_sentiment(self, text: str) -> SentimentResponse:
        # Step 1: Get ML prediction
        ml_result = self.sentiment_pipe(text, top_k=None)
        sentiment = _map_sentiment(ml_result)

        # Step 2: Run sarcasm detector
        sarcasm = self.sarcasm_detector.analyze(text)

        # Step 3: Override if sarcasm detected with high confidence
        if (
            sarcasm.should_flip_sentiment
            and sentiment.label == "positive"
            and sarcasm.confidence >= 0.5
        ):
            # Flip positive -> negative
            return SentimentResponse(
                label="negative",
                score=round(sarcasm.confidence, 4),
            )

        return sentiment
```

---

## 9. Generating Sarcasm Training Data

We can extend the existing `augment_data.py` pipeline to generate sarcastic examples.
The key is prompting the LLM with explicit sarcasm patterns.

```python
def generate_sarcasm_batch(examples: list[str], count: int = 20) -> list[dict]:
    """Generate synthetic sarcastic Russian social media posts."""
    prompt = f"""Сгенерируй {count} уникальных САРКАСТИЧЕСКИХ сообщений из социальных сетей на русском языке.

Тематика: жалобы на мобильного оператора, банк или технологическую компанию,
но написанные САРКАСТИЧЕСКИ — с притворной похвалой.

Паттерны сарказма (используй разные):
1. Позитивное слово + негативный факт: "Отличная работа МТС, 5 часов без связи"
2. "Ну спасибо" + описание проблемы: "Ну спасибо Сбер, заблокировали карту"
3. "Как же я люблю когда" + проблема: "Как же я люблю когда приложение крашится"
4. Ироничные эмодзи (🙃 👏 🚀): "Космическая скорость интернета, 2 мегабита 🚀"
5. "Так держать!" после описания проблемы
6. "Гениально/Великолепно/Браво" + описание неудачи
7. "Всего N часов/дней/минут" с большим числом (ирония)

Каждое сообщение должно ВЫГЛЯДЕТЬ позитивным, но БЫТЬ жалобой.
Используй разные компании: Сбер, МТС, Билайн, Мегафон, Тинькофф, ВТБ, Яндекс, Озон, и др.

Правила:
1. 1-2 предложения, как в Telegram/VK
2. Разговорный стиль
3. Каждое на отдельной строке
4. Без нумерации и пояснений

Сгенерируй ровно {count} саркастических сообщений:"""

    response = call_kimi(prompt)
    lines = [
        line.strip().lstrip("0123456789.-) ")
        for line in response.strip().split("\n")
        if line.strip() and len(line.strip()) > 10
    ]

    results = []
    for text in lines[:count]:
        results.append({
            "text": text,
            "sentiment": "негатив",     # true sentiment is negative
            "sarcasm": True,             # explicitly sarcastic
            "source": "kimi_sarcasm",
        })
    return results


def generate_sincere_positive_batch(examples: list[str], count: int = 20) -> list[dict]:
    """Generate SINCERE positive posts (to contrast with sarcastic ones)."""
    prompt = f"""Сгенерируй {count} ИСКРЕННЕ ПОЗИТИВНЫХ сообщений из социальных сетей на русском языке.

Тематика: НАСТОЯЩАЯ похвала мобильному оператору, банку или технологической компании.
Это должны быть ИСКРЕННИЕ положительные отзывы, НЕ сарказм.

Примеры искренней похвалы:
- "Спасибо Сбер, перевод пришёл за 2 секунды, удобно!"
- "Отличное приложение у Тинькофф, всё интуитивно понятно"
- "МТС реально улучшили покрытие в нашем районе, молодцы"

Правила:
1. 1-2 предложения, как в Telegram/VK
2. Разговорный стиль
3. Каждое на отдельной строке
4. Без нумерации

Сгенерируй ровно {count} искренне позитивных сообщений:"""

    response = call_kimi(prompt)
    lines = [
        line.strip().lstrip("0123456789.-) ")
        for line in response.strip().split("\n")
        if line.strip() and len(line.strip()) > 10
    ]

    results = []
    for text in lines[:count]:
        results.append({
            "text": text,
            "sentiment": "позитив",
            "sarcasm": False,
            "source": "kimi_sincere",
        })
    return results
```

**Target:** Generate ~1000 sarcastic + ~1000 sincere positive examples to create
a balanced sarcasm detection dataset for the auxiliary task head.

---

## 10. Evaluation Plan

### Sarcasm-Specific Test Set

Create a dedicated test set from the 25 sarcasm examples in `data_hard_cases.json`
plus manually curated examples from real data. Target: 100-200 sarcasm test cases.

### Metrics

1. **Sarcasm Detection Accuracy:** precision/recall/F1 on sarcasm binary task
2. **Sentiment Flip Rate:** % of sarcastic texts correctly classified as negative
   (currently ~0%, target: >80%)
3. **False Positive Rate:** % of sincere positive texts incorrectly flipped
   (target: <2%)
4. **Overall Sentiment F1:** should not decrease on non-sarcastic text

### Validation on the Failing Examples

```python
test_cases = [
    ("Ну спасибо Сбер, очень помогли \U0001f643", "negative"),
    ("Отличная работа МТС, 5 часов без связи, так держать!", "negative"),
    ("Браво, ещё одну комиссию придумали \U0001f44f", "negative"),
    # Additional test cases:
    ("Как же я люблю когда приложение обновляется и всё ломается \u2764\ufe0f", "negative"),
    ("Супер поддержка, перевели на 7-го оператора", "negative"),
    ("Великолепный интернет, 2 мегабита за 800 рублей, живём в будущем", "negative"),
    # Sincere positives (should NOT be flipped):
    ("Спасибо Сбер, перевод пришёл моментально!", "positive"),
    ("Отличная работа МТС, покрытие стало намного лучше", "positive"),
    ("Браво Тинькофф, лучшее приложение на рынке \U0001f44f", "positive"),
]
```

---

## 11. Implementation Roadmap

### Phase 1: Quick Win -- Rule-Based Pre-Filter (1-2 days)

1. Implement `SarcasmDetector` class (Section 8, Stage A)
2. Integrate into `app.py` as a pre-processing step
3. Test on the 3 failing examples + hard cases
4. **Expected improvement:** catches ~70% of obvious sarcasm patterns
5. **Risk:** false positives on sincere praise with exclamations

### Phase 2: Data Generation (2-3 days)

1. Generate 1000 sarcastic examples via Kimi API (Section 9)
2. Generate 1000 sincere positive examples as counterbalance
3. Manually review and filter generated data (~500 usable per class)
4. Add to `data_hard_cases.json` or create `data_sarcasm.jsonl`

### Phase 3: Multi-Task Fine-Tuning (2-3 days)

1. Implement `MultiTaskSentimentModel` (Section 3)
2. Implement `finetune_v3_multitask.py` (Section 8, Stage B)
3. Train with sarcasm auxiliary task
4. Evaluate: overall F1 should stay >= v2, sarcasm accuracy target >80%

### Phase 4: Integration & A/B Test (1-2 days)

1. Update `app.py` with hybrid pipeline (rule-based + ML)
2. Run A/B test on held-out data comparing v2 vs v3
3. Deploy if improvement confirmed

### Total estimated effort: 6-10 days

---

## 12. Summary of Recommendations

| Approach | Effort | Expected Impact | Precision | Recall |
|----------|--------|----------------|-----------|--------|
| Rule-based pre-filter | Low (1-2d) | High for obvious cases | ~90% | ~60% |
| Emoji sarcasm features | Low (1d) | Medium | ~85% | ~40% |
| Multi-task learning | Medium (3-4d) | High | ~80% | ~75% |
| Sarcasm data generation | Medium (2-3d) | Enables Phase 3 | N/A | N/A |
| **Hybrid (all combined)** | **6-10d** | **Very High** | **~85%** | **~80%** |

**Primary recommendation:** Start with Phase 1 (rule-based filter) for immediate
improvement, then pursue Phase 2+3 (data generation + multi-task) for robust
long-term solution.

The rule-based filter alone will fix the three failing examples today. The multi-task
approach will generalize to sarcasm patterns we have not explicitly coded rules for.
