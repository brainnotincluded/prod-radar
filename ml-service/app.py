"""
Prod Radar ML Service
=====================
FastAPI service providing sentiment analysis, embeddings, and risk classification
for Russian-language social media mentions.

Endpoints:
  GET  /health            — Healthcheck
  POST /sentiment         — Single text sentiment
  POST /sentiment/batch   — Batch sentiment
  POST /embedding         — Single text embedding
  POST /embedding/batch   — Batch embeddings
  POST /classify-risk     — Risk detection
"""

import logging
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline, AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────

# Phase 1: ruRoBERTa-large fine-tuned on 318K samples (F1-macro: 0.868)
# Falls back to HuggingFace model if local not found
SENTIMENT_MODEL = "models/phase1-ruroberta/final"
SENTIMENT_MODEL_FALLBACK = "Daniil125/prodradar-sentiment-ru"

# Embeddings: sentence-transformers model for Russian, outputs 768-dim vectors
EMBEDDING_MODEL = "cointegrated/LaBSE-en-ru"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_BATCH_SIZE = 128


# ─── Pydantic Models ─────────────────────────────────────────────────

class TextRequest(BaseModel):
    text: str
    lang: str = "ru"

class SentimentResponse(BaseModel):
    label: str  # positive / negative / neutral / mixed
    score: float = Field(ge=0.0, le=1.0)

class BatchItem(BaseModel):
    id: str
    text: str

class BatchSentimentRequest(BaseModel):
    items: list[BatchItem]

class BatchSentimentResponse(BaseModel):
    results: list[dict]

class EmbeddingResponse(BaseModel):
    embedding: list[float]

class BatchEmbeddingRequest(BaseModel):
    items: list[BatchItem]

class BatchEmbeddingResponse(BaseModel):
    results: list[dict]

class RiskRequest(BaseModel):
    text: str
    risk_words: list[str]

class RiskResponse(BaseModel):
    is_risk: bool
    matched: list[str]
    confidence: float = Field(ge=0.0, le=1.0)

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    device: str = ""
    latency_ms: float = 0.0


# ─── Model Manager ───────────────────────────────────────────────────

class ModelManager:
    def __init__(self):
        self.sentiment_pipe = None
        self.embed_tokenizer = None
        self.embed_model = None
        self.ready = False

    def load(self):
        log.info(f"Loading models on device={DEVICE}...")
        t0 = time.time()

        # Sentiment pipeline — local Phase 1 model or HuggingFace fallback
        import os
        model_path = SENTIMENT_MODEL if os.path.isdir(SENTIMENT_MODEL) else SENTIMENT_MODEL_FALLBACK
        log.info(f"Loading sentiment model: {model_path}")
        self.sentiment_pipe = pipeline(
            "text-classification",
            model=model_path,
            device=0 if DEVICE == "cuda" else -1,
            truncation=True,
            max_length=512,
        )

        # Embedding model
        log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embed_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        self.embed_model = AutoModel.from_pretrained(EMBEDDING_MODEL)
        self.embed_model.to(DEVICE)
        self.embed_model.eval()

        self.ready = True
        log.info(f"All models loaded in {time.time() - t0:.1f}s")

    def _classify_single(self, text: str) -> SentimentResponse:
        """Classify a single text chunk (no sentence splitting)."""
        result = self.sentiment_pipe(text, top_k=None)
        return _map_sentiment(result)

    def predict_sentiment(self, text: str) -> SentimentResponse:
        """Sentence-level aggregation for long texts + post-processing."""
        sentences = _split_sentences(text)

        # Short text — classify directly
        if len(sentences) <= 1 or len(text) < 300:
            raw = self._classify_single(text)
            return _postprocess_sentiment(text, raw)

        # Classify each sentence
        results = self.sentiment_pipe(
            sentences, top_k=None, batch_size=32
        )
        sentiments = [_map_sentiment(r) for r in results]

        # Accumulate emotional signals only (ignore neutral completely)
        n = len(sentiments)
        pos_total = 0.0
        neg_total = 0.0
        pos_count = 0
        neg_count = 0
        for s in sentiments:
            if s.label == "positive":
                pos_total += s.score
                pos_count += 1
            elif s.label == "negative":
                neg_total += s.score
                neg_count += 1

        emotional_count = pos_count + neg_count
        emotional_ratio = emotional_count / n if n > 0 else 0

        # Need at least 15% emotional sentences to color the whole text
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

        raw = SentimentResponse(label=best_label, score=round(best_score, 4))
        return _postprocess_sentiment(text, raw)

    def predict_sentiment_batch(self, texts: list[str]) -> list[SentimentResponse]:
        return [self.predict_sentiment(t) for t in texts]

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        encoded = self.embed_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            outputs = self.embed_model(**encoded)
            # Mean pooling over token embeddings
            attention_mask = encoded["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            embeddings = torch.sum(
                token_embeddings * input_mask_expanded, 1
            ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            # L2 normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().tolist()


# ─── Sentiment Label Mapping ─────────────────────────────────────────

# Map model-specific labels to our standard: positive/negative/neutral/mixed
_LABEL_MAP = {
    # rubert-tiny2-cedr labels
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "speech": "neutral",
    "skip": "neutral",
    # Common HF sentiment labels
    "POSITIVE": "positive",
    "NEGATIVE": "negative",
    "NEUTRAL": "neutral",
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    # Emotion labels that map to sentiment
    "joy": "positive",
    "sadness": "negative",
    "anger": "negative",
    "fear": "negative",
    "surprise": "neutral",
    "disgust": "negative",
    "no_emotion": "neutral",
}


def _map_sentiment(raw_results: list[dict]) -> SentimentResponse:
    """Map model output to standard sentiment label with confidence.

    The fine-tuned model uses sigmoid (multi-label) outputs, not softmax.
    We pick the highest-scoring label, with a neutral-bias margin to compensate
    for the model's underrepresentation of neutral in training data (16.5%).
    """
    if not raw_results:
        return SentimentResponse(label="neutral", score=0.5)

    # Aggregate raw scores by our standard labels (take max per label)
    scores: dict[str, float] = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for item in raw_results:
        mapped = _LABEL_MAP.get(item["label"], "neutral")
        scores[mapped] = max(scores[mapped], item["score"])

    # Pick the dominant label by raw score
    best_label = max(scores, key=lambda k: scores[k])
    best_score = scores[best_label]

    return SentimentResponse(label=best_label, score=round(best_score, 4))


# ─── Phase 2: Sarcasm / Churn / Emoji Post-Processing ─────────────────

import re

_POSITIVE_WORDS = {
    "спасибо", "отлично", "отличная", "отличный", "отличное",
    "прекрасно", "прекрасная", "замечательно", "замечательная",
    "браво", "молодцы", "супер", "класс", "круто", "классно",
    "великолепно", "великолепная", "потрясающе", "потрясающая",
    "шикарно", "шикарная", "обожаю", "люблю", "лучший", "лучшая",
    "идеально", "безупречно", "так держать", "респект", "топ",
}

_NEGATIVE_CONTEXT = {
    "без связи", "не работает", "сломал", "списали", "заблокировали",
    "проблема", "висит", "лежит", "тормозит", "крашится", "глючит",
    "ошибка", "комиссию", "комиссия", "штраф", "переплата",
    "очередь", "ожидание", "не отвечают", "не перезвонили",
    "не решили", "навязали", "отказали", "сломалось", "часов",
    "дней", "недель", "месяц", "месяца",
}

_SARCASM_EMOJIS = {"🙃", "🤡", "💩"}
_MODERATE_SARCASM_EMOJIS = {"👏", "🔥", "🚀", "🏆", "💪"}

_NEGATIVE_EMOJIS = {"💀", "😤", "😡", "🤬", "😠", "💀", "🙄", "😒"}
_POSITIVE_EMOJIS = {"❤️", "❤", "😍", "👍", "🎉", "🥰", "💕", "😊"}

_IRONIC_PATTERNS = [
    r"так\s+держать", r"очень\s+(помогли|выгодно|удобно)",
    r"ну\s+спасибо", r"всего[\s\-]то",
    r"как\s+же\s+я\s+(люблю|обожаю)", r"обожаю\s+когда",
]

_CHURN_PATTERNS = [
    r"ухожу\s+(в|к|на)", r"перейду\s+(в|к|на)", r"перехожу\s+(в|к|на)",
    r"закрываю\s+(счёт|счет|карту)", r"достали", r"надоело",
    r"отключаюсь", r"расторгаю",
]


def _postprocess_sentiment(text: str, sentiment: SentimentResponse) -> SentimentResponse:
    """Apply rule-based corrections for sarcasm, churn, and emoji patterns."""
    text_lower = text.lower().strip()
    label = sentiment.label
    score = sentiment.score

    # --- Sarcasm detection: positive word + negative context ---
    has_pos_word = any(w in text_lower for w in _POSITIVE_WORDS)
    has_neg_context = any(m in text_lower for m in _NEGATIVE_CONTEXT)
    has_sarcasm_emoji = any(e in text for e in _SARCASM_EMOJIS)
    has_moderate_emoji = any(e in text for e in _MODERATE_SARCASM_EMOJIS)
    has_ironic = any(re.search(p, text_lower) for p in _IRONIC_PATTERNS)

    sarcasm_score = 0.0
    if has_pos_word and has_neg_context:
        sarcasm_score += 0.4
    if has_sarcasm_emoji:
        sarcasm_score += 0.3
    elif has_moderate_emoji and has_pos_word:
        sarcasm_score += 0.2
    if has_ironic:
        sarcasm_score += 0.25

    # Flip positive → negative if sarcasm detected
    if sarcasm_score >= 0.35 and label == "positive":
        return SentimentResponse(label="negative", score=round(max(score, sarcasm_score), 4))

    # --- Churn detection: "ухожу", "перейду к конкурентам" → negative ---
    if any(re.search(p, text_lower) for p in _CHURN_PATTERNS):
        if label != "negative":
            return SentimentResponse(label="negative", score=round(max(score, 0.65), 4))

    # --- Emoji-only or emoji-dominant posts ---
    text_no_space = text_lower.replace(" ", "")
    if len(text_no_space) < 20:  # short post, emoji matters more
        has_neg_emoji = any(e in text for e in _NEGATIVE_EMOJIS)
        has_pos_emoji = any(e in text for e in _POSITIVE_EMOJIS)
        if has_neg_emoji and not has_pos_emoji and label != "negative":
            return SentimentResponse(label="negative", score=round(max(score, 0.60), 4))
        if has_pos_emoji and not has_neg_emoji and label != "positive":
            return SentimentResponse(label="positive", score=round(max(score, 0.60), 4))

    return sentiment


# ─── Risk Classification ─────────────────────────────────────────────

def classify_risk(
    text: str,
    risk_words: list[str],
    models: ModelManager,
) -> RiskResponse:
    """Hybrid risk detection: keyword matching + sentiment signal."""
    text_lower = text.lower()

    # Keyword matching
    matched = [w for w in risk_words if w.lower() in text_lower]

    # Get sentiment signal to boost confidence
    sentiment = models.predict_sentiment(text)
    is_negative = sentiment.label in ("negative", "mixed")

    # Compute confidence
    if matched and is_negative:
        confidence = min(0.95, 0.6 + 0.1 * len(matched) + sentiment.score * 0.2)
    elif matched:
        confidence = min(0.85, 0.4 + 0.1 * len(matched))
    elif is_negative and sentiment.score > 0.8:
        confidence = 0.5  # Negative but no keywords — moderate risk
    else:
        confidence = 0.1

    is_risk = confidence >= 0.4

    return RiskResponse(
        is_risk=is_risk,
        matched=matched,
        confidence=round(confidence, 4),
    )


# ─── FastAPI App ──────────────────────────────────────────────────────

models = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    models.load()
    yield


app = FastAPI(
    title="Prod Radar ML Service",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    t0 = time.time()
    # Quick inference to measure latency
    latency = 0.0
    if models.ready:
        models.predict_sentiment("тест")
        latency = (time.time() - t0) * 1000
    return HealthResponse(
        status="ok" if models.ready else "loading",
        models_loaded=models.ready,
        device=DEVICE,
        latency_ms=round(latency, 1),
    )


@app.post("/sentiment", response_model=SentimentResponse)
def sentiment(req: TextRequest):
    if not models.ready:
        raise HTTPException(503, "Models not loaded yet")
    return models.predict_sentiment(req.text)


@app.post("/sentiment/batch", response_model=BatchSentimentResponse)
def sentiment_batch(req: BatchSentimentRequest):
    if not models.ready:
        raise HTTPException(503, "Models not loaded yet")
    if len(req.items) > MAX_BATCH_SIZE:
        raise HTTPException(400, f"Batch size exceeds limit of {MAX_BATCH_SIZE}")

    texts = [item.text for item in req.items]
    results = models.predict_sentiment_batch(texts)
    return BatchSentimentResponse(
        results=[
            {"id": item.id, "label": r.label, "score": r.score}
            for item, r in zip(req.items, results)
        ]
    )


@app.post("/embedding", response_model=EmbeddingResponse)
def embedding(req: TextRequest):
    if not models.ready:
        raise HTTPException(503, "Models not loaded yet")
    vec = models.embed(req.text)
    return EmbeddingResponse(embedding=vec)


@app.post("/embedding/batch", response_model=BatchEmbeddingResponse)
def embedding_batch(req: BatchEmbeddingRequest):
    if not models.ready:
        raise HTTPException(503, "Models not loaded yet")
    if len(req.items) > MAX_BATCH_SIZE:
        raise HTTPException(400, f"Batch size exceeds limit of {MAX_BATCH_SIZE}")

    texts = [item.text for item in req.items]
    vecs = models.embed_batch(texts)
    return BatchEmbeddingResponse(
        results=[
            {"id": item.id, "embedding": vec}
            for item, vec in zip(req.items, vecs)
        ]
    )


@app.post("/classify-risk", response_model=RiskResponse)
def risk(req: RiskRequest):
    if not models.ready:
        raise HTTPException(503, "Models not loaded yet")
    return classify_risk(req.text, req.risk_words, models)


# ─── Combined endpoint for BrandRadar enricher ───────────────────────

class AnalyzeRequest(BaseModel):
    text: str

class AnalyzeResponse(BaseModel):
    sentiment_label: str
    sentiment_score: float
    relevance_score: float = 0.0
    similarity_score: float = 0.0
    embedding: list[float]

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """Combined sentiment + relevance + similarity + embedding (used by enricher)."""
    if not models.ready:
        raise HTTPException(503, "Models not loaded yet")
    sentiment = models.predict_sentiment(req.text)
    emb = models.embed(req.text)

    # Relevance: based on sentiment confidence (higher confidence = more relevant)
    # Texts with strong sentiment (pos or neg) are more relevant than neutral
    if sentiment.label in ("positive", "negative"):
        relevance = min(sentiment.score * 1.2, 1.0)
    else:
        relevance = max(0.3, sentiment.score * 0.5)

    # Similarity: placeholder (real dedup is done via embedding cosine distance in enricher)
    similarity = 0.0

    return AnalyzeResponse(
        sentiment_label=sentiment.label,
        sentiment_score=sentiment.score,
        relevance_score=round(relevance, 4),
        similarity_score=round(similarity, 4),
        embedding=emb,
    )


# ─── Detailed sentence-level analysis (Perplexity-style) ─────────────

def _split_sentences(text: str) -> list[str]:
    """Split Russian text into sentences."""
    # Handle common abbreviations to avoid false splits
    text = re.sub(r'(\d)\.\s*(\d)', r'\1[DOT]\2', text)  # decimals
    for abbr in ["г.", "гг.", "т.д.", "т.п.", "т.е.", "др.", "пр.", "руб.", "коп.",
                  "млн.", "млрд.", "трлн.", "тыс.", "ул.", "д.", "стр.", "корп."]:
        text = text.replace(abbr, abbr.replace(".", "[DOT]"))

    # Split on sentence-ending punctuation
    parts = re.split(r'(?<=[.!?])\s+', text)

    # Restore dots
    sentences = []
    for p in parts:
        p = p.replace("[DOT]", ".").strip()
        if len(p) > 5:
            sentences.append(p)
    return sentences if sentences else [text]


class SentenceAnalysis(BaseModel):
    text: str
    label: str
    score: float
    index: int


class DetailedAnalyzeRequest(BaseModel):
    text: str
    risk_words: list[str] = []


class DetailedAnalyzeResponse(BaseModel):
    # Overall
    sentiment_label: str
    sentiment_score: float
    embedding: list[float]
    # Per-sentence breakdown
    sentences: list[SentenceAnalysis]
    # Summary stats
    sentence_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    # Risk
    risk_detected: bool = False
    risk_words_matched: list[str] = []
    # Highlights — most positive and most negative sentences
    most_positive: SentenceAnalysis | None = None
    most_negative: SentenceAnalysis | None = None


@app.post("/analyze/detailed", response_model=DetailedAnalyzeResponse)
def analyze_detailed(req: DetailedAnalyzeRequest):
    """Perplexity-style detailed analysis: overall + per-sentence sentiment."""
    if not models.ready:
        raise HTTPException(503, "Models not loaded yet")

    # Overall sentiment + embedding
    overall = models.predict_sentiment(req.text)
    emb = models.embed(req.text)

    # Split into sentences and classify each
    raw_sentences = _split_sentences(req.text)
    sentence_sentiments = models.predict_sentiment_batch(raw_sentences)

    sentences = []
    pos_count = neg_count = neu_count = 0
    most_pos = most_neg = None

    for i, (sent_text, sent_result) in enumerate(zip(raw_sentences, sentence_sentiments)):
        sa = SentenceAnalysis(
            text=sent_text,
            label=sent_result.label,
            score=sent_result.score,
            index=i,
        )
        sentences.append(sa)

        if sent_result.label == "positive":
            pos_count += 1
            if most_pos is None or sent_result.score > most_pos.score:
                most_pos = sa
        elif sent_result.label == "negative":
            neg_count += 1
            if most_neg is None or sent_result.score > most_neg.score:
                most_neg = sa
        else:
            neu_count += 1

    # Risk detection
    risk_matched = []
    if req.risk_words:
        text_lower = req.text.lower()
        risk_matched = [w for w in req.risk_words if w.lower() in text_lower]

    return DetailedAnalyzeResponse(
        sentiment_label=overall.label,
        sentiment_score=overall.score,
        embedding=emb,
        sentences=sentences,
        sentence_count=len(sentences),
        positive_count=pos_count,
        negative_count=neg_count,
        neutral_count=neu_count,
        risk_detected=len(risk_matched) > 0,
        risk_words_matched=risk_matched,
        most_positive=most_pos,
        most_negative=most_neg,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
