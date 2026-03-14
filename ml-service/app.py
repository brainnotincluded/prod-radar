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

# Fine-tuned on 188K Russian social media mentions (F1-macro: 0.76)
# Falls back to base model if fine-tuned not found
SENTIMENT_MODEL = "models/sentiment-finetuned/final"
SENTIMENT_MODEL_FALLBACK = "cointegrated/rubert-tiny2-cedr-emotion-detection"

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

        # Sentiment pipeline — try fine-tuned model first
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

    def predict_sentiment(self, text: str) -> SentimentResponse:
        result = self.sentiment_pipe(text, top_k=None)
        return _map_sentiment(result)

    def predict_sentiment_batch(self, texts: list[str]) -> list[SentimentResponse]:
        results = self.sentiment_pipe(texts, top_k=None, batch_size=32)
        return [_map_sentiment(r) for r in results]

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
    We pick the highest-scoring label and use its raw score as confidence.
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
    embedding: list[float]

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """Combined sentiment + embedding in one call (used by enricher service)."""
    if not models.ready:
        raise HTTPException(503, "Models not loaded yet")
    sentiment = models.predict_sentiment(req.text)
    emb = models.embed(req.text)
    return AnalyzeResponse(
        sentiment_label=sentiment.label,
        sentiment_score=sentiment.score,
        embedding=emb,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
