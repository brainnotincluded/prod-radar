"""
Prod Radar ML Service v2 — Multi-Task Edition
===============================================
FastAPI service serving a multi-task model (sentiment + relevance + similarity
in one forward pass) plus LaBSE embeddings.

Model: MultiTaskSentimentModel (cointegrated/rubert-tiny2, 3 heads)
  - Sentiment: 3-class (positive/negative/neutral)
  - Relevance: binary sigmoid score
  - Similarity: regression sigmoid score

Embedding: LaBSE-en-ru (768-dim, L2-normalized)

Endpoints:
  GET  /health              — Healthcheck with latency probe
  POST /analyze             — Combined sentiment + relevance + similarity + embedding
  POST /sentiment           — Single text sentiment
  POST /sentiment/batch     — Batch sentiment (max 128)
  POST /relevance           — Relevance scoring with keyword matching
  POST /embedding           — Single text embedding (768-dim)
  POST /embedding/batch     — Batch embeddings
  POST /classify-risk       — Risk detection (keyword + sentiment)
  POST /analyze/detailed    — Sentence-level Perplexity-style analysis
"""

import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("ml-service-v2")

# ─── Config ───────────────────────────────────────────────────────────

MULTITASK_MODEL_DIR = Path("models/multitask-final")
MULTITASK_BASE_MODEL = "cointegrated/rubert-tiny2"
MULTITASK_MAX_SEQ_LEN = 128

EMBEDDING_MODEL_NAME = "cointegrated/LaBSE-en-ru"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_BATCH_SIZE = 128

# Sentiment label index mapping (matches training: SENTIMENT_LABELS)
SENTIMENT_LABELS = ["positive", "negative", "neutral"]


# ─── Multi-Task Model Architecture ───────────────────────────────────

class MultiTaskSentimentModel(nn.Module):
    """Three-headed model: sentiment + relevance + similarity."""

    def __init__(self, model_name, num_sentiment_classes=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.sentiment_head = nn.Linear(hidden, num_sentiment_classes)
        self.relevance_head = nn.Linear(hidden, 1)
        self.similarity_head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        sentiment_logits = self.sentiment_head(cls_output)
        relevance_logit = self.relevance_head(cls_output).squeeze(-1)
        similarity_logit = self.similarity_head(cls_output).squeeze(-1)
        return sentiment_logits, relevance_logit, similarity_logit


# ─── Pydantic Models ─────────────────────────────────────────────────

class TextRequest(BaseModel):
    text: str
    lang: str = "ru"


class SentimentResponse(BaseModel):
    label: str
    score: float = Field(ge=0.0, le=1.0)


class BatchItem(BaseModel):
    id: str
    text: str


class BatchSentimentRequest(BaseModel):
    items: list[BatchItem]


class BatchSentimentResponse(BaseModel):
    results: list[dict]


class AnalyzeRequest(BaseModel):
    text: str


class AnalyzeResponse(BaseModel):
    sentiment_label: str
    sentiment_score: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0)
    similarity_score: float = Field(ge=0.0, le=1.0)
    embedding: list[float]


class RelevanceRequest(BaseModel):
    text: str
    keywords: list[str] = []


class RelevanceResponse(BaseModel):
    relevance_score: float = Field(ge=0.0, le=1.0)
    is_relevant: bool
    matched_keywords: list[str]


class EmbeddingRequest(BaseModel):
    text: str


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


class SentenceAnalysis(BaseModel):
    text: str
    label: str
    score: float
    index: int


class DetailedAnalyzeRequest(BaseModel):
    text: str
    risk_words: list[str] = []


class DetailedAnalyzeResponse(BaseModel):
    sentiment_label: str
    sentiment_score: float
    embedding: list[float]
    relevance_score: float
    similarity_score: float
    sentences: list[SentenceAnalysis]
    sentence_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    risk_detected: bool = False
    risk_words_matched: list[str] = []
    most_positive: SentenceAnalysis | None = None
    most_negative: SentenceAnalysis | None = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    device: str = ""
    multitask_model: str = ""
    embedding_model: str = ""
    latency_ms: float = 0.0


# ─── Phase 2: Sarcasm / Churn / Emoji Post-Processing ────────────────

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

_NEGATIVE_EMOJIS = {"💀", "😤", "😡", "🤬", "😠", "🙄", "😒"}
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

    # Flip positive -> negative if sarcasm detected
    if sarcasm_score >= 0.35 and label == "positive":
        return SentimentResponse(label="negative", score=round(max(score, sarcasm_score), 4))

    # --- Churn detection: "ухожу", "перейду к конкурентам" -> negative ---
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


# ─── Sentence Splitting ──────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split Russian text into sentences, handling common abbreviations."""
    text = re.sub(r'(\d)\.\s*(\d)', r'\1[DOT]\2', text)  # decimals
    for abbr in [
        "г.", "гг.", "т.д.", "т.п.", "т.е.", "др.", "пр.", "руб.", "коп.",
        "млн.", "млрд.", "трлн.", "тыс.", "ул.", "д.", "стр.", "корп.",
    ]:
        text = text.replace(abbr, abbr.replace(".", "[DOT]"))

    parts = re.split(r'(?<=[.!?])\s+', text)

    sentences = []
    for p in parts:
        p = p.replace("[DOT]", ".").strip()
        if len(p) > 5:
            sentences.append(p)
    return sentences if sentences else [text]


# ─── Model Manager ───────────────────────────────────────────────────

class ModelManager:
    """Loads and manages the multi-task model and LaBSE embedding model."""

    def __init__(self):
        self.multitask_model: MultiTaskSentimentModel | None = None
        self.multitask_tokenizer = None
        self.embed_tokenizer = None
        self.embed_model = None
        self.model_config: dict = {}
        self.ready = False

    def load(self):
        log.info("Loading models on device=%s ...", DEVICE)
        t0 = time.time()

        # --- Multi-task model ---
        model_dir = MULTITASK_MODEL_DIR
        config_path = model_dir / "model_config.json"
        weights_path = model_dir / "model.pt"
        tokenizer_path = model_dir / "tokenizer"

        if config_path.exists():
            with open(config_path) as f:
                self.model_config = json.load(f)
            log.info("Model config: %s", self.model_config)

        base_model = self.model_config.get("base_model", MULTITASK_BASE_MODEL)
        num_classes = self.model_config.get("num_sentiment_classes", 3)

        log.info("Loading multi-task model: base=%s", base_model)
        self.multitask_model = MultiTaskSentimentModel(base_model, num_classes)

        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=True)
            self.multitask_model.load_state_dict(state_dict)
            log.info("Loaded weights from %s", weights_path)
        else:
            log.warning("No weights found at %s — using base model weights only", weights_path)

        self.multitask_model.to(DEVICE)
        self.multitask_model.eval()

        if tokenizer_path.exists():
            self.multitask_tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            log.info("Loaded tokenizer from %s", tokenizer_path)
        else:
            self.multitask_tokenizer = AutoTokenizer.from_pretrained(base_model)
            log.info("Loaded tokenizer from HuggingFace: %s", base_model)

        # --- LaBSE embedding model ---
        log.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        self.embed_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        self.embed_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
        self.embed_model.to(DEVICE)
        self.embed_model.eval()

        self.ready = True
        log.info("All models loaded in %.1fs", time.time() - t0)

    # ─── Multi-task inference ────────────────────────────────────────

    def _tokenize(self, texts: list[str]) -> dict:
        """Tokenize texts for the multi-task model."""
        max_len = self.model_config.get("max_seq_len", MULTITASK_MAX_SEQ_LEN)
        return self.multitask_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

    def _infer_multitask(
        self, texts: list[str]
    ) -> tuple[list[dict], list[float], list[float]]:
        """Run multi-task inference on a batch of texts.

        Returns:
            sentiments: list of {"label": str, "score": float}
            relevance_scores: list of floats 0-1
            similarity_scores: list of floats 0-1
        """
        encoded = self._tokenize(texts)
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)

        with torch.no_grad():
            sent_logits, rel_logit, sim_logit = self.multitask_model(
                input_ids, attention_mask
            )
            sentiment_probs = torch.softmax(sent_logits, dim=-1)
            relevance_scores = torch.sigmoid(rel_logit).cpu().tolist()
            similarity_scores = torch.sigmoid(sim_logit).cpu().tolist()

        # Handle scalar case (single item batch) — ensure lists
        if isinstance(relevance_scores, float):
            relevance_scores = [relevance_scores]
        if isinstance(similarity_scores, float):
            similarity_scores = [similarity_scores]

        sentiments = []
        for i in range(sentiment_probs.size(0)):
            probs = sentiment_probs[i].cpu()
            best_idx = probs.argmax().item()
            best_score = probs[best_idx].item()

            label = SENTIMENT_LABELS[best_idx]

            # Neutral margin fix: if winning label barely beats neutral,
            # default to neutral (compensates for training data imbalance)
            if label != "neutral":
                neutral_prob = probs[2].item()  # index 2 = neutral
                if neutral_prob > 0 and (best_score - neutral_prob) < 0.10:
                    label = "neutral"
                    best_score = neutral_prob

            sentiments.append({"label": label, "score": round(best_score, 4)})

        return sentiments, relevance_scores, similarity_scores

    def _aggregate_sentences(
        self, text: str
    ) -> tuple[list[dict], list[float], list[float]]:
        """Split text into sentences and aggregate multi-task results.

        Long texts (>300 chars) are split into sentences and classified
        individually. Results are aggregated by normalized voting/averaging
        across all sentences, giving equal weight to each sentence regardless
        of length. This fixes the truncation problem where the model only
        saw the first ~300 chars of 800+ char texts.
        """
        sentences = _split_sentences(text)

        # Short text or single sentence — classify directly
        if len(sentences) <= 1 or len(text) < 300:
            return self._infer_multitask([text])

        # Classify all sentences in one batch
        sentiments, rel_scores, sim_scores = self._infer_multitask(sentences)

        # Aggregate sentiment: majority vote by count, tiebreak by avg score
        n = len(sentences)
        label_counts = {"positive": 0, "negative": 0, "neutral": 0}
        label_score_sums = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        for s in sentiments:
            label_counts[s["label"]] += 1
            label_score_sums[s["label"]] += s["score"]

        # Winner = label with most sentences; tiebreak by total score
        best_label = max(label_counts, key=lambda k: (label_counts[k], label_score_sums[k]))
        best_score = label_score_sums[best_label] / max(label_counts[best_label], 1)

        # Only flip to neutral if sentiment labels are truly rare (<20% of sentences)
        if best_label != "neutral":
            sentiment_ratio = label_counts[best_label] / n
            if sentiment_ratio < 0.20:
                best_label = "neutral"
                best_score = label_score_sums["neutral"] / max(label_counts["neutral"], 1)

        agg_sentiment = [{"label": best_label, "score": round(best_score, 4)}]

        # Aggregate relevance & similarity: mean across sentences
        agg_rel = [round(sum(rel_scores) / n, 4)]
        agg_sim = [round(sum(sim_scores) / n, 4)]

        return agg_sentiment, agg_rel, agg_sim

    def predict_sentiment(self, text: str) -> SentimentResponse:
        """Single text sentiment with sentence-level aggregation + post-processing."""
        sentiments, _, _ = self._aggregate_sentences(text)
        raw = SentimentResponse(**sentiments[0])
        return _postprocess_sentiment(text, raw)

    def predict_sentiment_batch(self, texts: list[str]) -> list[SentimentResponse]:
        """Batch sentiment with sentence-level aggregation + post-processing."""
        if not texts:
            return []
        results = []
        for text in texts:
            sentiments, _, _ = self._aggregate_sentences(text)
            raw = SentimentResponse(**sentiments[0])
            results.append(_postprocess_sentiment(text, raw))
        return results

    def predict_relevance(self, text: str) -> float:
        """Single text relevance score with sentence-level aggregation."""
        _, relevance_scores, _ = self._aggregate_sentences(text)
        return round(relevance_scores[0], 4)

    def predict_all(
        self, text: str
    ) -> tuple[SentimentResponse, float, float]:
        """Single text: sentiment + relevance + similarity with sentence aggregation."""
        sentiments, rel_scores, sim_scores = self._aggregate_sentences(text)
        raw = SentimentResponse(**sentiments[0])
        sentiment = _postprocess_sentiment(text, raw)
        return sentiment, round(rel_scores[0], 4), round(sim_scores[0], 4)

    # ─── Embedding inference (LaBSE) ────────────────────────────────

    def embed(self, text: str) -> list[float]:
        """Single text embedding via LaBSE."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embeddings via LaBSE with mean pooling + L2 normalization."""
        encoded = self.embed_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            outputs = self.embed_model(**encoded)
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
    title="Prod Radar ML Service v2",
    description="Multi-task sentiment + relevance + similarity + LaBSE embeddings",
    version="2.0.0",
    lifespan=lifespan,
)


def _require_ready():
    """Raise 503 if models are not loaded yet."""
    if not models.ready:
        raise HTTPException(status_code=503, detail="Models not loaded yet")


# ─── 1. GET /health ──────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    t0 = time.time()
    latency = 0.0
    if models.ready:
        # Quick inference probe
        models.predict_sentiment("тест")
        latency = (time.time() - t0) * 1000

    return HealthResponse(
        status="ok" if models.ready else "loading",
        models_loaded=models.ready,
        device=DEVICE,
        multitask_model=str(MULTITASK_MODEL_DIR),
        embedding_model=EMBEDDING_MODEL_NAME,
        latency_ms=round(latency, 1),
    )


# ─── 2. POST /analyze ────────────────────────────────────────────────

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """Combined endpoint: sentiment + relevance + similarity + embedding in one call."""
    _require_ready()

    sentiment, relevance, similarity = models.predict_all(req.text)
    emb = models.embed(req.text)

    return AnalyzeResponse(
        sentiment_label=sentiment.label,
        sentiment_score=sentiment.score,
        relevance_score=relevance,
        similarity_score=similarity,
        embedding=emb,
    )


# ─── 3. POST /sentiment ──────────────────────────────────────────────

@app.post("/sentiment", response_model=SentimentResponse)
def sentiment(req: TextRequest):
    _require_ready()
    return models.predict_sentiment(req.text)


# ─── 4. POST /sentiment/batch ────────────────────────────────────────

@app.post("/sentiment/batch", response_model=BatchSentimentResponse)
def sentiment_batch(req: BatchSentimentRequest):
    _require_ready()
    if len(req.items) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(req.items)} exceeds limit of {MAX_BATCH_SIZE}",
        )

    texts = [item.text for item in req.items]
    results = models.predict_sentiment_batch(texts)
    return BatchSentimentResponse(
        results=[
            {"id": item.id, "label": r.label, "score": r.score}
            for item, r in zip(req.items, results)
        ]
    )


# ─── 5. POST /relevance ──────────────────────────────────────────────

@app.post("/relevance", response_model=RelevanceResponse)
def relevance(req: RelevanceRequest):
    """Relevance scoring: model score + keyword matching."""
    _require_ready()

    rel_score = models.predict_relevance(req.text)

    # Keyword matching
    text_lower = req.text.lower()
    matched_keywords = [kw for kw in req.keywords if kw.lower() in text_lower]

    # Boost relevance if keywords match
    if matched_keywords:
        keyword_boost = min(0.3, 0.1 * len(matched_keywords))
        rel_score = min(1.0, rel_score + keyword_boost)

    is_relevant = rel_score >= 0.5

    return RelevanceResponse(
        relevance_score=round(rel_score, 4),
        is_relevant=is_relevant,
        matched_keywords=matched_keywords,
    )


# ─── 6. POST /embedding ──────────────────────────────────────────────

@app.post("/embedding", response_model=EmbeddingResponse)
def embedding(req: EmbeddingRequest):
    _require_ready()
    vec = models.embed(req.text)
    return EmbeddingResponse(embedding=vec)


# ─── 7. POST /embedding/batch ────────────────────────────────────────

@app.post("/embedding/batch", response_model=BatchEmbeddingResponse)
def embedding_batch(req: BatchEmbeddingRequest):
    _require_ready()
    if len(req.items) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(req.items)} exceeds limit of {MAX_BATCH_SIZE}",
        )

    texts = [item.text for item in req.items]
    vecs = models.embed_batch(texts)
    return BatchEmbeddingResponse(
        results=[
            {"id": item.id, "embedding": vec}
            for item, vec in zip(req.items, vecs)
        ]
    )


# ─── 8. POST /classify-risk ──────────────────────────────────────────

@app.post("/classify-risk", response_model=RiskResponse)
def risk(req: RiskRequest):
    _require_ready()
    return classify_risk(req.text, req.risk_words, models)


# ─── 9. POST /analyze/detailed ───────────────────────────────────────

@app.post("/analyze/detailed", response_model=DetailedAnalyzeResponse)
def analyze_detailed(req: DetailedAnalyzeRequest):
    """Perplexity-style detailed analysis: overall + per-sentence + risk."""
    _require_ready()

    # Overall: sentiment + relevance + similarity in one pass
    overall_sentiment, overall_relevance, overall_similarity = models.predict_all(
        req.text
    )
    emb = models.embed(req.text)

    # Split into sentences and classify each
    raw_sentences = _split_sentences(req.text)
    sentence_sentiments = models.predict_sentiment_batch(raw_sentences)

    sentences = []
    pos_count = neg_count = neu_count = 0
    most_pos = most_neg = None

    for i, (sent_text, sent_result) in enumerate(
        zip(raw_sentences, sentence_sentiments)
    ):
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
        sentiment_label=overall_sentiment.label,
        sentiment_score=overall_sentiment.score,
        embedding=emb,
        relevance_score=overall_relevance,
        similarity_score=overall_similarity,
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


# ─── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
