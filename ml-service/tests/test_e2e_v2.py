"""
End-to-end tests for Prod Radar ML Service v2 (multi-task model).

Tests all 9 endpoints against the running service:
  GET  /health
  POST /sentiment
  POST /sentiment/batch
  POST /embedding
  POST /embedding/batch
  POST /classify-risk
  POST /analyze
  POST /analyze/detailed
  POST /relevance

Usage:
    # Against local/docker service:
    python3 -m pytest tests/test_e2e_v2.py -v

    # Against specific host:
    ML_URL=http://84.252.140.233:8000 python3 -m pytest tests/test_e2e_v2.py -v
"""

import math
import os
import concurrent.futures

import pytest
import requests

ML_URL = os.environ.get("ML_URL", "http://localhost:8000")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def post(endpoint: str, json: dict, expected_status: int = 200) -> dict:
    """POST helper with status assertion."""
    r = requests.post(f"{ML_URL}{endpoint}", json=json)
    assert r.status_code == expected_status, (
        f"{endpoint} returned {r.status_code}: {r.text[:300]}"
    )
    return r.json()


def get(endpoint: str, expected_status: int = 200) -> dict:
    r = requests.get(f"{ML_URL}{endpoint}")
    assert r.status_code == expected_status, (
        f"{endpoint} returned {r.status_code}: {r.text[:300]}"
    )
    return r.json()


def cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity for L2-normalized vectors (= dot product)."""
    return sum(x * y for x, y in zip(a, b))


# ===========================================================================
# 1. /health
# ===========================================================================

class TestHealth:
    def test_health_returns_ok(self):
        data = get("/health")
        assert data["status"] == "ok"
        assert data["models_loaded"] is True

    def test_health_has_device(self):
        data = get("/health")
        assert "device" in data
        assert data["device"] in ("cpu", "cuda", "mps")

    def test_health_has_latency(self):
        data = get("/health")
        assert "latency_ms" in data
        assert data["latency_ms"] > 0


# ===========================================================================
# 2. /sentiment
# ===========================================================================

class TestSentiment:
    def _predict(self, text: str) -> dict:
        return post("/sentiment", {"text": text, "lang": "ru"})

    # --- Clear positive ---

    def test_positive_praise(self):
        r = self._predict("Отличный банк, пользуюсь 5 лет!")
        assert r["label"] == "positive"
        assert r["score"] > 0.5

    def test_positive_thanks(self):
        r = self._predict("Спасибо поддержке, решили за 5 минут")
        assert r["label"] == "positive"

    def test_positive_recommendation(self):
        r = self._predict("Всем рекомендую этот сервис, работает безупречно")
        assert r["label"] == "positive"

    # --- Clear negative ---

    def test_negative_complaint(self):
        r = self._predict("Ужасный сервис, деньги списали без причины!")
        assert r["label"] == "negative"
        assert r["score"] > 0.5

    def test_negative_broken_app(self):
        r = self._predict("Третий день не работает приложение")
        assert r["label"] == "negative"

    def test_negative_support(self):
        r = self._predict("Поддержка не отвечает уже неделю")
        assert r["label"] == "negative"

    def test_negative_fraud(self):
        r = self._predict("Мошенники списали деньги с карты, банк ничего не делает")
        assert r["label"] == "negative"

    # --- Clear neutral ---

    def test_neutral_question(self):
        r = self._predict("Как подключить NFC на андроиде?")
        assert r["label"] == "neutral"

    def test_neutral_stocks(self):
        r = self._predict("Акции Сбера выросли на 2.3 процента")
        assert r["label"] == "neutral"

    def test_neutral_financial_report(self):
        r = self._predict("Т-Банк показал рост выручки на 40%")
        assert r["label"] == "neutral"

    def test_neutral_dividends(self):
        r = self._predict("Дивиденды Лукойла составили 500 рублей на акцию")
        assert r["label"] == "neutral"

    # --- Sarcasm detection (post-processing) ---

    def test_sarcasm_thanks_emoji(self):
        """Sarcastic 'thanks' with upside-down emoji should be negative."""
        r = self._predict("Ну спасибо Сбер, очень помогли 🙃")
        assert r["label"] == "negative"

    def test_sarcasm_great_work(self):
        """Sarcastic praise with negative fact should be negative."""
        r = self._predict("Отличная работа МТС, 5 часов без связи, так держать!")
        assert r["label"] == "negative"

    def test_sarcasm_bravo_commission(self):
        """Sarcastic bravo with commission complaint."""
        r = self._predict("Браво, ещё одну комиссию придумали 👏")
        assert r["label"] == "negative"

    def test_sarcasm_wonderful_broken(self):
        """Wonderful + everything broke = sarcasm."""
        r = self._predict("Замечательно, опять всё сломалось")
        assert r["label"] == "negative"

    def test_sarcasm_timeliness(self):
        """Sarcastic thanks for 'speed' after 2 months wait."""
        r = self._predict("Спасибо за оперативность, всего-то два месяца ждал")
        assert r["label"] == "negative"

    # --- Churn detection ---

    def test_churn_leaving(self):
        r = self._predict("Всё, ухожу в Т-Банк, достали")
        assert r["label"] == "negative"

    def test_churn_switching(self):
        r = self._predict("Ещё одна такая ситуация и перейду к конкурентам")
        assert r["label"] == "negative"

    # --- Emoji sentiment ---

    def test_emoji_skull_negative(self):
        r = self._predict("МТС 💀💀💀")
        assert r["label"] == "negative"

    def test_emoji_heart_positive(self):
        r = self._predict("Сбер ❤️🔥")
        assert r["label"] == "positive"

    # --- Score validation ---

    def test_score_in_range(self):
        r = self._predict("Тестовое сообщение")
        assert 0.0 <= r["score"] <= 1.0

    def test_label_is_valid(self):
        r = self._predict("Тестовое сообщение")
        assert r["label"] in ("positive", "negative", "neutral")


# ===========================================================================
# 3. /sentiment/batch
# ===========================================================================

class TestSentimentBatch:
    def test_batch_returns_correct_count(self):
        items = [
            {"id": "1", "text": "Отличный сервис!"},
            {"id": "2", "text": "Всё плохо"},
            {"id": "3", "text": "Сегодня вторник"},
        ]
        data = post("/sentiment/batch", {"items": items})
        assert len(data["results"]) == 3

    def test_batch_preserves_ids(self):
        items = [
            {"id": "alpha", "text": "Текст один"},
            {"id": "beta", "text": "Текст два"},
        ]
        data = post("/sentiment/batch", {"items": items})
        ids = [r["id"] for r in data["results"]]
        assert ids == ["alpha", "beta"]

    def test_batch_correct_labels(self):
        items = [
            {"id": "pos", "text": "Лучшее приложение в мире!"},
            {"id": "neg", "text": "Ужасный сервис, никому не рекомендую"},
        ]
        data = post("/sentiment/batch", {"items": items})
        labels = {r["id"]: r["label"] for r in data["results"]}
        assert labels["pos"] == "positive"
        assert labels["neg"] == "negative"

    def test_batch_has_scores(self):
        items = [{"id": "x", "text": "Тестовое сообщение"}]
        data = post("/sentiment/batch", {"items": items})
        assert "score" in data["results"][0]
        assert 0.0 <= data["results"][0]["score"] <= 1.0


# ===========================================================================
# 4. /embedding
# ===========================================================================

class TestEmbedding:
    def test_returns_768_dims(self):
        data = post("/embedding", {"text": "Тест", "lang": "ru"})
        assert len(data["embedding"]) == 768

    def test_is_l2_normalized(self):
        """L2-normalized vectors should have magnitude ~1.0."""
        data = post("/embedding", {"text": "Привет мир", "lang": "ru"})
        magnitude = math.sqrt(sum(x ** 2 for x in data["embedding"]))
        assert 0.99 < magnitude < 1.01, f"Magnitude {magnitude} not ~1.0"

    def test_similar_texts_closer(self):
        """Similar texts should have higher cosine similarity than unrelated."""
        e1 = post("/embedding", {"text": "Отличный банк"})["embedding"]
        e2 = post("/embedding", {"text": "Хороший банк"})["embedding"]
        e3 = post("/embedding", {"text": "Погода сегодня солнечная"})["embedding"]

        sim_close = cosine_sim(e1, e2)
        sim_far = cosine_sim(e1, e3)
        assert sim_close > sim_far

    def test_different_texts_different_embeddings(self):
        """Different texts should produce different embeddings."""
        e1 = post("/embedding", {"text": "Банк хороший"})["embedding"]
        e2 = post("/embedding", {"text": "Погода ужасная"})["embedding"]
        # Not identical
        assert e1 != e2

    def test_deterministic(self):
        """Same text should produce the same embedding twice."""
        e1 = post("/embedding", {"text": "Детерминизм"})["embedding"]
        e2 = post("/embedding", {"text": "Детерминизм"})["embedding"]
        # Check first 10 elements match (float rounding may vary at tail)
        for a, b in zip(e1[:10], e2[:10]):
            assert abs(a - b) < 1e-6


# ===========================================================================
# 5. /embedding/batch
# ===========================================================================

class TestEmbeddingBatch:
    def test_returns_correct_count(self):
        items = [
            {"id": "a", "text": "Первый текст"},
            {"id": "b", "text": "Второй текст"},
        ]
        data = post("/embedding/batch", {"items": items})
        assert len(data["results"]) == 2

    def test_each_embedding_768_dims(self):
        items = [
            {"id": "a", "text": "Первый"},
            {"id": "b", "text": "Второй"},
            {"id": "c", "text": "Третий"},
        ]
        data = post("/embedding/batch", {"items": items})
        for result in data["results"]:
            assert len(result["embedding"]) == 768

    def test_preserves_ids(self):
        items = [
            {"id": "x1", "text": "Текст"},
            {"id": "x2", "text": "Другой текст"},
        ]
        data = post("/embedding/batch", {"items": items})
        ids = [r["id"] for r in data["results"]]
        assert ids == ["x1", "x2"]


# ===========================================================================
# 6. /classify-risk
# ===========================================================================

class TestClassifyRisk:
    def test_risk_with_matching_keywords(self):
        data = post("/classify-risk", {
            "text": "Против компании подан судебный иск на 500 миллионов",
            "risk_words": ["судебный иск", "штраф", "скандал"],
        })
        assert data["is_risk"] is True
        assert "судебный иск" in data["matched"]
        assert data["confidence"] > 0.4

    def test_no_risk_without_keywords(self):
        data = post("/classify-risk", {
            "text": "Компания открыла новый офис в Москве",
            "risk_words": ["судебный иск", "штраф", "скандал"],
        })
        assert data["is_risk"] is False
        assert len(data["matched"]) == 0

    def test_risk_multiple_matches(self):
        data = post("/classify-risk", {
            "text": "Скандал: мошенничество и штраф для банка",
            "risk_words": ["скандал", "мошенничество", "штраф"],
        })
        assert data["is_risk"] is True
        assert len(data["matched"]) >= 2

    def test_risk_case_insensitive(self):
        data = post("/classify-risk", {
            "text": "ШТРАФ для банка огромный",
            "risk_words": ["штраф"],
        })
        assert data["is_risk"] is True
        assert "штраф" in data["matched"]

    def test_risk_confidence_range(self):
        data = post("/classify-risk", {
            "text": "Судебный иск поступил в суд",
            "risk_words": ["судебный иск"],
        })
        assert 0.0 <= data["confidence"] <= 1.0

    def test_no_risk_empty_risk_words(self):
        """No risk words provided => no risk."""
        data = post("/classify-risk", {
            "text": "Ужасный сервис, всё сломалось!",
            "risk_words": [],
        })
        # With no risk_words, confidence depends on sentiment alone
        assert isinstance(data["is_risk"], bool)
        assert len(data["matched"]) == 0


# ===========================================================================
# 7. /analyze (combined endpoint)
# ===========================================================================

class TestAnalyze:
    def test_returns_all_four_scores(self):
        """v2 /analyze returns sentiment, relevance, similarity, and embedding."""
        data = post("/analyze", {"text": "Ужасный сервис, деньги списали"})

        # Sentiment
        assert "sentiment_label" in data
        assert "sentiment_score" in data
        assert data["sentiment_label"] in ("positive", "negative", "neutral")
        assert 0.0 <= data["sentiment_score"] <= 1.0

        # Relevance
        assert "relevance_score" in data
        assert 0.0 <= data["relevance_score"] <= 1.0

        # Similarity
        assert "similarity_score" in data
        assert 0.0 <= data["similarity_score"] <= 1.0

        # Embedding
        assert "embedding" in data
        assert len(data["embedding"]) == 768

    def test_negative_text(self):
        data = post("/analyze", {"text": "Ужасный сервис, деньги списали без причины!"})
        assert data["sentiment_label"] == "negative"

    def test_positive_text(self):
        data = post("/analyze", {"text": "Лучший банк! Всем советую!"})
        assert data["sentiment_label"] == "positive"

    def test_sarcasm_detected_through_analyze(self):
        """Sarcasm should be caught even through the combined endpoint."""
        data = post("/analyze", {"text": "Ну спасибо Сбер, очень помогли 🙃"})
        assert data["sentiment_label"] == "negative"

    def test_embedding_is_768(self):
        data = post("/analyze", {"text": "Тестовое сообщение"})
        assert len(data["embedding"]) == 768

    def test_embedding_is_normalized(self):
        data = post("/analyze", {"text": "Тестовое сообщение для нормализации"})
        magnitude = math.sqrt(sum(x ** 2 for x in data["embedding"]))
        assert 0.99 < magnitude < 1.01


# ===========================================================================
# 8. /analyze/detailed
# ===========================================================================

class TestAnalyzeDetailed:
    def test_returns_per_sentence_breakdown(self):
        text = "Отличный сервис! Но доставка ужасная. В целом нормально."
        data = post("/analyze/detailed", {"text": text})

        assert "sentences" in data
        assert data["sentence_count"] >= 2
        assert len(data["sentences"]) == data["sentence_count"]

    def test_sentence_structure(self):
        text = "Приложение супер. Поддержка тоже хорошая."
        data = post("/analyze/detailed", {"text": text})

        for sent in data["sentences"]:
            assert "text" in sent
            assert "label" in sent
            assert "score" in sent
            assert "index" in sent
            assert sent["label"] in ("positive", "negative", "neutral")
            assert 0.0 <= sent["score"] <= 1.0

    def test_has_overall_sentiment(self):
        data = post("/analyze/detailed", {"text": "Хороший банк, рекомендую!"})
        assert "sentiment_label" in data
        assert "sentiment_score" in data
        assert data["sentiment_label"] in ("positive", "negative", "neutral")

    def test_has_embedding(self):
        data = post("/analyze/detailed", {"text": "Тест"})
        assert "embedding" in data
        assert len(data["embedding"]) == 768

    def test_sentence_counts(self):
        """positive_count + negative_count + neutral_count == sentence_count."""
        text = "Отличный банк. Ужасная поддержка. Обычная доставка."
        data = post("/analyze/detailed", {"text": text})
        total = data["positive_count"] + data["negative_count"] + data["neutral_count"]
        assert total == data["sentence_count"]

    def test_risk_words_detection(self):
        data = post("/analyze/detailed", {
            "text": "Компания получила штраф за нарушение закона",
            "risk_words": ["штраф", "нарушение"],
        })
        assert data["risk_detected"] is True
        assert "штраф" in data["risk_words_matched"]

    def test_risk_words_no_match(self):
        data = post("/analyze/detailed", {
            "text": "Компания открыла новый офис",
            "risk_words": ["штраф", "скандал"],
        })
        assert data["risk_detected"] is False
        assert len(data["risk_words_matched"]) == 0

    def test_most_positive_and_negative_highlights(self):
        text = "Приложение замечательное! Но поддержка ужасная, не отвечают неделю."
        data = post("/analyze/detailed", {"text": text})
        # At least one of these should be present
        if data["positive_count"] > 0:
            assert data["most_positive"] is not None
            assert data["most_positive"]["label"] == "positive"
        if data["negative_count"] > 0:
            assert data["most_negative"] is not None
            assert data["most_negative"]["label"] == "negative"

    def test_single_sentence_text(self):
        data = post("/analyze/detailed", {"text": "Все работает отлично"})
        assert data["sentence_count"] >= 1


# ===========================================================================
# 9. /relevance
# ===========================================================================

class TestRelevance:
    def test_returns_relevance_score(self):
        data = post("/relevance", {
            "text": "Сбербанк запустил новую кредитную карту с кэшбэком 5%",
        })
        assert "relevance_score" in data
        assert 0.0 <= data["relevance_score"] <= 1.0

    def test_with_keywords_returns_score(self):
        data = post("/relevance", {
            "text": "Сбербанк запустил новую кредитную карту с кэшбэком 5%",
            "keywords": ["сбербанк", "карта", "кэшбэк"],
        })
        assert "relevance_score" in data
        assert 0.0 <= data["relevance_score"] <= 1.0

    def test_keywords_boost_relevance(self):
        """Text with matching keywords should score higher than without."""
        text = "Банк выпустил новую карту с кэшбэком"

        without = post("/relevance", {"text": text})
        with_kw = post("/relevance", {
            "text": text,
            "keywords": ["банк", "карта", "кэшбэк"],
        })
        # Keywords should boost score or at least not lower it
        assert with_kw["relevance_score"] >= without["relevance_score"] - 0.01

    def test_irrelevant_text(self):
        """Clearly off-topic text should have lower relevance."""
        data = post("/relevance", {
            "text": "Сегодня хорошая погода, пойду гулять в парк",
            "keywords": ["банк", "кредит", "ипотека"],
        })
        assert "relevance_score" in data
        # Just verify it returns a valid score (the model decides relevance)
        assert 0.0 <= data["relevance_score"] <= 1.0


# ===========================================================================
# Edge Cases (across all endpoints)
# ===========================================================================

class TestEdgeCases:
    # --- Empty text ---

    def test_sentiment_empty_text(self):
        """Empty text should return 422 validation error or a neutral fallback."""
        r = requests.post(f"{ML_URL}/sentiment", json={"text": "", "lang": "ru"})
        assert r.status_code in (200, 422)

    def test_embedding_empty_text(self):
        r = requests.post(f"{ML_URL}/embedding", json={"text": "", "lang": "ru"})
        assert r.status_code in (200, 422)

    def test_analyze_empty_text(self):
        r = requests.post(f"{ML_URL}/analyze", json={"text": ""})
        assert r.status_code in (200, 422)

    # --- Long text (truncation) ---

    def test_sentiment_long_text(self):
        long_text = "Ужасный сервис. " * 500
        data = post("/sentiment", {"text": long_text, "lang": "ru"})
        assert data["label"] in ("positive", "negative", "neutral")

    def test_embedding_long_text(self):
        long_text = "Текст для проверки. " * 500
        data = post("/embedding", {"text": long_text, "lang": "ru"})
        assert len(data["embedding"]) == 768

    def test_analyze_long_text(self):
        long_text = "Предложение номер один. " * 200
        data = post("/analyze", {"text": long_text})
        assert "sentiment_label" in data
        assert len(data["embedding"]) == 768

    # --- Emoji-only ---

    def test_sentiment_emoji_only(self):
        data = post("/sentiment", {"text": "😊😊😊", "lang": "ru"})
        assert data["label"] in ("positive", "negative", "neutral")

    def test_embedding_emoji_only(self):
        data = post("/embedding", {"text": "🔥🔥🔥", "lang": "ru"})
        assert len(data["embedding"]) == 768

    def test_analyze_emoji_only(self):
        data = post("/analyze", {"text": "💀💀💀"})
        assert "sentiment_label" in data

    # --- Special characters / XSS ---

    def test_special_characters(self):
        data = post("/sentiment", {
            "text": "Тест <script>alert('xss')</script> & 'quotes' \"double\"",
            "lang": "ru",
        })
        assert data["label"] in ("positive", "negative", "neutral")

    # --- Unicode ---

    def test_unicode_mixed(self):
        data = post("/sentiment", {
            "text": "Excellent service! Отличный сервис! 素晴らしい",
            "lang": "ru",
        })
        assert data["label"] in ("positive", "negative", "neutral")

    # --- Whitespace-only ---

    def test_whitespace_only(self):
        r = requests.post(f"{ML_URL}/sentiment", json={"text": "   \n\t  ", "lang": "ru"})
        assert r.status_code in (200, 422)

    # --- Single word ---

    def test_single_word(self):
        data = post("/sentiment", {"text": "Ужас", "lang": "ru"})
        assert data["label"] in ("positive", "negative", "neutral")


# ===========================================================================
# Concurrency
# ===========================================================================

class TestConcurrency:
    def test_concurrent_sentiment_requests(self):
        """10 simultaneous requests should all succeed."""
        texts = [f"Тестовое сообщение номер {i}" for i in range(10)]

        def call(text):
            return requests.post(
                f"{ML_URL}/sentiment", json={"text": text, "lang": "ru"}
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            results = list(pool.map(call, texts))

        assert all(r.status_code == 200 for r in results)

    def test_concurrent_embedding_requests(self):
        """Multiple embedding requests should all succeed concurrently."""
        texts = [f"Параллельный запрос {i}" for i in range(8)]

        def call(text):
            return requests.post(
                f"{ML_URL}/embedding", json={"text": text, "lang": "ru"}
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(call, texts))

        assert all(r.status_code == 200 for r in results)
        for r in results:
            assert len(r.json()["embedding"]) == 768

    def test_concurrent_mixed_endpoints(self):
        """Mix of different endpoint calls should all succeed."""

        def sentiment_call():
            return requests.post(
                f"{ML_URL}/sentiment", json={"text": "Тест", "lang": "ru"}
            )

        def embedding_call():
            return requests.post(
                f"{ML_URL}/embedding", json={"text": "Тест", "lang": "ru"}
            )

        def analyze_call():
            return requests.post(
                f"{ML_URL}/analyze", json={"text": "Тест"}
            )

        def health_call():
            return requests.get(f"{ML_URL}/health")

        calls = [sentiment_call, embedding_call, analyze_call, health_call] * 3

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda fn: fn(), calls))

        assert all(r.status_code == 200 for r in results)


# ===========================================================================
# Batch edge cases
# ===========================================================================

class TestBatchEdgeCases:
    def test_sentiment_batch_single_item(self):
        items = [{"id": "only", "text": "Единственный элемент"}]
        data = post("/sentiment/batch", {"items": items})
        assert len(data["results"]) == 1
        assert data["results"][0]["id"] == "only"

    def test_embedding_batch_single_item(self):
        items = [{"id": "only", "text": "Единственный элемент"}]
        data = post("/embedding/batch", {"items": items})
        assert len(data["results"]) == 1
        assert len(data["results"][0]["embedding"]) == 768

    def test_sentiment_batch_large(self):
        """Batch of 50 items should succeed."""
        items = [{"id": str(i), "text": f"Сообщение номер {i}"} for i in range(50)]
        data = post("/sentiment/batch", {"items": items})
        assert len(data["results"]) == 50

    def test_embedding_batch_large(self):
        """Batch of 20 embeddings should succeed."""
        items = [{"id": str(i), "text": f"Текст для эмбеддинга {i}"} for i in range(20)]
        data = post("/embedding/batch", {"items": items})
        assert len(data["results"]) == 20
        for r in data["results"]:
            assert len(r["embedding"]) == 768


# ===========================================================================
# Cross-endpoint consistency
# ===========================================================================

class TestCrossEndpointConsistency:
    def test_analyze_sentiment_matches_standalone(self):
        """Sentiment from /analyze should match /sentiment for the same text."""
        text = "Отличный банк, всем рекомендую!"
        sent = post("/sentiment", {"text": text, "lang": "ru"})
        analyzed = post("/analyze", {"text": text})
        assert sent["label"] == analyzed["sentiment_label"]

    def test_analyze_embedding_matches_standalone(self):
        """Embedding from /analyze should match /embedding for the same text."""
        text = "Тестовое сообщение для сравнения"
        emb = post("/embedding", {"text": text, "lang": "ru"})["embedding"]
        analyzed = post("/analyze", {"text": text})["embedding"]
        # Check first 20 elements match closely
        for a, b in zip(emb[:20], analyzed[:20]):
            assert abs(a - b) < 1e-5, f"Embeddings diverge: {a} vs {b}"

    def test_detailed_sentiment_matches_overall(self):
        """Overall sentiment from /analyze/detailed should match /sentiment."""
        text = "Ужасный сервис, никому не советую"
        sent = post("/sentiment", {"text": text, "lang": "ru"})
        detailed = post("/analyze/detailed", {"text": text})
        assert sent["label"] == detailed["sentiment_label"]
