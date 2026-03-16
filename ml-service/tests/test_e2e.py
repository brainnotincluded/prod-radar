"""
End-to-end tests for Prod Radar ML Service.
Tests all endpoints against the running service.

Usage:
    # Against local/docker service:
    python3 -m pytest tests/test_e2e.py -v

    # Against specific host:
    ML_URL=http://84.252.140.233:8000 python3 -m pytest tests/test_e2e.py -v
"""

import os
import requests
import pytest

ML_URL = os.environ.get("ML_URL", "http://localhost:8000")


# ─── Health ───────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_ok(self):
        r = requests.get(f"{ML_URL}/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["models_loaded"] is True

    def test_health_has_latency(self):
        r = requests.get(f"{ML_URL}/health").json()
        assert "latency_ms" in r
        assert r["latency_ms"] > 0


# ─── Sentiment ────────────────────────────────────────────────────────

class TestSentiment:
    def _predict(self, text):
        r = requests.post(f"{ML_URL}/sentiment", json={"text": text, "lang": "ru"})
        assert r.status_code == 200
        return r.json()

    # --- Clear sentiment ---

    def test_positive_praise(self):
        r = self._predict("Отличный банк, пользуюсь 5 лет!")
        assert r["label"] == "positive"
        assert r["score"] > 0.5

    def test_positive_thanks(self):
        r = self._predict("Спасибо поддержке, решили за 5 минут")
        assert r["label"] == "positive"

    def test_negative_complaint(self):
        r = self._predict("Ужасный сервис, деньги списали без причины!")
        assert r["label"] == "negative"
        assert r["score"] > 0.5

    def test_negative_broken(self):
        r = self._predict("Третий день не работает приложение")
        assert r["label"] == "negative"

    def test_negative_support(self):
        r = self._predict("Поддержка не отвечает уже неделю")
        assert r["label"] == "negative"

    def test_neutral_question(self):
        r = self._predict("Как подключить NFC на андроиде?")
        assert r["label"] == "neutral"

    def test_neutral_stocks(self):
        r = self._predict("Акции Сбера выросли на 2.3 процента")
        assert r["label"] == "neutral"

    # --- Sarcasm detection (Phase 2) ---

    def test_sarcasm_thanks_emoji(self):
        """Sarcastic 'thanks' with upside-down emoji should be negative."""
        r = self._predict("Ну спасибо Сбер, очень помогли 🙃")
        assert r["label"] == "negative"

    def test_sarcasm_great_work(self):
        """Sarcastic praise with negative fact should be negative."""
        r = self._predict("Отличная работа МТС, 5 часов без связи, так держать!")
        assert r["label"] == "negative"

    def test_sarcasm_bravo(self):
        """Sarcastic bravo with commission complaint should be negative."""
        r = self._predict("Браво, ещё одну комиссию придумали 👏")
        assert r["label"] == "negative"

    def test_sarcasm_broken(self):
        """Wonderful + everything broke = sarcasm = negative."""
        r = self._predict("Замечательно, опять всё сломалось")
        assert r["label"] == "negative"

    def test_sarcasm_timeliness(self):
        """Sarcastic thanks for 'speed' after 2 months wait."""
        r = self._predict("Спасибо за оперативность, всего-то два месяца ждал")
        assert r["label"] == "negative"

    # --- Churn detection (Phase 2) ---

    def test_churn_leaving(self):
        """Threatening to switch providers should be negative."""
        r = self._predict("Всё, ухожу в Т-Банк, достали")
        assert r["label"] == "negative"

    def test_churn_switching(self):
        r = self._predict("Ещё одна такая ситуация и перейду к конкурентам")
        assert r["label"] == "negative"

    # --- Emoji sentiment (Phase 2) ---

    def test_emoji_skull_negative(self):
        """Skull emojis = negative for short posts."""
        r = self._predict("МТС 💀💀💀")
        assert r["label"] == "negative"

    def test_emoji_heart_positive(self):
        """Heart + fire emojis = positive for short posts."""
        r = self._predict("Сбер ❤️🔥")
        assert r["label"] == "positive"

    # --- Passive aggressive (Phase 2) ---

    def test_passive_aggressive_money(self):
        r = self._predict("Ничего страшного что деньги списали, бывает, уже привыкли")
        assert r["label"] == "negative"

    def test_passive_aggressive_update(self):
        r = self._predict("Конечно, обновление прекрасное, если не считать что всё сломалось")
        assert r["label"] == "negative"

    # --- Mixed sentiment ---

    def test_mixed_positive_dominant(self):
        r = self._predict("Дорого, но качество связи отличное")
        assert r["label"] == "positive"

    def test_mixed_negative_dominant(self):
        r = self._predict("Приложение красивое, но тормозит ужасно")
        assert r["label"] == "negative"

    # --- Question complaints ---

    def test_question_complaint(self):
        r = self._predict("Это нормально что связь пропадает каждый час??")
        assert r["label"] == "negative"

    # --- Financial neutral ---

    def test_financial_neutral_revenue(self):
        r = self._predict("Т-Банк показал рост выручки на 40%")
        assert r["label"] == "neutral"

    def test_financial_neutral_dividends(self):
        r = self._predict("Дивиденды Лукойла составили 500 рублей на акцию")
        assert r["label"] == "neutral"

    # --- Score validation ---

    def test_score_range(self):
        """Score must be between 0 and 1."""
        r = self._predict("Тестовое сообщение")
        assert 0.0 <= r["score"] <= 1.0

    def test_label_values(self):
        """Label must be one of the valid values."""
        r = self._predict("Тестовое сообщение")
        assert r["label"] in ("positive", "negative", "neutral")


# ─── Sentiment Batch ──────────────────────────────────────────────────

class TestSentimentBatch:
    def test_batch_returns_results(self):
        items = [
            {"id": "1", "text": "Отличный сервис!"},
            {"id": "2", "text": "Всё плохо"},
            {"id": "3", "text": "Сегодня вторник"},
        ]
        r = requests.post(f"{ML_URL}/sentiment/batch", json={"items": items})
        assert r.status_code == 200
        data = r.json()
        assert len(data["results"]) == 3
        assert data["results"][0]["id"] == "1"
        assert data["results"][1]["id"] == "2"
        assert data["results"][2]["id"] == "3"

    def test_batch_correct_labels(self):
        items = [
            {"id": "pos", "text": "Лучшее приложение в мире!"},
            {"id": "neg", "text": "Ужасный сервис, никому не рекомендую"},
        ]
        r = requests.post(f"{ML_URL}/sentiment/batch", json={"items": items}).json()
        labels = {x["id"]: x["label"] for x in r["results"]}
        assert labels["pos"] == "positive"
        assert labels["neg"] == "negative"


# ─── Embedding ────────────────────────────────────────────────────────

class TestEmbedding:
    def test_embedding_returns_768_dims(self):
        r = requests.post(f"{ML_URL}/embedding", json={"text": "Тест", "lang": "ru"})
        assert r.status_code == 200
        data = r.json()
        assert len(data["embedding"]) == 768

    def test_embedding_is_normalized(self):
        """L2-normalized vectors should have magnitude ~1.0."""
        r = requests.post(f"{ML_URL}/embedding", json={"text": "Тест", "lang": "ru"}).json()
        import math
        magnitude = math.sqrt(sum(x**2 for x in r["embedding"]))
        assert 0.99 < magnitude < 1.01

    def test_similar_texts_close_embeddings(self):
        """Similar texts should have high cosine similarity."""
        e1 = requests.post(f"{ML_URL}/embedding", json={"text": "Отличный банк"}).json()["embedding"]
        e2 = requests.post(f"{ML_URL}/embedding", json={"text": "Хороший банк"}).json()["embedding"]
        e3 = requests.post(f"{ML_URL}/embedding", json={"text": "Погода сегодня солнечная"}).json()["embedding"]

        def cosine(a, b):
            dot = sum(x*y for x,y in zip(a, b))
            return dot  # already normalized

        sim_similar = cosine(e1, e2)
        sim_different = cosine(e1, e3)
        assert sim_similar > sim_different


# ─── Embedding Batch ──────────────────────────────────────────────────

class TestEmbeddingBatch:
    def test_batch_returns_correct_count(self):
        items = [
            {"id": "a", "text": "Первый текст"},
            {"id": "b", "text": "Второй текст"},
        ]
        r = requests.post(f"{ML_URL}/embedding/batch", json={"items": items})
        assert r.status_code == 200
        data = r.json()
        assert len(data["results"]) == 2
        assert len(data["results"][0]["embedding"]) == 768


# ─── Risk Classification ─────────────────────────────────────────────

class TestRisk:
    def test_risk_with_matching_keywords(self):
        r = requests.post(f"{ML_URL}/classify-risk", json={
            "text": "Против компании подан судебный иск на 500 миллионов",
            "risk_words": ["судебный иск", "штраф", "скандал"],
        })
        assert r.status_code == 200
        data = r.json()
        assert data["is_risk"] is True
        assert "судебный иск" in data["matched"]
        assert data["confidence"] > 0.4

    def test_no_risk_without_keywords(self):
        r = requests.post(f"{ML_URL}/classify-risk", json={
            "text": "Компания открыла новый офис в Москве",
            "risk_words": ["судебный иск", "штраф", "скандал"],
        }).json()
        assert r["is_risk"] is False
        assert len(r["matched"]) == 0

    def test_risk_multiple_matches(self):
        r = requests.post(f"{ML_URL}/classify-risk", json={
            "text": "Скандал: мошенничество и штраф для банка",
            "risk_words": ["скандал", "мошенничество", "штраф"],
        }).json()
        assert r["is_risk"] is True
        assert len(r["matched"]) >= 2


# ─── Analyze (combined endpoint for enricher) ────────────────────────

class TestAnalyze:
    def test_analyze_returns_all_fields(self):
        r = requests.post(f"{ML_URL}/analyze", json={
            "text": "Ужасный сервис, деньги списали"
        })
        assert r.status_code == 200
        data = r.json()
        assert "sentiment_label" in data
        assert "sentiment_score" in data
        assert "embedding" in data
        assert data["sentiment_label"] in ("positive", "negative", "neutral")
        assert 0.0 <= data["sentiment_score"] <= 1.0
        assert len(data["embedding"]) == 768

    def test_analyze_negative_text(self):
        r = requests.post(f"{ML_URL}/analyze", json={
            "text": "Ужасный сервис, деньги списали без причины!"
        }).json()
        assert r["sentiment_label"] == "negative"

    def test_analyze_sarcasm_detected(self):
        """Enricher should receive correct negative label for sarcasm."""
        r = requests.post(f"{ML_URL}/analyze", json={
            "text": "Ну спасибо Сбер, очень помогли 🙃"
        }).json()
        assert r["sentiment_label"] == "negative"

    def test_analyze_embedding_is_768(self):
        r = requests.post(f"{ML_URL}/analyze", json={
            "text": "Тестовое сообщение"
        }).json()
        assert len(r["embedding"]) == 768


# ─── Edge Cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_text(self):
        """Empty text should return 422 or a neutral response."""
        r = requests.post(f"{ML_URL}/sentiment", json={"text": "", "lang": "ru"})
        # Either validation error or neutral
        assert r.status_code in (200, 422)

    def test_very_long_text(self):
        """Long text should be truncated, not crash."""
        long_text = "Ужасный сервис. " * 500
        r = requests.post(f"{ML_URL}/sentiment", json={"text": long_text, "lang": "ru"})
        assert r.status_code == 200

    def test_emoji_only(self):
        """Emoji-only input should not crash."""
        r = requests.post(f"{ML_URL}/sentiment", json={"text": "😊😊😊", "lang": "ru"})
        assert r.status_code == 200

    def test_special_characters(self):
        """Text with special chars should not crash."""
        r = requests.post(f"{ML_URL}/sentiment", json={
            "text": "Тест <script>alert('xss')</script> & 'quotes' \"double\"",
            "lang": "ru"
        })
        assert r.status_code == 200

    def test_concurrent_requests(self):
        """Multiple simultaneous requests should all succeed."""
        import concurrent.futures
        texts = [f"Тестовое сообщение номер {i}" for i in range(10)]

        def call(text):
            return requests.post(f"{ML_URL}/sentiment", json={"text": text, "lang": "ru"})

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            results = list(pool.map(call, texts))

        assert all(r.status_code == 200 for r in results)
