# Prod Radar ML Service v2 — API Documentation

> Multi-task ML-сервис для анализа русскоязычных упоминаний брендов в социальных сетях и СМИ.
> Sentiment + Relevance + Similarity + Embeddings + Risk Detection в одном сервисе.

---

## Содержание

- [Архитектура](#архитектура)
- [Информация о модели](#информация-о-модели)
- [Метрики производительности](#метрики-производительности)
- [Эндпоинты](#эндпоинты)
  - [GET /health](#get-health)
  - [POST /analyze](#post-analyze)
  - [POST /sentiment](#post-sentiment)
  - [POST /sentiment/batch](#post-sentimentbatch)
  - [POST /relevance](#post-relevance)
  - [POST /embedding](#post-embedding)
  - [POST /embedding/batch](#post-embeddingbatch)
  - [POST /classify-risk](#post-classify-risk)
  - [POST /analyze/detailed](#post-analyzedetailed)
- [Коды ошибок](#коды-ошибок)
- [Развертывание](#развертывание)
- [Клиент TypeScript](#клиент-typescript)
- [Клиент Python](#клиент-python)

---

## Архитектура

```
                        HTTP :8000
                            |
                    ┌───────▼───────┐
                    │   FastAPI     │
                    │   (uvicorn)   │
                    └───────┬───────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
     ┌────────▼──────┐ ┌───▼───┐ ┌───────▼────────┐
     │  Multi-task   │ │ LaBSE │ │  Rule Engine   │
     │  rubert-tiny2 │ │ en-ru │ │  (Phase 2)     │
     │  (29M params) │ │(175M) │ │                │
     └──────┬────────┘ └───┬───┘ └───────┬────────┘
            │              │             │
     ┌──────┴──────┐   ┌──┴──┐   ┌──────┴──────┐
     │ Sentiment   │   │ 768 │   │ Sarcasm     │
     │ (3 класса)  │   │ dim │   │ Churn       │
     │ Relevance   │   │ vec │   │ Emoji       │
     │ (бинарная)  │   └─────┘   └─────────────┘
     │ Similarity  │
     │ (регрессия) │
     └─────────────┘

     ┌──────────────────────────────────────────────┐
     │              Inference Pipeline               │
     │                                              │
     │  Текст ──> Tokenizer ──> Encoder ──> Heads   │
     │              │                ↓              │
     │              │         ┌──────┼──────┐       │
     │              │         │      │      │       │
     │              │      Sent.  Relev. Simil.     │
     │              │       ↓                       │
     │              │   Post-process (Phase 2)      │
     │              │   - Sarcasm detection          │
     │              │   - Churn signals              │
     │              │   - Emoji correction           │
     └──────────────────────────────────────────────┘
```

---

## Информация о модели

### Multi-task модель (основная)

| Параметр | Значение |
|---|---|
| Базовая модель | `cointegrated/rubert-tiny2` |
| Параметров | **29M** |
| Hidden size | 312 |
| Max sequence length | 128 токенов |
| Задачи | Sentiment (3 класса), Relevance (бинарная), Similarity (регрессия) |
| Датасет | 188K образцов (Мобайл_X-Prod.xlsx) |
| Split | 85% train / 7.5% val / 7.5% test |
| Оптимизатор | AdamW (lr=2e-5, weight_decay=0.01) |
| Loss weights | 0.5 * sentiment + 0.35 * relevance + 0.15 * similarity |

### Головы модели (heads)

```
Encoder (rubert-tiny2, 312-dim CLS)
    │
    ├── Sentiment Head:  Linear(312, 3)   → softmax → positive/negative/neutral
    ├── Relevance Head:  Linear(312, 1)   → sigmoid → 0.0-1.0
    └── Similarity Head: Linear(312, 1)   → sigmoid → 0.0-1.0
```

### Модель эмбеддингов

| Параметр | Значение |
|---|---|
| Модель | `cointegrated/LaBSE-en-ru` |
| Параметров | ~175M |
| Размерность | **768** |
| Нормализация | L2 (magnitude = 1.0) |
| Pooling | Mean pooling over token embeddings |
| Применение | Дедупликация, кластеризация, семантический поиск |

### Phase 2: Правила пост-обработки

Поверх нейросетевого предсказания работает rule-based слой:

- **Сарказм**: позитивные слова + негативный контекст + ирон. эмодзи (🙃🤡💩) -> flip positive -> negative
- **Отток (churn)**: паттерны "ухожу в...", "перейду к...", "закрываю счет" -> negative
- **Эмодзи**: для коротких постов (<20 символов) эмодзи определяют тональность (💀😤 -> negative, ❤️😊 -> positive)

---

## Метрики производительности

### Качество модели (test set)

| Задача | Метрика | Значение |
|---|---|---|
| Sentiment | F1-macro | **0.777** |
| Sentiment | F1-positive | ~0.82 |
| Sentiment | F1-negative | ~0.78 |
| Sentiment | F1-neutral | ~0.73 |
| Relevance | F1 | **0.979** (98%) |
| Similarity | MSE | **0.004** |

### Скорость

| Операция | Latency (CPU) | Latency (CUDA) |
|---|---|---|
| Sentiment (single) | ~5ms | ~3ms |
| Sentiment (batch x10) | ~8ms | ~5ms |
| Embedding (single) | ~10ms | ~6ms |
| Health check | ~5ms | ~4ms |
| Startup (cold) | ~15s | ~8s |

### Лимиты

| Параметр | Значение |
|---|---|
| Max batch size | 128 элементов |
| Max sequence length | 512 токенов (truncation) |
| Concurrent requests | Ограничено uvicorn workers |

---

## Эндпоинты

Базовый URL: `http://localhost:8000`

---

### GET /health

Проверка состояния сервиса. Используется как liveness/readiness probe в Docker.

**Ответ:**

| Поле | Тип | Описание |
|---|---|---|
| `status` | `string` | `"ok"` или `"loading"` |
| `models_loaded` | `boolean` | Загружены ли модели |
| `device` | `string` | `"cuda"` или `"cpu"` |
| `latency_ms` | `float` | Время одного inference в миллисекундах |

**curl:**

```bash
curl http://localhost:8000/health
```

**Ответ (200):**

```json
{
  "status": "ok",
  "models_loaded": true,
  "device": "cuda",
  "latency_ms": 3.9
}
```

**Коды ответа:**

| Код | Описание |
|---|---|
| 200 | Сервис доступен (проверяйте `models_loaded` для readiness) |

---

### POST /analyze

Комбинированный эндпоинт: sentiment + embedding в одном запросе. Основной эндпоинт для enricher-сервиса.

**Запрос:**

| Поле | Тип | Обязательное | Описание |
|---|---|---|---|
| `text` | `string` | да | Текст для анализа |

**Ответ:**

| Поле | Тип | Описание |
|---|---|---|
| `sentiment_label` | `string` | `"positive"`, `"negative"` или `"neutral"` |
| `sentiment_score` | `float` | Уверенность 0.0-1.0 |
| `embedding` | `float[]` | 768-мерный L2-нормализованный вектор |

**curl:**

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Отличный сервис, спасибо за оперативную поддержку!"}'
```

**Ответ (200):**

```json
{
  "sentiment_label": "positive",
  "sentiment_score": 0.9234,
  "embedding": [0.012, -0.034, 0.051, "... (768 floats)"]
}
```

**Коды ответа:**

| Код | Описание |
|---|---|
| 200 | Успешный анализ |
| 422 | Ошибка валидации (отсутствует поле `text`) |
| 503 | Модели еще загружаются |

---

### POST /sentiment

Анализ тональности одного текста.

**Запрос:**

| Поле | Тип | Обязательное | По умолчанию | Описание |
|---|---|---|---|---|
| `text` | `string` | да | — | Текст для анализа |
| `lang` | `string` | нет | `"ru"` | Язык текста |

**Ответ:**

| Поле | Тип | Описание |
|---|---|---|
| `label` | `string` | `"positive"`, `"negative"` или `"neutral"` |
| `score` | `float` | Уверенность 0.0-1.0 |

**curl:**

```bash
curl -X POST http://localhost:8000/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "Ужасный сервис, деньги списали без причины!", "lang": "ru"}'
```

**Ответ (200):**

```json
{
  "label": "negative",
  "score": 0.9712
}
```

**curl (сарказм):**

```bash
curl -X POST http://localhost:8000/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "Ну спасибо Сбер, очень помогли 🙃"}'
```

```json
{
  "label": "negative",
  "score": 0.85
}
```

**Коды ответа:**

| Код | Описание |
|---|---|
| 200 | Успешный анализ |
| 422 | Ошибка валидации |
| 503 | Модели еще загружаются |

---

### POST /sentiment/batch

Пакетный анализ тональности. Максимум **128** элементов.

**Запрос:**

| Поле | Тип | Обязательное | Описание |
|---|---|---|---|
| `items` | `BatchItem[]` | да | Массив элементов |

**BatchItem:**

| Поле | Тип | Обязательное | Описание |
|---|---|---|---|
| `id` | `string` | да | Уникальный идентификатор |
| `text` | `string` | да | Текст для анализа |

**Ответ:**

| Поле | Тип | Описание |
|---|---|---|
| `results` | `object[]` | Массив результатов с полями `id`, `label`, `score` |

**curl:**

```bash
curl -X POST http://localhost:8000/sentiment/batch \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"id": "msg-001", "text": "Отличный сервис!"},
      {"id": "msg-002", "text": "Всё сломалось, ничего не работает"},
      {"id": "msg-003", "text": "Как подключить NFC?"}
    ]
  }'
```

**Ответ (200):**

```json
{
  "results": [
    {"id": "msg-001", "label": "positive", "score": 0.92},
    {"id": "msg-002", "label": "negative", "score": 0.88},
    {"id": "msg-003", "label": "neutral", "score": 0.71}
  ]
}
```

**Коды ответа:**

| Код | Описание |
|---|---|
| 200 | Успешный анализ |
| 400 | Размер batch превышает лимит 128 |
| 422 | Ошибка валидации |
| 503 | Модели еще загружаются |

---

### POST /relevance

Оценка релевантности текста с учетом ключевых слов. Определяет, относится ли упоминание к целевому бренду/продукту.

> **Примечание:** Этот эндпоинт является частью новой multi-task модели v2. Relevance head выдает бинарную классификацию (relevant / irrelevant) с вероятностью 0.0-1.0.

**Запрос:**

| Поле | Тип | Обязательное | Описание |
|---|---|---|---|
| `text` | `string` | да | Текст упоминания |
| `keywords` | `string[]` | нет | Ключевые слова проекта для контекста |

**Ответ:**

| Поле | Тип | Описание |
|---|---|---|
| `is_relevant` | `boolean` | Релевантен ли текст (threshold >= 0.5) |
| `relevance_score` | `float` | Вероятность релевантности 0.0-1.0 |
| `keyword_matches` | `string[]` | Найденные ключевые слова из текста |

**curl:**

```bash
curl -X POST http://localhost:8000/relevance \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Сбербанк запустил новый сервис денежных переводов",
    "keywords": ["Сбербанк", "Сбер", "SberBank"]
  }'
```

**Ответ (200):**

```json
{
  "is_relevant": true,
  "relevance_score": 0.97,
  "keyword_matches": ["Сбербанк"]
}
```

**Коды ответа:**

| Код | Описание |
|---|---|
| 200 | Успешная оценка |
| 422 | Ошибка валидации |
| 503 | Модели еще загружаются |

---

### POST /embedding

Генерация 768-мерного вектора текста. Используется для дедупликации, кластеризации и семантического поиска.

**Запрос:**

| Поле | Тип | Обязательное | По умолчанию | Описание |
|---|---|---|---|---|
| `text` | `string` | да | — | Текст для векторизации |
| `lang` | `string` | нет | `"ru"` | Язык текста |

**Ответ:**

| Поле | Тип | Описание |
|---|---|---|
| `embedding` | `float[]` | 768-мерный L2-нормализованный вектор (magnitude = 1.0) |

**curl:**

```bash
curl -X POST http://localhost:8000/embedding \
  -H "Content-Type: application/json" \
  -d '{"text": "Лукойл повысил дивиденды", "lang": "ru"}'
```

**Ответ (200):**

```json
{
  "embedding": [0.0123, -0.0345, 0.0567, "... (768 floats, L2-normalized)"]
}
```

**Коды ответа:**

| Код | Описание |
|---|---|
| 200 | Успешная генерация |
| 422 | Ошибка валидации |
| 503 | Модели еще загружаются |

---

### POST /embedding/batch

Пакетная генерация эмбеддингов. Максимум **128** элементов.

**Запрос:**

| Поле | Тип | Обязательное | Описание |
|---|---|---|---|
| `items` | `BatchItem[]` | да | Массив элементов (`id` + `text`) |

**Ответ:**

| Поле | Тип | Описание |
|---|---|---|
| `results` | `object[]` | Массив с полями `id` и `embedding` |

**curl:**

```bash
curl -X POST http://localhost:8000/embedding/batch \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"id": "doc-1", "text": "Сбербанк запустил новый сервис"},
      {"id": "doc-2", "text": "МТС обновил тарифную линейку"}
    ]
  }'
```

**Ответ (200):**

```json
{
  "results": [
    {"id": "doc-1", "embedding": [0.012, -0.034, "... (768 floats)"]},
    {"id": "doc-2", "embedding": [-0.005, 0.021, "... (768 floats)"]}
  ]
}
```

**Коды ответа:**

| Код | Описание |
|---|---|
| 200 | Успешная генерация |
| 400 | Размер batch превышает лимит 128 |
| 422 | Ошибка валидации |
| 503 | Модели еще загружаются |

---

### POST /classify-risk

Гибридное определение рисков: keyword matching + сигнал тональности.

**Запрос:**

| Поле | Тип | Обязательное | Описание |
|---|---|---|---|
| `text` | `string` | да | Текст для анализа |
| `risk_words` | `string[]` | да | Список ключевых слов-индикаторов риска |

**Ответ:**

| Поле | Тип | Описание |
|---|---|---|
| `is_risk` | `boolean` | Обнаружен ли риск (`confidence >= 0.4`) |
| `matched` | `string[]` | Найденные рисковые слова |
| `confidence` | `float` | Уверенность 0.0-1.0 |

**Логика расчета confidence:**

| Условие | Диапазон |
|---|---|
| Keywords + negative sentiment | 0.60-0.95 |
| Keywords only | 0.40-0.85 |
| Negative sentiment only (score > 0.8) | ~0.50 |
| Ничего не найдено | ~0.10 |

**curl:**

```bash
curl -X POST http://localhost:8000/classify-risk \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Против компании подан судебный иск на 500 миллионов рублей",
    "risk_words": ["судебный иск", "штраф", "скандал", "мошенничество", "банкротство"]
  }'
```

**Ответ (200):**

```json
{
  "is_risk": true,
  "matched": ["судебный иск"],
  "confidence": 0.92
}
```

**curl (без риска):**

```bash
curl -X POST http://localhost:8000/classify-risk \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Компания открыла новый офис в Москве",
    "risk_words": ["судебный иск", "штраф", "скандал"]
  }'
```

```json
{
  "is_risk": false,
  "matched": [],
  "confidence": 0.1
}
```

**Коды ответа:**

| Код | Описание |
|---|---|
| 200 | Успешный анализ |
| 422 | Ошибка валидации |
| 503 | Модели еще загружаются |

---

### POST /analyze/detailed

Детальный анализ в стиле Perplexity: общая тональность + разбивка по предложениям + выделение ключевых фрагментов.

**Запрос:**

| Поле | Тип | Обязательное | По умолчанию | Описание |
|---|---|---|---|---|
| `text` | `string` | да | — | Текст для анализа |
| `risk_words` | `string[]` | нет | `[]` | Слова-индикаторы риска |

**Ответ:**

| Поле | Тип | Описание |
|---|---|---|
| `sentiment_label` | `string` | Общая тональность |
| `sentiment_score` | `float` | Уверенность 0.0-1.0 |
| `embedding` | `float[]` | 768-мерный вектор |
| `sentences` | `SentenceAnalysis[]` | Разбивка по предложениям |
| `sentence_count` | `int` | Количество предложений |
| `positive_count` | `int` | Количество позитивных предложений |
| `negative_count` | `int` | Количество негативных предложений |
| `neutral_count` | `int` | Количество нейтральных предложений |
| `risk_detected` | `boolean` | Обнаружены ли рисковые слова |
| `risk_words_matched` | `string[]` | Найденные рисковые слова |
| `most_positive` | `SentenceAnalysis \| null` | Самое позитивное предложение |
| `most_negative` | `SentenceAnalysis \| null` | Самое негативное предложение |

**SentenceAnalysis:**

| Поле | Тип | Описание |
|---|---|---|
| `text` | `string` | Текст предложения |
| `label` | `string` | Тональность предложения |
| `score` | `float` | Уверенность 0.0-1.0 |
| `index` | `int` | Порядковый номер предложения (0-based) |

**curl:**

```bash
curl -X POST http://localhost:8000/analyze/detailed \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Приложение Сбера обновилось. Дизайн стал красивым и современным. Но переводы теперь тормозят. Поддержка не помогла решить проблему.",
    "risk_words": ["сбой", "утечка", "мошенничество"]
  }'
```

**Ответ (200):**

```json
{
  "sentiment_label": "negative",
  "sentiment_score": 0.78,
  "embedding": [0.012, -0.034, "... (768 floats)"],
  "sentences": [
    {"text": "Приложение Сбера обновилось.", "label": "neutral", "score": 0.65, "index": 0},
    {"text": "Дизайн стал красивым и современным.", "label": "positive", "score": 0.91, "index": 1},
    {"text": "Но переводы теперь тормозят.", "label": "negative", "score": 0.87, "index": 2},
    {"text": "Поддержка не помогла решить проблему.", "label": "negative", "score": 0.82, "index": 3}
  ],
  "sentence_count": 4,
  "positive_count": 1,
  "negative_count": 2,
  "neutral_count": 1,
  "risk_detected": false,
  "risk_words_matched": [],
  "most_positive": {"text": "Дизайн стал красивым и современным.", "label": "positive", "score": 0.91, "index": 1},
  "most_negative": {"text": "Но переводы теперь тормозят.", "label": "negative", "score": 0.87, "index": 2}
}
```

**Коды ответа:**

| Код | Описание |
|---|---|
| 200 | Успешный анализ |
| 422 | Ошибка валидации |
| 503 | Модели еще загружаются |

---

## Коды ошибок

| HTTP код | Значение | Описание |
|---|---|---|
| **200** | OK | Успешный запрос |
| **400** | Bad Request | Некорректный запрос (batch size > 128) |
| **422** | Unprocessable Entity | Ошибка валидации Pydantic (отсутствуют обязательные поля, неверные типы) |
| **503** | Service Unavailable | Модели еще загружаются (при холодном старте ~8-15 секунд) |

**Формат ошибки 422:**

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "text"],
      "msg": "Field required",
      "input": {}
    }
  ]
}
```

**Формат ошибки 400:**

```json
{
  "detail": "Batch size exceeds limit of 128"
}
```

**Формат ошибки 503:**

```json
{
  "detail": "Models not loaded yet"
}
```

---

## Развертывание

### Docker (CPU)

```bash
cd ml-service

# Сборка
docker build -f Dockerfile.cpu -t prod-radar-ml:cpu .

# Запуск
docker run -d \
  --name ml-service \
  -p 8000:8000 \
  -v ml-models-cache:/root/.cache/huggingface \
  prod-radar-ml:cpu
```

### Docker (CUDA/GPU)

```bash
cd ml-service

# Сборка
docker build -f Dockerfile -t prod-radar-ml:gpu .

# Запуск
docker run -d \
  --name ml-service \
  --gpus all \
  -p 8000:8000 \
  -v ml-models-cache:/root/.cache/huggingface \
  prod-radar-ml:gpu
```

### Переменные окружения

| Переменная | Значение по умолчанию | Описание |
|---|---|---|
| `TRANSFORMERS_CACHE` | `/root/.cache/huggingface` | Путь кэша моделей HuggingFace |
| `HF_HOME` | `/root/.cache/huggingface` | Домашняя директория HuggingFace |

### Docker Compose (healthcheck)

```yaml
ml-service:
  build:
    context: ./ml-service
    dockerfile: Dockerfile.cpu
  ports:
    - "8000:8000"
  volumes:
    - ml-models-cache:/root/.cache/huggingface
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
    interval: 30s
    timeout: 10s
    start_period: 60s
    retries: 3
  restart: unless-stopped

volumes:
  ml-models-cache:
```

### Локальный запуск (без Docker)

```bash
cd ml-service
pip install -r requirements.txt
python app.py
# или:
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

### Зависимости (requirements.txt)

```
fastapi==0.115.12
uvicorn[standard]==0.34.2
torch>=2.0.0
transformers>=4.40.0
sentence-transformers>=3.0.0
numpy>=1.24.0
pydantic>=2.0.0
```

---

## Клиент TypeScript

Полный клиент для интеграции с фронтендом или Node.js-бэкендом.

```typescript
// ml-client.ts

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

// ── Types ────────────────────────────────────────────────────────────

interface SentimentResult {
  label: 'positive' | 'negative' | 'neutral';
  score: number;
}

interface BatchItem {
  id: string;
  text: string;
}

interface BatchSentimentResult {
  results: Array<{ id: string; label: string; score: number }>;
}

interface EmbeddingResult {
  embedding: number[];
}

interface BatchEmbeddingResult {
  results: Array<{ id: string; embedding: number[] }>;
}

interface RiskResult {
  is_risk: boolean;
  matched: string[];
  confidence: number;
}

interface AnalyzeResult {
  sentiment_label: string;
  sentiment_score: number;
  embedding: number[];
}

interface SentenceAnalysis {
  text: string;
  label: string;
  score: number;
  index: number;
}

interface DetailedAnalyzeResult {
  sentiment_label: string;
  sentiment_score: number;
  embedding: number[];
  sentences: SentenceAnalysis[];
  sentence_count: number;
  positive_count: number;
  negative_count: number;
  neutral_count: number;
  risk_detected: boolean;
  risk_words_matched: string[];
  most_positive: SentenceAnalysis | null;
  most_negative: SentenceAnalysis | null;
}

interface HealthResult {
  status: string;
  models_loaded: boolean;
  device: string;
  latency_ms: number;
}

// ── Client Class ─────────────────────────────────────────────────────

class MLServiceClient {
  private baseUrl: string;
  private timeout: number;

  constructor(baseUrl: string = ML_SERVICE_URL, timeoutMs: number = 10000) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.timeout = timeoutMs;
  }

  private async request<T>(path: string, options?: RequestInit): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeout);

    try {
      const res = await fetch(`${this.baseUrl}${path}`, {
        ...options,
        signal: controller.signal,
        headers: { 'Content-Type': 'application/json', ...options?.headers },
      });

      if (!res.ok) {
        const error = await res.text();
        throw new Error(`ML Service error ${res.status}: ${error}`);
      }

      return res.json() as Promise<T>;
    } finally {
      clearTimeout(timer);
    }
  }

  // ── Health ──

  async health(): Promise<HealthResult> {
    return this.request<HealthResult>('/health');
  }

  async waitForReady(maxRetries = 30, intervalMs = 2000): Promise<void> {
    for (let i = 0; i < maxRetries; i++) {
      try {
        const h = await this.health();
        if (h.models_loaded) return;
      } catch {
        // service not ready yet
      }
      await new Promise(r => setTimeout(r, intervalMs));
    }
    throw new Error('ML Service did not become ready in time');
  }

  // ── Sentiment ──

  async sentiment(text: string): Promise<SentimentResult> {
    return this.request<SentimentResult>('/sentiment', {
      method: 'POST',
      body: JSON.stringify({ text, lang: 'ru' }),
    });
  }

  async sentimentBatch(items: BatchItem[]): Promise<BatchSentimentResult> {
    return this.request<BatchSentimentResult>('/sentiment/batch', {
      method: 'POST',
      body: JSON.stringify({ items }),
    });
  }

  // ── Embedding ──

  async embedding(text: string): Promise<number[]> {
    const res = await this.request<EmbeddingResult>('/embedding', {
      method: 'POST',
      body: JSON.stringify({ text, lang: 'ru' }),
    });
    return res.embedding;
  }

  async embeddingBatch(items: BatchItem[]): Promise<BatchEmbeddingResult> {
    return this.request<BatchEmbeddingResult>('/embedding/batch', {
      method: 'POST',
      body: JSON.stringify({ items }),
    });
  }

  // ── Risk ──

  async classifyRisk(text: string, riskWords: string[]): Promise<RiskResult> {
    return this.request<RiskResult>('/classify-risk', {
      method: 'POST',
      body: JSON.stringify({ text, risk_words: riskWords }),
    });
  }

  // ── Analyze ──

  async analyze(text: string): Promise<AnalyzeResult> {
    return this.request<AnalyzeResult>('/analyze', {
      method: 'POST',
      body: JSON.stringify({ text }),
    });
  }

  async analyzeDetailed(text: string, riskWords: string[] = []): Promise<DetailedAnalyzeResult> {
    return this.request<DetailedAnalyzeResult>('/analyze/detailed', {
      method: 'POST',
      body: JSON.stringify({ text, risk_words: riskWords }),
    });
  }

  // ── Utility ──

  /** Compute cosine similarity between two L2-normalized embeddings. */
  static cosineSimilarity(a: number[], b: number[]): number {
    let dot = 0;
    for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
    return dot; // already L2-normalized, so dot product = cosine similarity
  }
}

// ── Usage Examples ───────────────────────────────────────────────────

async function example() {
  const ml = new MLServiceClient('http://localhost:8000');

  // Wait for service readiness
  await ml.waitForReady();

  // Single sentiment
  const sent = await ml.sentiment('Отличный банк, пользуюсь 5 лет!');
  console.log(sent); // { label: 'positive', score: 0.93 }

  // Batch sentiment (for enricher pipeline)
  const batch = await ml.sentimentBatch([
    { id: 'uuid-1', text: 'Лучшее приложение!' },
    { id: 'uuid-2', text: 'Всё сломалось' },
  ]);
  console.log(batch.results);

  // Embeddings for dedup
  const emb1 = await ml.embedding('Отличный банк');
  const emb2 = await ml.embedding('Хороший банк');
  const similarity = MLServiceClient.cosineSimilarity(emb1, emb2);
  console.log(`Similarity: ${similarity}`); // ~0.85+

  // Combined analyze (enricher main call)
  const analysis = await ml.analyze('Ужасный сервис, деньги списали');
  console.log(analysis.sentiment_label, analysis.sentiment_score);

  // Detailed analysis
  const detailed = await ml.analyzeDetailed(
    'Дизайн красивый. Но переводы тормозят. Поддержка не помогла.',
    ['сбой', 'утечка']
  );
  console.log(`Sentences: ${detailed.sentence_count}`);
  console.log(`Positive: ${detailed.positive_count}, Negative: ${detailed.negative_count}`);
  if (detailed.most_negative) {
    console.log(`Worst: "${detailed.most_negative.text}" (${detailed.most_negative.score})`);
  }
}

export { MLServiceClient };
export type {
  SentimentResult,
  BatchItem,
  BatchSentimentResult,
  EmbeddingResult,
  BatchEmbeddingResult,
  RiskResult,
  AnalyzeResult,
  DetailedAnalyzeResult,
  HealthResult,
  SentenceAnalysis,
};
```

---

## Клиент Python

Полный клиент для интеграции с Python-бэкендом или скриптами.

```python
# ml_client.py

"""
Prod Radar ML Service Client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python client for the multi-task ML service.

Usage:
    from ml_client import MLClient

    client = MLClient("http://localhost:8000")
    result = client.sentiment("Отличный сервис!")
    print(result)  # {'label': 'positive', 'score': 0.93}
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ── Data Classes ──────────────────────────────────────────────────────

@dataclass
class SentimentResult:
    label: str  # positive / negative / neutral
    score: float


@dataclass
class RiskResult:
    is_risk: bool
    matched: list[str]
    confidence: float


@dataclass
class AnalyzeResult:
    sentiment_label: str
    sentiment_score: float
    embedding: list[float]


@dataclass
class SentenceAnalysis:
    text: str
    label: str
    score: float
    index: int


@dataclass
class DetailedAnalyzeResult:
    sentiment_label: str
    sentiment_score: float
    embedding: list[float]
    sentences: list[SentenceAnalysis]
    sentence_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    risk_detected: bool
    risk_words_matched: list[str]
    most_positive: Optional[SentenceAnalysis]
    most_negative: Optional[SentenceAnalysis]


@dataclass
class HealthResult:
    status: str
    models_loaded: bool
    device: str
    latency_ms: float


# ── Client ────────────────────────────────────────────────────────────

class MLClient:
    """HTTP client for Prod Radar ML Service."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 10.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        self.session = requests.Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[502, 503, 504],
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def _post(self, path: str, json: dict) -> dict:
        r = self.session.post(
            f"{self.base_url}{path}",
            json=json,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def _get(self, path: str) -> dict:
        r = self.session.get(
            f"{self.base_url}{path}",
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    # ── Health ──

    def health(self) -> HealthResult:
        data = self._get("/health")
        return HealthResult(**data)

    def wait_for_ready(self, max_retries: int = 30, interval: float = 2.0) -> None:
        """Block until the ML service reports models_loaded=True."""
        for _ in range(max_retries):
            try:
                h = self.health()
                if h.models_loaded:
                    return
            except Exception:
                pass
            time.sleep(interval)
        raise TimeoutError("ML Service did not become ready in time")

    # ── Sentiment ──

    def sentiment(self, text: str, lang: str = "ru") -> SentimentResult:
        data = self._post("/sentiment", {"text": text, "lang": lang})
        return SentimentResult(label=data["label"], score=data["score"])

    def sentiment_batch(
        self, items: list[dict[str, str]]
    ) -> list[dict]:
        """
        Args:
            items: List of {"id": "...", "text": "..."} dicts

        Returns:
            List of {"id": "...", "label": "...", "score": float} dicts
        """
        data = self._post("/sentiment/batch", {"items": items})
        return data["results"]

    # ── Embedding ──

    def embedding(self, text: str, lang: str = "ru") -> list[float]:
        data = self._post("/embedding", {"text": text, "lang": lang})
        return data["embedding"]

    def embedding_batch(
        self, items: list[dict[str, str]]
    ) -> list[dict]:
        """
        Args:
            items: List of {"id": "...", "text": "..."} dicts

        Returns:
            List of {"id": "...", "embedding": [...]} dicts
        """
        data = self._post("/embedding/batch", {"items": items})
        return data["results"]

    # ── Risk ──

    def classify_risk(self, text: str, risk_words: list[str]) -> RiskResult:
        data = self._post("/classify-risk", {"text": text, "risk_words": risk_words})
        return RiskResult(
            is_risk=data["is_risk"],
            matched=data["matched"],
            confidence=data["confidence"],
        )

    # ── Analyze ──

    def analyze(self, text: str) -> AnalyzeResult:
        data = self._post("/analyze", {"text": text})
        return AnalyzeResult(
            sentiment_label=data["sentiment_label"],
            sentiment_score=data["sentiment_score"],
            embedding=data["embedding"],
        )

    def analyze_detailed(
        self, text: str, risk_words: list[str] | None = None
    ) -> DetailedAnalyzeResult:
        payload = {"text": text, "risk_words": risk_words or []}
        data = self._post("/analyze/detailed", payload)

        sentences = [
            SentenceAnalysis(**s) for s in data["sentences"]
        ]
        most_pos = SentenceAnalysis(**data["most_positive"]) if data.get("most_positive") else None
        most_neg = SentenceAnalysis(**data["most_negative"]) if data.get("most_negative") else None

        return DetailedAnalyzeResult(
            sentiment_label=data["sentiment_label"],
            sentiment_score=data["sentiment_score"],
            embedding=data["embedding"],
            sentences=sentences,
            sentence_count=data["sentence_count"],
            positive_count=data["positive_count"],
            negative_count=data["negative_count"],
            neutral_count=data["neutral_count"],
            risk_detected=data["risk_detected"],
            risk_words_matched=data["risk_words_matched"],
            most_positive=most_pos,
            most_negative=most_neg,
        )

    # ── Utility ──

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two L2-normalized embeddings."""
        return sum(x * y for x, y in zip(a, b))

    @staticmethod
    def euclidean_distance(a: list[float], b: list[float]) -> float:
        """Euclidean distance between two embeddings."""
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ── Usage Example ─────────────────────────────────────────────────────

if __name__ == "__main__":
    client = MLClient("http://localhost:8000")

    # Wait for service
    print("Waiting for ML service...")
    client.wait_for_ready()
    print("Service ready!")

    # Health
    h = client.health()
    print(f"Status: {h.status}, device: {h.device}, latency: {h.latency_ms}ms")

    # Sentiment
    r = client.sentiment("Отличный банк, пользуюсь 5 лет!")
    print(f"Sentiment: {r.label} ({r.score:.2f})")

    # Sarcasm detection
    r = client.sentiment("Ну спасибо Сбер, очень помогли 🙃")
    print(f"Sarcasm: {r.label} ({r.score:.2f})")

    # Batch sentiment
    batch = client.sentiment_batch([
        {"id": "1", "text": "Лучшее приложение!"},
        {"id": "2", "text": "Всё сломалось"},
        {"id": "3", "text": "Как подключить NFC?"},
    ])
    for item in batch:
        print(f"  {item['id']}: {item['label']} ({item['score']:.2f})")

    # Embedding + similarity
    emb1 = client.embedding("Отличный банк")
    emb2 = client.embedding("Хороший банк")
    emb3 = client.embedding("Погода сегодня солнечная")
    print(f"Similar: {client.cosine_similarity(emb1, emb2):.3f}")
    print(f"Different: {client.cosine_similarity(emb1, emb3):.3f}")

    # Risk
    risk = client.classify_risk(
        "Против компании подан судебный иск на 500 миллионов",
        ["судебный иск", "штраф", "скандал"],
    )
    print(f"Risk: {risk.is_risk}, matched: {risk.matched}, confidence: {risk.confidence:.2f}")

    # Combined analyze
    a = client.analyze("Ужасный сервис, деньги списали")
    print(f"Analyze: {a.sentiment_label} ({a.sentiment_score:.2f}), embedding dim={len(a.embedding)}")

    # Detailed
    d = client.analyze_detailed(
        "Дизайн красивый. Но переводы тормозят. Поддержка не помогла.",
        ["сбой", "утечка"],
    )
    print(f"Sentences: {d.sentence_count}")
    for s in d.sentences:
        print(f"  [{s.index}] {s.label} ({s.score:.2f}): {s.text}")
    if d.most_negative:
        print(f"Worst: {d.most_negative.text}")
```

---

## Маппинг на схему БД

```
mentions.sentiment_label    <- response.label / sentiment_label
mentions.sentiment_score    <- response.score / sentiment_score
mentions.embedding          <- response.embedding          (float[768])
mentions.matched_risk_words <- response.matched            (text[])
mentions.is_relevant        <- relevance_score >= 0.5
mentions.relevance_score    <- response.relevance_score    (float)
```

---

## Запуск тестов

```bash
# Локально (сервис должен быть запущен)
cd ml-service
python -m pytest tests/test_e2e.py -v

# Против удаленного сервера
ML_URL=http://84.252.140.233:8000 python -m pytest tests/test_e2e.py -v
```
