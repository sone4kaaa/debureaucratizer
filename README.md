# Дебюрократизатор (Debureaucratizer)

Интеллектуальная система, которая **переписывает тексты с канцелярского и юридического русского на простой, понятный язык** — с сохранением смысла, фактов, дат, сумм и юридических формулировок.

## Тема проекта

- **Что делает:** принимает текст на «официальном» языке (письма госорганов, договоры, уведомления, предписания) и возвращает его версию простым русским языком.
- **Для кого:** граждане, разбирающие официальные письма и договоры; сотрудники, проверяющие понятность документов для клиентов.
- **Вход:** текст на канцелярите (`bureau`), например: «В соответствии с вышеизложенным уведомляем о необходимости осуществления оплаты в срок до…».
- **Выход:** тот же смысл простым языком (`plain`): «Сообщаем: нужно оплатить до…».

### Маппинг на требования задания

| Компонент задания | Реализация в проекте |
|---|---|
| «Перевод» source → target | Переписывание `bureau` → `plain` (перенос стиля) |
| Глоссарий терминов (≥ 500) | Пары «канцеляризм → простой эквивалент» («осуществлять» → «делать/выполнять») |
| Translation Memory (≥ 1000) | Пары «сложное предложение → простое предложение» |
| Rule-based Validator | Канцеляризмы исчезли (словарь), текст сократился, даты/суммы/имена сохранились (regex), нет эха входа |
| Critic (LLM-as-a-Judge) | Оценки: смысл сохранён (adequacy), естественность (fluency), терминология (terminology) |
| RAG | Гибридный поиск похожих примеров упрощения (BM25 + Qdrant + RRF) |

## Команда

- Студент: все роли (backend / ML-RAG / инфра-QA) в одном лице.

## Стек

Python 3.12, FastAPI, SQLAlchemy 2.0 (async), Postgres, Qdrant, Redis + Celery, Ollama (основная LLM) + HuggingFace Transformers (запасная), sentence-transformers, rank-bm25, Streamlit, Docker Compose, Poetry, pytest (покрытие ≥ 70%).

Мультиагентная система реализована **вручную** — без LangChain / LangGraph / CrewAI.

## Быстрый старт (Docker)

```bash
cp .env.example .env   # заполнить JWT_SECRET_KEY и т.д.
docker compose -f infrastructure/docker-compose.yml up -d
```

## Локальный запуск (разработка)

```bash
python -m poetry install
python -m poetry run uvicorn backend.api.main:app --reload
streamlit run streamlit_app/app.py
```

Переменные окружения — см. `.env.example`.

## Структура

```
backend/          # FastAPI-приложение: агенты, RAG, API, worker
streamlit_app/    # UI (8 страниц)
scripts/          # импорт данных, подготовка и прогон eval
tests/            # unit + integration тесты
infrastructure/   # Dockerfile, docker-compose.yml, nginx.conf
data/import/      # эталонные данные (CSV/JSON)
```
