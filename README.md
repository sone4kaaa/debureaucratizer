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
| Глоссарий терминов (≥ 500) | 650 пар «канцеляризм → простой эквивалент» (`data/import/glossary.csv`) |
| Translation Memory (≥ 1000) | 5876 пар «сложное предложение → простое» (`data/import/tm.csv`, `tm.json`) |
| Rule-based Validator | Канцеляризмы исчезли (словарь), текст на кириллице, даты/суммы/плейсхолдеры сохранились, нет эха входа, нет дословных фрагментов ≥ 4 слов |
| Critic (LLM-as-a-Judge) | Оценки adequacy / fluency / terminology + overall_score, JSON-парсинг с regex-fallback |
| RAG | Гибридный поиск похожих примеров упрощения (BM25 + Qdrant + RRF k=60) |

## Стек

Python 3.12, FastAPI, SQLAlchemy 2.0 (async), Postgres, Qdrant, Redis + Celery, Ollama (основная LLM, `qwen2.5:7b`) + HuggingFace Transformers (запасная), sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`), rank-bm25, Streamlit, Nginx, Docker Compose, Poetry, pytest (покрытие ≥ 70%).

Мультиагентная система реализована **вручную** — без LangChain / LangGraph / CrewAI.

## Архитектура

```
Streamlit UI (8501) ─┐
9999999Nginx (80) ──────────┴─> FastAPI (8000) ──> Ollama (11434) / HuggingFace
                              │  ├── Validator -> Critic -> Editor (Supervisor)
                              │  ├── RAG: BM25 + Qdrant -> RRF
                              │  ├── Postgres (глоссарий, TM, пользователи, кэш, HITL)
                              │  └── Redis -> Celery worker (индексация, батчи, очистка кэша)
                              └── Qdrant (6333, векторный поиск)
```

Цикл агентов: **Validator** (правила) → **Critic** (оценка 0–10) → при необходимости **Editor** (правки) → повторная валидация/оценка. **Supervisor** управляет адаптивным бюджетом итераций (0–3 по скору критика), откатывает ухудшения (snapshot/rollback) и ведёт `reasoning_log`.

### 8 режимов оценки (Ablation Grid)

| Режим | Глоссарий | TM | Critic | Editor |
|---|---|---|---|---|
| `baseline` | ✗ | — | ✗ | ✗ |
| `glossary` | ✓ | — | ✗ | ✗ |
| `tm` | ✓ | BM25 | ✗ | ✗ |
| `tm_hybrid` | ✓ | hybrid | ✗ | ✗ |
| `agents_only` | ✗ | — | ✓ | ✓ |
| `tm_hybrid_critic` | ✓ | hybrid | ✓ | ✗ |
| `tm_hybrid_editor` | ✓ | hybrid | ✗ | ✓ |
| `full` | ✓ | hybrid | ✓ | ✓ |

## Быстрый старт (Docker)

```bash
cp .env.example .env   # при желании поменять JWT_SECRET_KEY, OLLAMA_MODEL
docker compose -f infrastructure/docker-compose.yml up -d
```

Поднимаются 9 сервисов: `db` (Postgres), `redis`, `qdrant`, `ollama`,
`ollama-pull` (скачивает модель один раз), `app` (API), `worker` (Celery),
`ui` (Streamlit), `nginx`.

- UI: http://localhost:8501
- API через nginx: http://localhost/health
- Swagger UI: http://localhost:8000/docs

После старта загрузите эталонные данные и создайте админа
(выполняется на хосте или в контейнере `app`):

```bash
docker compose -f infrastructure/docker-compose.yml exec app python scripts/import/add_glossary.py --builtin
docker compose -f infrastructure/docker-compose.yml exec app python scripts/import/add_tm.py --builtin
docker compose -f infrastructure/docker-compose.yml exec app python scripts/import/create_admin.py admin@example.com --password 'Secret123'
docker compose -f infrastructure/docker-compose.yml restart app   # TM -> BM25/Qdrant
```

## Локальный запуск (разработка)

Требования: Python 3.12, Poetry, Ollama (`ollama pull qwen2.5:7b`).

```bash
cp .env.example .env                  # по умолчанию SQLite + localhost
python -m poetry install
python -m poetry run python scripts/import/add_glossary.py --builtin
python -m poetry run python scripts/import/add_tm.py --builtin
python -m poetry run python scripts/import/create_admin.py admin@example.com
python -m poetry run uvicorn backend.api.main:app --reload
streamlit run streamlit_app/app.py    # в соседнем терминале
```

## Переменные окружения (.env)

| Переменная | По умолчанию | Описание |
|---|---|---|
| `DEBUG` | `true` | SQL-эхо и подробные логи |
| `DATABASE_URL` | `sqlite:///./data/app.db` | DSN; в compose переопределяется на Postgres |
| `JWT_SECRET_KEY` | `change-me-in-production` | Секрет JWT (**сменить в проде**, `openssl rand -hex 32`); в prod-режиме пустое значение запрещено |
| `JWT_ALGORITHM` / `JWT_EXPIRATION_MINUTES` | `HS256` / `1440` | Параметры токенов |
| `OLLAMA_HOST` / `OLLAMA_PORT` | `localhost` / `11434` | Адрес Ollama |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Основная модель |
| `TRANSLATION_ENGINE` / `AGENT_ENGINE` | `auto` | `auto` \| `ollama` \| `huggingface` |
| `FALLBACK_TO_HF_ON_ERROR` | `true` | Фолбэк на HF при недоступной Ollama |
| `HF_MODEL` / `HF_AGENT_MODEL` | `facebook/nllb-200-distilled-600M` | Запасные HF-модели |
| `HF_TOKEN` | пусто | Токен HuggingFace (для приватных датасетов) |
| `ENSEMBLE_MODELS` | `qwen2.5:7b` | Модели ансамбля через запятую |
| `QDRANT_HOST` / `QDRANT_PORT` / `QDRANT_COLLECTION` | `localhost` / `6333` / `tm_entries` | Векторная БД |
| `REDIS_HOST` / `REDIS_PORT` | `localhost` / `6379` | Redis |
| `CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND` | `redis://localhost:6379/0` (`/1`) | Брокер и бэкенд Celery |
| `EVAL_MODE` | `full` | Режим пайплайна по умолчанию (см. таблицу выше) |
| `CACHEABLE_DOMAIN` | `general` | Только этот домен кэшируется |
| `CORS_ORIGINS` | `http://localhost:8501` | Разрешённые origin'ы |
| `LOG_JSON` | `false` | JSON-формат логов |
| `EVAL_MAX_SRC_LEN` / `EVAL_MAX_TGT_LEN` | `512` / `2000` | Нормализация в prepare_eval_datasets |

## API: примеры запросов

```bash
# Регистрация и вход
curl -X POST http://localhost:8000/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"email":"user@example.com","username":"user1","password":"Secret123"}'

TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"user@example.com","password":"Secret123"}' | python -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')

# Упрощение текста
curl -X POST http://localhost:8000/translate \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"text":"В соответствии с вышеизложенным уведомляем о необходимости осуществления оплаты в срок до 15 марта 2025 года.","eval_mode":"full"}'

# Батч и ансамбль
curl -X POST http://localhost:8000/translate/batch \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"texts":["Текст один.","Текст два."]}'

curl -X POST http://localhost:8000/translate/ensemble \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"text":"Канцелярский текст.","models":["qwen2.5:7b"]}'

# Админка (роль admin): глоссарий, TM, кэш
curl http://localhost:8000/admin/glossary/list -H "Authorization: Bearer $TOKEN"
curl http://localhost:8000/admin/cache/stats  -H "Authorization: Bearer $TOKEN"

# HITL (роль editor/admin)
curl http://localhost:8000/hitl/uncertain -H "Authorization: Bearer $TOKEN"

# Health
curl http://localhost:8000/healthz   # liveness
curl http://localhost:8000/health    # DB + Redis + Qdrant + Ollama
```

## Скрипты

```bash
# Импорт данных (CSV legacy/lang-coded и JSON; --list показывает builtin-наборы)
python scripts/import/add_glossary.py data/import/glossary.csv | --builtin | --list
python scripts/import/add_tm.py       data/import/tm.csv       | --builtin | --list
python scripts/import/create_admin.py admin@example.com [--password ...]
python scripts/import/export_datasets.py            # экспорт builtin в data/import/

# Оценка качества
python scripts/eval/prepare_eval_datasets.py [--hf-dataset NAME]   # data/eval/*.json
python scripts/eval/run_eval.py data/eval/bureau-plain_10.json \
    [--modes baseline full] [--no-neural] [--out reports/eval.json]
```

Датасеты собраны синтетически (LLM-генерация + ручная проверка шаблонов) —
см. `scripts/import/builtin_data.py`; при использовании внешнего датасета
`prepare_eval_datasets.py --hf-dataset` берёт его с HuggingFace (`HF_TOKEN` в env).

## Тестирование

```bash
python -m poetry run pytest tests/ -q \
  --cov=backend --cov-report=term-missing --cov-report=html
```

- unit: агенты (Validator/Critic/Editor/Supervisor), retrieval (BM25/RRF/hybrid),
  хранилища (cache/glossary/TM), HITL, пайплайн, worker-задачи, утилиты;
- integration (`@pytest.mark.integration`): auth, translate/multi/batch/ensemble,
  admin-роуты и роли, HITL API;
- внешние сервисы (Ollama, Qdrant, Redis) в тестах замокированы.

## Структура

```
backend/            FastAPI: agents/ retrieval/ data/ services/ api/ worker/ eval/ hitl/
streamlit_app/      UI: app.py + 8 страниц + utils/api.py
scripts/import/     импорт и экспорт данных, create_admin
scripts/eval/       prepare_eval_datasets, run_eval, mode_labels
tests/              unit/ + integration/
infrastructure/     Dockerfile (multi-stage), docker-compose.yml (9 сервисов), nginx.conf
data/import/        эталонные данные: glossary.csv (650), tm.csv/tm.json (5876)
data/eval/          оценочные датасеты 10..1000 + manifest.json
```

## Скриншоты UI

Скриншоты страниц Streamlit добавляются после локального запуска
(`streamlit run streamlit_app/app.py`) в папку `docs/screenshots/`.

## Ограничения

- Основной движок — локальная Ollama-модель: качество упрощения зависит от неё,
  первый запрос медленный (загрузка в память).
- HF-фолбэк (`nllb-200-distilled-600M`) — межъязыковой переводчик, для
  переноса стиля bureau→plain подходит ограниченно (нужен только как аварийный).
- Глоссарий и TM — синтетические, покрывают типовой канцелярит; для продакшена
  набор нужно расширять реальными парами (например, через HITL-очередь).
