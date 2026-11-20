# Дебюрократизатор — комплект запуска (kit)

Комплект закрывает всё, чего не хватало в репозитории `sone4kaaa/debureaucratizer`
(коммит `init`): исправляет баг запуска, добавляет импорт эталонных данных,
Streamlit-интерфейс и полный docker-compose с LLM, векторным поиском и воркером.

После установки работать можно **из браузера**, Postman не нужен:
- UI: http://localhost:8501
- Swagger UI API: http://localhost:8000/docs
- Админка кэша/справочников: страница «Админ» в UI (нужна роль admin)

---

## 1. Что сломано в исходном репозитории и что добавляет kit

| Проблема | Решение в kit |
|---|---|
| **API не стартует**: в `backend/data/glossary.py` метод `list` затеняет встроенный `list`, аннотация `list[dict[str, Any]]` падает с `TypeError` при загрузке модуля | `fix_boot.patch` — добавить `from __future__ import annotations` в начало файла |
| Нет ни одной страницы Streamlit: `streamlit run streamlit_app/app.py` падает | Добавлены `app.py` + 8 страниц + клиент API (`streamlit_app/`) |
| `scripts/` пустые: встроенный глоссарий (651 пара) и TM (5876 пар) не заливаются в БД | `scripts/import/import_builtin.py` — идемпотентный импорт + назначение роли admin |
| Регистрация создаёт только `role="user"`, стать админом нечем | Флаг `--admin <username>` в скрипте импорта |
| docker-compose содержит только Postgres + app (без LLM/Qdrant/Redis/UI) | Новый `infrastructure/docker-compose.yml` — 8 сервисов |
| Dockerfile не копирует `streamlit_app/` | Обновлён `infrastructure/Dockerfile` |
| `GLOSSARY_PAIRS` содержит дубликаты по ключу → `UNIQUE constraint` при импорте | Скрипт дедуплицирует перед вставкой |

---

## 2. Локальный запуск (Linux/macOS/WSL, проверено на Python 3.12)

### 2.1 Требования

- Python 3.12
- Poetry (`pipx install poetry` или `pip install poetry`)
- **Ollama** — основной движок перевода: https://ollama.com/download
  - без него запросы уходят в запасной HuggingFace-движок, а модель
    `facebook/nllb-200-distilled-600M` — это межъязыковой переводчик,
    задачу «канцелярит -> простой русский» она не решает;
- ~6 ГБ диска под зависимости (torch и др.) + ~4.7 ГБ под модель qwen2.5:7b.

### 2.2 Шаги

```bash
# 1. Клон и зависимости
git clone https://github.com/sone4kaaa/debureaucratizer.git
cd debureaucratizer

# 2. ФИКС ЗАПУСКА (без него API падает)
#    добавить второй строкой в backend/data/glossary.py:
#      from __future__ import annotations
#    или целиком: git apply fix_boot.patch
git apply /путь/к/kit/fix_boot.patch

# 3. Окружение
cp .env.example .env          # для локальной отладки значения по умолчанию рабочие
python -m poetry install      # несколько минут: torch, transformers и пр.

# 4. Модель Ollama
ollama pull qwen2.5:7b        # или другая: значение OLLAMA_MODEL в .env

# 5. Эталонные данные: глоссарий 650 пар + TM 5876 пар (идемпотентно)
python -m poetry run python scripts/import/import_builtin.py
# (перед импортом остановите API, если он запущен — иначе блокировка SQLite)

# 6. Сделайте себя админом (после регистрации, см. шаг 8)
python -m poetry run python scripts/import/import_builtin.py --admin <username>

# 7. Backend API (терминал 1)
python -m poetry run uvicorn backend.api.main:app --reload

# 8. UI (терминал 2)
python -m poetry run streamlit run streamlit_app/app.py
```

Дальше:
- http://localhost:8501 — интерфейс: вкладка «Вход» -> «Регистрация»;
- затем назначьте админа (шаг 6) и обновите страницу;
- http://localhost:8000/docs — Swagger API (альтернатива Postman);
- http://localhost:8000/healthz — здоровье сервиса.

### 2.3 Переменные окружения

Минимально важные в `.env` (полный список — `.env.example`):

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./data/app.db` | SQLite для разработки; Postgres для Docker |
| `OLLAMA_HOST` / `OLLAMA_PORT` | `localhost` / `11434` | Где стоит Ollama |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Модель-упрощатель |
| `TRANSLATION_ENGINE` | `auto` | `auto` / `ollama` / `huggingface` |
| `EVAL_MODE` | `full` | Режим пайплайна (можно менять per-request) |
| `JWT_SECRET_KEY` | `change-me-in-production` | При `DEBUG=false` обязателен ключ >= 32 симв. |

---

## 3. Запуск в Docker (полный стек одной командой)

Состав: `db` (Postgres 16) + `app` (FastAPI) + `worker` (Celery) +
`ui` (Streamlit) + `ollama` + `ollama-pull` (скачивает модель) +
`qdrant` (векторный поиск) + `redis` (брокер).

```bash
cd debureaucratizer
git apply /путь/к/kit/fix_boot.patch
# скопируйте файлы из kit в репозиторий с сохранением путей (см. раздел 4)
cp .env.example .env
docker compose -f infrastructure/docker-compose.yml up -d
```

- **Первый запуск долгий**: сборка образа (poetry install, ~2-3 ГБ) и скачивание
  модели qwen2.5:7b (~4.7 ГБ) сервисом `ollama-pull`. Прогресс:
  `docker compose -f infrastructure/docker-compose.yml logs -f ollama-pull`.
- Готовность API: `curl http://localhost:8000/healthz`
- UI: http://localhost:8501, Swagger: http://localhost:8000/docs
- Зарегистрируйтесь в UI, затем назначьте админа прямо в Postgres:
  ```bash
  docker compose -f infrastructure/docker-compose.yml exec db \
    psql -U rung_user -d rung_db -c "UPDATE users SET role='admin' WHERE username='<ваш_логин>';"
  ```
  (либо запустите kit-скрипт импорта с `--admin` внутри контейнера `app`).
- Импорт данных в Docker: контейнеры создают свою БД, встроенные данные нужно
  импортировать один раз:
  ```bash
  docker compose -f infrastructure/docker-compose.yml exec app \
    python scripts/import/import_builtin.py
  ```
- Обратите внимание: в `.env` переменные `OLLAMA_HOST=localhost` и т.п. —
  compose перекрывает их сервисными именами (`ollama`, `qdrant`, `redis`),
  локальный `.env` править не нужно.
- Остановить всё: `docker compose -f infrastructure/docker-compose.yml down`
  (данные БД, Qdrant и весов модели остаются в томах).

---

## 4. Куда класть файлы kit

Структура kit повторяет структуру репозитория — просто скопируйте содержимое
папки `debureaucratizer-kit/` в корень клона с заменой:

```
debureaucratizer-kit/
├── SETUP.md                                    # этот файл
├── fix_boot.patch                              # патч исправления запуска
├── scripts/import/import_builtin.py            # импорт данных + --admin
├── streamlit_app/
│   ├── app.py                                  # главная: вход/регистрация
│   ├── utils/api.py                            # HTTP-клиент UI -> API
│   └── pages/
│       ├── 1_Перевод.py                        # упрощение одного текста
│       ├── 2_Мульти_и_батч.py                  # multi/batch endpoints
│       ├── 3_Ensemble.py                       # ансамбль моделей + критик
│       ├── 4_Глоссарий.py                      # admin: справочник
│       ├── 5_Память_переводов.py               # admin: TM + импорт CSV
│       ├── 6_Оценка_качества.py                # BLEU / chrF++ / BERTScore
│       ├── 7_HITL.py                           # editor/admin: спорные переводы
│       └── 8_Админ.py                          # admin: кэш и статус
├── infrastructure/
│   ├── docker-compose.yml                      # полный стек (заменяет старый)
│   └── Dockerfile                              # + COPY streamlit_app/
└── tests_smoke.md                              # чек-лист быстрой проверки
```

---

## 5. Быстрая проверка (smoke)

1. `curl http://localhost:8000/healthz` -> `{"status":"ok"}`
2. UI открылся, регистрация прошла, справа вверху ваша роль.
3. «Перевод»: вставьте канцелярит, нажмите «Упростить» — появится результат,
   оценка критика и журнал агентов (первый запрос медленный: модель
   загружается в память Ollama).
4. «Глоссарий»: видно 650 записей (после импорта).
5. «Память переводов»: 5876 записей.
6. «Оценка качества»: две строки гипотез/эталонов -> BLEU и chrF++.
