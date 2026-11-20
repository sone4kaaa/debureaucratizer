"""Доменные константы проекта.

Тема — перенос стиля с канцелярита на простой русский, поэтому вместо
«языковых пар» оперируем «стилевыми парами»: source — канцелярский стиль,
target — простой русский.
"""

# Поддерживаемые стили (аналог LANGUAGES в шаблоне)
STYLES: dict[str, str] = {
    "bureau": "Канцелярит (официально-деловой)",
    "legal": "Юридический язык",
    "official": "Официальные уведомления госорганов",
    "plain": "Простой русский",
}

# Стили, надёжно поддерживаемые локальной моделью Ollama (аналог KNOWN_LANGUAGES)
KNOWN_STYLES: set[str] = {"bureau", "legal", "official", "plain"}

# Допустимые стилевые пары: упрощаем только в plain
SOURCE_STYLES: set[str] = {"bureau", "legal", "official"}
TARGET_STYLE: str = "plain"

# Пороги мультиагентной системы
CRITIC_PASS_THRESHOLD: float = 7.0
ADAPTIVE_BUDGET_LOW: float = 4.0
ADAPTIVE_BUDGET_MID: float = 7.0

# Валидатор
GLOSSARY_SIMILARITY_THRESHOLD: float = 0.7
SCRIPT_SHARE_THRESHOLD: float = 0.5

# Домены текстов (для few-shot критика и кэша)
DOMAINS: set[str] = {"general", "legal", "gov", "finance", "hr", "medical"}
CACHEABLE_DOMAIN: str = "general"

# Сид воспроизводимости всех LLM-вызовов
DEFAULT_SEED: int = 42
