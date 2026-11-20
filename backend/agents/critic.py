"""Critic — LLM-as-a-Judge для оценки упрощения текста."""

import json
import re
from typing import Any

from backend.agents.base import Agent
from backend.core.constants import CRITIC_PASS_THRESHOLD
from backend.utils.logging import logger

_UNIVERSAL_FEWSHOT = """
Примеры оценки (source — канцелярит, hypothesis — упрощение):

1. Хорошее упрощение:
source: "В соответствии с вышеизложенным уведомляем о необходимости осуществления оплаты."
hypothesis: "Сообщаем: нужно оплатить."
{"adequacy": 9, "fluency": 9, "terminology": 8, "overall_score": 9, "suggestions": []}

2. Потерян смысл:
source: "Оплату необходимо произвести в срок до 15 марта 2025 года."
hypothesis: "Оплатите когда-нибудь."
{"adequacy": 1, "fluency": 7, "terminology": 2, "overall_score": 2, "suggestions": ["вернуть точную дату — 15 марта 2025 года"]}

3. Канцелярит остался:
source: "Настоящим извещаем о необходимости предоставления документов."
hypothesis: "Настоящим уведомляем: предоставьте документы."
{"adequacy": 7, "fluency": 3, "terminology": 3, "overall_score": 3, "suggestions": ["убрать 'настоящим уведомляем' — это канцелярит"]}

4. Эхо источника (текст не изменён):
source: "В целях исполнения договора просим предоставить отчётность."
hypothesis: "В целях исполнения договора просим предоставить отчётность."
{"adequacy": 2, "fluency": 0, "terminology": 2, "overall_score": 0, "suggestions": ["переписать простым языком"]}
"""

_DOMAIN_FEWSHOT: dict[str, str] = {
    "legal": """
Пример для юридического домена — юридические факты (суммы, сроки, статьи) нельзя менять:
source: "Сторона обязуется уплатить неустойку в размере 0,1% за каждый день просрочки."
hypothesis: "За каждый день просрочки нужно платить неустойку 0,1%."
{"adequacy": 10, "fluency": 9, "terminology": 9, "overall_score": 9, "suggestions": []}
""",
    "gov": """
Пример для госуведомлений — сохранять адресата и требуемое действие:
source: "Уведомляем о необходимости явки в отделение МФЦ по адресу: ул. Ленина, 1."
hypothesis: "Придите в МФЦ по адресу: ул. Ленина, 1."
{"adequacy": 9, "fluency": 10, "terminology": 9, "overall_score": 9, "suggestions": []}
""",
}

_LANGUAGE_RULES = """
Правила для русского языка:
- простой русский ≠ разговорный сленг («ввалить бабок» — недопустимо);
- вежливое «вы» сохраняется, если было в источнике;
- имена собственные, суммы, даты, номера статей менять нельзя.
"""

_COMMON_RULES = """
Правила оценки (0–10):
1. adequacy: насколько сохранён смысл и факты. Изменённая дата/сумма/имя → adequacy = 0.
2. fluency: насколько текст естественен. Канцеляризмы («осуществлять», «в целях», «надлежащим образом») → fluency <= 5. Грамматические ошибки → fluency <= 5.
3. FLUENCY = 0 и adequacy = 0, если: hypothesis дословно повторяет source (утечка источника), содержит CJK-пунктуацию или английские мета-фразы («Here is the simplified text»).
4. terminology: обязательные термины (ФИО, названия организаций, реквизиты) сохранены.
5. overall_score — средневзвешенное, целое или с одним знаком после запятой.
6. suggestions — конкретные, применимые правки (не «сделай лучше»).
Ответь ТОЛЬКО JSON: {"adequacy": N, "fluency": N, "terminology": N, "overall_score": N, "suggestions": [...]}.
"""


def _extract_json_spans(text: str) -> list[str]:
    """Сканер: вырезает JSON-объекты с балансировкой скобок и учётом строк."""
    spans: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append(text[start: i + 1])
                start = -1
    return spans


def _parse_scores(raw: str) -> dict[str, Any] | None:
    for span in _extract_json_spans(raw):
        try:
            data = json.loads(span)
        except json.JSONDecodeError:
            continue
        if "overall_score" in data:
            return data
    # fallback-регулярка, если LLM не выдал валидный JSON
    m = re.search(r'"?overall_score"?\s*:\s*(\d+(?:\.\d+)?)', raw)
    if m:
        return {"overall_score": float(m.group(1))}
    return None


class CriticAgent(Agent):
    temperature = 0.0
    max_new_tokens = 512

    def __init__(self, engine: str | None = None) -> None:
        super().__init__("critic", engine=engine)

    def _build_prompt(self, source: str, hypothesis: str, domain: str) -> str:
        parts = [
            "Ты — строгий эксперт-оценщик упрощения официальных текстов на русском языке.",
            _UNIVERSAL_FEWSHOT,
        ]
        if domain in _DOMAIN_FEWSHOT:
            parts.append(_DOMAIN_FEWSHOT[domain])
        parts.append(_COMMON_RULES)
        parts.append(_LANGUAGE_RULES)
        parts.append(
            f'\nОцени:\nsource: "{source}"\nhypothesis: "{hypothesis}"\nJSON:'
        )
        return "\n".join(parts)

    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        source = state.get("source_text", "")
        hypothesis = state.get("translated_text", "")
        domain = state.get("domain", "general")
        prompt = self._build_prompt(source, hypothesis, domain)

        try:
            raw = await self._call_llm(prompt)
        except Exception as exc:
            logger.warning("Critic: LLM недоступна (%s)", exc)
            result = self._failed_result()
            state.update(result)
            self._last_result = result
            return state

        parsed = _parse_scores(raw)
        if parsed is None:
            logger.warning("Critic: не удалось распарсить ответ LLM")
            result = self._failed_result()
            state.update(result)
            self._last_result = result
            return state

        score = parsed.get("overall_score")
        try:
            score = float(score) if score is not None else None
        except (TypeError, ValueError):
            score = None

        result = {
            "critic_adequacy": parsed.get("adequacy"),
            "critic_fluency": parsed.get("fluency"),
            "critic_terminology": parsed.get("terminology"),
            "critic_score": score,
            "critic_suggestions": parsed.get("suggestions") or [],
            "critic_failed": score is None,
            "critic_passed": score is not None and score >= CRITIC_PASS_THRESHOLD,
        }
        state.update(result)
        self._last_result = result
        return state

    @staticmethod
    def _failed_result() -> dict[str, Any]:
        # Fallback-оценка 5.0 не используется: None означает «критик не сработал»
        return {
            "critic_adequacy": None,
            "critic_fluency": None,
            "critic_terminology": None,
            "critic_score": None,
            "critic_suggestions": [],
            "critic_failed": True,
            "critic_passed": False,
        }
