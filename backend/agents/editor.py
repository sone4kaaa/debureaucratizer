"""Editor — LLM-редактор, исправляющий упрощение по замечаниям."""

from typing import Any

from backend.agents.base import Agent
from backend.utils.logging import logger


class EditorAgent(Agent):
    temperature = 0.3
    max_new_tokens = 1024

    def __init__(self, engine: str | None = None) -> None:
        super().__init__("editor", engine=engine)

    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        errors = state.get("validator_errors") or []
        warnings = state.get("validator_warnings") or []
        suggestions = state.get("critic_suggestions") or []

        if state.get("critic_failed"):
            return self._skip(state, "critic_failed")
        if not errors and not warnings and not suggestions:
            return self._skip(state, "no_issues")

        original = state.get("translated_text", "")
        prompt = self._build_prompt(
            state.get("source_text", ""), original, errors, warnings, suggestions
        )

        result: dict[str, Any] = {
            "editor_iterations": state.get("editor_iterations", 0) + 1,
        }
        try:
            corrected = (await self._call_llm(prompt)).strip()
        except Exception as exc:
            logger.warning("Editor: LLM недоступна (%s)", exc)
            result.update(
                editor_corrected=None, editor_changed=False, editor_failed=True
            )
            state.update(result)
            self._last_result = result
            return state

        if not corrected:
            # Пустой ответ — сохраняем оригинал
            result.update(
                editor_corrected=None, editor_changed=False, editor_failed=True
            )
        else:
            result.update(
                editor_corrected=corrected,
                editor_changed=corrected != original,
                editor_failed=False,
            )
            if result["editor_changed"]:
                state["translated_text"] = corrected
        state.update(result)
        self._last_result = result
        return state

    def _skip(self, state: dict[str, Any], reason: str) -> dict[str, Any]:
        result = {
            "editor_corrected": None,
            "editor_changed": False,
            "editor_failed": False,
            "editor_iterations": state.get("editor_iterations", 0),
            "editor_skip_reason": reason,
        }
        state.update(result)
        self._last_result = result
        return state

    @staticmethod
    def _build_prompt(
        source: str,
        current: str,
        errors: list[str],
        warnings: list[str],
        suggestions: list[str],
    ) -> str:
        issues = "\n".join(
            [*(f"- ОШИБКА: {e}" for e in errors)]
            + [*(f"- ПРЕДУПРЕЖДЕНИЕ: {w}" for w in warnings)]
            + [*(f"- ПРЕДЛОЖЕНИЕ: {s}" for s in suggestions)]
        )
        return f"""Ты — редактор, упрощающий официальный текст на русском языке.

Исходный текст (канцелярит):
{source}

Текущее упрощение:
{current}

Замечания:
{issues}

Жёсткие правила:
- применяй замечания буквально; не добавляй новые факты, субъекты и объекты;
- даты, суммы, имена, номера статей и реквизиты менять нельзя;
- NO-CHANGE — лучше, чем сломанное изменение: если непонятно, как исправить, оставь как есть;
- верни ТОЛЬКО исправленный текст, без пояснений и кавычек.

Исправленный текст:"""
