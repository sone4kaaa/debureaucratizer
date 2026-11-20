"""Supervisor — оркестратор цикла Validator -> Critic -> (Editor -> Validator -> Critic)*."""

from typing import Any

from backend.agents.base import BaseAgent
from backend.core.constants import (
    ADAPTIVE_BUDGET_LOW,
    ADAPTIVE_BUDGET_MID,
    CRITIC_PASS_THRESHOLD,
)
from backend.utils.logging import logger

# Поля, очищаемые перед каждым прогоном
_EPHEMERAL_PREFIXES = ("critic_", "validator_", "supervisor_")
# Поля, не входящие в snapshot (остаются для отладки при rollback)
_NO_SNAPSHOT_KEYS = {"reasoning_log", "editor_rolled_back", "iteration"}

_EDITOR_DEFAULTS: dict[str, Any] = {
    "editor_corrected": None,
    "editor_changed": False,
    "editor_failed": False,
    "editor_iterations": 0,
    "editor_rolled_back": False,
}

STOPPED_REASONS = {
    "converged",
    "max_iterations",
    "no_change",
    "rollback_regression",
    "rollback_error",
    "critic_failed",
    "editor_disabled",
    "agents_disabled",
}


class SupervisorAgent(BaseAgent):
    def __init__(
        self,
        validator: BaseAgent,
        critic: BaseAgent | None,
        editor: BaseAgent | None,
        use_critic: bool = True,
        use_editor: bool = True,
    ) -> None:
        super().__init__("supervisor")
        self.validator = validator
        self.critic = critic
        self.editor = editor
        self.use_critic = use_critic and critic is not None
        self.use_editor = use_editor and editor is not None

    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        self._reset(state)
        log: list[dict[str, Any]] = []
        state["reasoning_log"] = log

        # --- Validator ---
        state = await self.validator.process(state)
        self._log(log, self.validator, "validate")

        if not self.use_critic and not self.use_editor:
            return self._finish(state, stopped_reason="agents_disabled", budget=0)

        # --- Critic ---
        if self.use_critic:
            state = await self.critic.process(state)  # type: ignore[union-attr]
            self._log(log, self.critic, "critique")  # type: ignore[arg-type]
            if state.get("critic_failed"):
                return self._finish(state, stopped_reason="critic_failed", budget=0)

        if not self.use_editor:
            return self._finish(
                state,
                stopped_reason="editor_disabled",
                budget=0,
            )

        # --- Цикл улучшений ---
        budget = self._adaptive_budget(state)
        initial_score = state.get("critic_score")
        state["initial_score"] = initial_score
        stopped_reason = "max_iterations"

        for iteration in range(1, budget + 1):
            snapshot = self._snapshot(state)
            state["iteration"] = iteration

            state = await self.editor.process(state)  # type: ignore[union-attr]
            self._log(log, self.editor, "edit")  # type: ignore[arg-type]

            if state.get("editor_failed"):
                stopped_reason = "rollback_error"
                state = self._restore(snapshot, state)
                state["editor_rolled_back"] = True
                break

            state = await self.validator.process(state)
            self._log(log, self.validator, "revalidate")

            if self.use_critic:
                state = await self.critic.process(state)  # type: ignore[union-attr]
                self._log(log, self.critic, "recritique")  # type: ignore[arg-type]
                if state.get("critic_failed"):
                    stopped_reason = "critic_failed"
                    break

            if not self._is_improvement(snapshot, state):
                state = self._restore(snapshot, state)
                state["editor_rolled_back"] = True
                stopped_reason = "rollback_regression"
                break

            if not state.get("editor_changed"):
                stopped_reason = "no_change"
                break

            if self._converged(state):
                stopped_reason = "converged"
                break
        else:
            if budget == 0:
                stopped_reason = "converged"

        if budget == 0:
            stopped_reason = "converged"

        return self._finish(state, stopped_reason=stopped_reason, budget=budget)

    # ---------- механизмы ----------

    def _reset(self, state: dict[str, Any]) -> None:
        for key in list(state):
            if key.startswith(_EPHEMERAL_PREFIXES):
                del state[key]
        state.update(_EDITOR_DEFAULTS)

    def _adaptive_budget(self, state: dict[str, Any]) -> int:
        score = state.get("critic_score")
        if score is None:
            # без оценки критика (editor-only) — умеренный бюджет
            return 3 if not self.use_critic else 3
        if score < ADAPTIVE_BUDGET_LOW:
            return 3
        if score < ADAPTIVE_BUDGET_MID:
            return 2
        if not state.get("validation_passed") or state.get("critic_suggestions"):
            return 1
        return 0

    def _snapshot(self, state: dict[str, Any]) -> dict[str, Any]:
        snap: dict[str, Any] = {}
        for key, value in state.items():
            if key in _NO_SNAPSHOT_KEYS:
                continue
            snap[key] = list(value) if isinstance(value, list) else value
        return snap

    def _restore(
        self, snapshot: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        # удалить ключи, добавленные в провалившейся итерации
        for key in list(state):
            if key not in snapshot and key not in _NO_SNAPSHOT_KEYS:
                del state[key]
        state.update(snapshot)
        return state

    def _is_improvement(
        self, snapshot: dict[str, Any], state: dict[str, Any]
    ) -> bool:
        if state.get("critic_failed"):
            return False
        old = snapshot.get("critic_score")
        new = state.get("critic_score")
        if old is not None and new is not None and new < old:
            return False
        if snapshot.get("validation_passed") and not state.get("validation_passed"):
            return False
        return True

    def _converged(self, state: dict[str, Any]) -> bool:
        if not state.get("validation_passed"):
            return False
        score = state.get("critic_score")
        if self.use_critic:
            return score is not None and score >= CRITIC_PASS_THRESHOLD and not state.get(
                "critic_suggestions"
            )
        return not state.get("validator_errors")

    def _compute_final(self, state: dict[str, Any]) -> bool:
        if state.get("critic_failed"):
            return False
        if not state.get("validation_passed"):
            return False
        score = state.get("critic_score")
        return score is None or score >= CRITIC_PASS_THRESHOLD

    def _log(
        self, log: list[dict[str, Any]], agent: BaseAgent, step: str
    ) -> None:
        entry: dict[str, Any] = {"agent": agent.name, "step": step}
        if agent.get_last_result():
            entry.update(agent.get_last_result())  # type: ignore[arg-type]
        log.append(entry)

    def _finish(
        self, state: dict[str, Any], stopped_reason: str, budget: int
    ) -> dict[str, Any]:
        success = self._compute_final(state)
        state["final_success"] = success
        state["stopped_reason"] = stopped_reason
        final_entry = {
            "agent": "supervisor",
            "step": "final",
            "iterations": state.get("iteration", 0),
            "stopped_reason": stopped_reason,
            "budget": budget,
            "initial_score": state.get("initial_score", state.get("critic_score")),
            "final_score": state.get("critic_score"),
            "final_success": success,
        }
        state.setdefault("reasoning_log", []).append(final_entry)
        self._last_result = final_entry
        logger.info(
            "Supervisor: %s (итераций=%s, success=%s)",
            stopped_reason,
            final_entry["iterations"],
            success,
        )
        return state
