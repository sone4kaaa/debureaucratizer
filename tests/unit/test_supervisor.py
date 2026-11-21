from typing import Any

import pytest

from backend.agents.base import BaseAgent
from backend.agents.supervisor import SupervisorAgent


class FakeAgent(BaseAgent):
    """Агент-заглушка: применяет заранее заданные результаты по вызовам."""

    def __init__(self, name: str, results: list[dict[str, Any]]) -> None:
        super().__init__(name)
        self.results = results
        self.calls = 0

    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        state.update(result)
        self._last_result = result
        return state


def _validator(passed: bool = True, errors: list[str] | None = None) -> FakeAgent:
    if passed:
        res = {"validator_errors": [], "validator_warnings": [], "validation_passed": True}
    else:
        res = {
            "validator_errors": errors or ["echo_translation"],
            "validator_warnings": [],
            "validation_passed": False,
        }
    return FakeAgent("validator", [res])


def _critic(scores: list[float | None], suggestions: list[str] | None = None) -> FakeAgent:
    results = []
    for s in scores:
        results.append(
            {
                "critic_score": s,
                "critic_suggestions": suggestions or [],
                "critic_failed": s is None,
                "critic_passed": s is not None and s >= 7.0,
            }
        )
    return FakeAgent("critic", results)


def _editor(texts: list[str]) -> FakeAgent:
    results = []
    for t in texts:
        results.append(
            {
                "translated_text": t,
                "editor_corrected": t,
                "editor_changed": True,
                "editor_failed": False,
                "editor_iterations": 1,
            }
        )
    return FakeAgent("editor", results)


BASE_STATE = {
    "source_text": "В соответствии с вышеизложенным уведомляем.",
    "translated_text": "Уведомляем в соответствии с вышеизложенным.",
    "source_lang": "bureau",
    "target_lang": "plain",
}


async def test_early_exit_converged() -> None:
    v = _validator()
    c = _critic([8.0])
    e = _editor(["ничего"])
    sup = SupervisorAgent(v, c, e)
    state = await sup.process(dict(BASE_STATE))
    assert state["stopped_reason"] == "converged"
    assert state["final_success"] is True
    assert e.calls == 0  # Editor не вызывался


async def test_low_score_two_iterations_budget_then_converge() -> None:
    v = _validator()
    c = _critic([5.0, 9.0], suggestions=["упростить"])
    c.results[1]["critic_suggestions"] = []  # на 9.0 замечаний уже нет -> converge
    e = _editor(["Сообщаем: надо оплатить."])
    sup = SupervisorAgent(v, c, e)
    state = await sup.process(dict(BASE_STATE))
    assert state["stopped_reason"] == "converged"
    assert state["final_success"] is True
    assert state["translated_text"] == "Сообщаем: надо оплатить."
    assert e.calls == 1


async def test_high_score_but_validation_failed_one_iteration() -> None:
    v = FakeAgent(
        "validator",
        [
            {"validator_errors": ["x"], "validator_warnings": [], "validation_passed": False},
            {"validator_errors": [], "validator_warnings": [], "validation_passed": True},
        ],
    )
    c = _critic([8.0, 8.5])
    e = _editor(["Исправленный текст."])
    sup = SupervisorAgent(v, c, e)
    state = await sup.process(dict(BASE_STATE))
    assert e.calls == 1
    assert state["stopped_reason"] == "converged"
    assert state["final_success"] is True


async def test_rollback_on_regression() -> None:
    original = BASE_STATE["translated_text"]
    v = _validator()
    c = _critic([6.0, 4.0])  # вторая итерация ухудшает score
    e = _editor(["Ухудшенный вариант."])
    sup = SupervisorAgent(v, c, e)
    state = await sup.process(dict(BASE_STATE))
    assert state["stopped_reason"] == "rollback_regression"
    assert state["editor_rolled_back"] is True
    assert state["translated_text"] == original  # состояние восстановлено
    assert state["critic_score"] == 6.0


async def test_editor_only_mode() -> None:
    v = _validator()
    e = _editor(["Простой текст."])
    sup = SupervisorAgent(v, critic=None, editor=e, use_critic=False, use_editor=True)
    state = await sup.process(dict(BASE_STATE))
    assert e.calls >= 1
    assert "critic_score" not in state
    assert state["stopped_reason"] in ("converged", "no_change", "max_iterations")


async def test_agents_disabled() -> None:
    v = _validator()
    sup = SupervisorAgent(v, critic=None, editor=None, use_critic=False, use_editor=False)
    state = await sup.process(dict(BASE_STATE))
    assert state["stopped_reason"] == "agents_disabled"
    assert state["validation_passed"] is True
    assert state["final_success"] is True


async def test_critic_failed_stops_pipeline() -> None:
    v = _validator()
    c = _critic([None])
    e = _editor(["не вызовется"])
    sup = SupervisorAgent(v, c, e)
    state = await sup.process(dict(BASE_STATE))
    assert state["stopped_reason"] == "critic_failed"
    assert state["final_success"] is False
    assert e.calls == 0


async def test_reasoning_log_complete() -> None:
    v = _validator()
    c = _critic([8.0])
    e = _editor(["x"])
    sup = SupervisorAgent(v, c, e)
    state = await sup.process(dict(BASE_STATE))
    log = state["reasoning_log"]
    agents = [entry["agent"] for entry in log]
    assert agents[0] == "validator"
    assert "critic" in agents
    assert agents[-1] == "supervisor"
    final = log[-1]
    assert final["stopped_reason"] == "converged"
    assert final["initial_score"] == 8.0
    assert final["final_score"] == 8.0


async def test_max_iterations_when_not_converging() -> None:
    v = _validator()
    # score 5 -> бюджет 2, критик не улучшается, но и не регрессирует
    c = _critic([5.0, 5.5, 5.5], suggestions=["ещё"])
    c.results[0]["critic_suggestions"] = ["ещё"]
    e = FakeAgent(
        "editor",
        [
            {"translated_text": "v1", "editor_changed": True, "editor_failed": False, "editor_iterations": 1},
            {"translated_text": "v2", "editor_changed": True, "editor_failed": False, "editor_iterations": 2},
        ],
    )
    sup = SupervisorAgent(v, c, e)
    state = await sup.process(dict(BASE_STATE))
    assert state["stopped_reason"] == "max_iterations"
    assert state["final_success"] is False  # score < 7


async def test_state_reusable_after_reset() -> None:
    v = _validator()
    c = _critic([8.0])
    sup = SupervisorAgent(v, c, None, use_editor=False)
    state = dict(BASE_STATE)
    await sup.process(state)
    state["critic_score"] = 1.0  # «мусор» снаружи
    await sup.process(state)
    assert state["critic_score"] == 8.0  # перезаписан свежим прогоном
    assert state["stopped_reason"] == "editor_disabled"
