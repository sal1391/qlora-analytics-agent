"""Regression test: evaluation must never show the gold answer to a predictor.

Test samples store the gold assistant turn in ``messages`` (same layout as the
training files). If the evaluation driver forwards that assistant message to a
predictor, the model can copy the answer out of its own prompt and every
metric becomes meaningless.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import run_condition  # noqa: E402


class RecordingPredictor:
    """Captures the messages it is asked to predict on."""

    def __init__(self) -> None:
        self.seen: list[list[dict]] = []

    def predict(self, messages):
        self.seen.append(messages)
        return {"skill": "GENERAL_QA", "safety_status": "safe", "sql": None}, "{}", 0.0


def _sample() -> dict:
    return {
        "id": "t-1",
        "task_type": "skill_routing",
        "skill": "SQL_ANALYST",
        "safety_status": "safe",
        "gold_sql": None,
        "messages": [
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": "Schema: ...\nQuestion: total revenue?"},
            {"role": "assistant", "content": '{"skill": "SQL_ANALYST"}'},
        ],
    }


def test_predictor_never_sees_gold_assistant_message():
    predictor = RecordingPredictor()

    class _NoDB:
        def execute(self, *_args, **_kwargs):  # pragma: no cover - not reached
            raise AssertionError("no SQL should execute in this test")

    run_condition(predictor, [_sample()], _NoDB())

    assert predictor.seen, "predictor was never called"
    for messages in predictor.seen:
        roles = [m["role"] for m in messages]
        assert "assistant" not in roles, (
            "gold assistant answer was passed to the predictor prompt: "
            f"roles={roles}"
        )
