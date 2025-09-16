import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pipeline.agents import memory_compressor  # noqa: E402


def test_run_returns_no_data(monkeypatch):
    monkeypatch.setattr(memory_compressor, "_call_tool", lambda name, payload: {"items": []})
    out = memory_compressor.run(memory_compressor.MemoryCompressInput(n=5, scope="test"))
    assert out == {"status": "no-data", "inserted": 0, "lessons": []}


def test_run_structures_lessons_and_filters_duplicates(monkeypatch):
    def fake_tool(name, payload):
        assert name == "get_recent_run_summaries"
        assert payload == {"n": 3}
        return {"items": [{"id": 1, "summary": "Alpha drawdown"}]}

    calls: list[tuple[dict, str, dict]] = []

    def fake_insert(lesson, scope: str, meta: dict):
        calls.append((lesson, scope, meta))

    def fake_llm(sys_prompt: str, user_prompt: str):
        payload = json.loads(user_prompt.split("Summaries: ", 1)[1])
        assert payload[0]["summary"] == "Alpha drawdown"
        assert "title" in sys_prompt
        return {
            "lessons": [
                {"title": "Risk limits", "insight": "hit max loss", "action": "cut size", "risk": "low liquidity"},
                {"title": "Risk limits", "insight": "hit max loss", "action": "cut size", "risk": "low liquidity"},
                "Не усреднять убыточные позиции",
            ]
        }

    monkeypatch.setattr(memory_compressor, "_call_tool", fake_tool)
    monkeypatch.setattr(memory_compressor, "insert_agent_lesson", fake_insert)
    monkeypatch.setattr(memory_compressor, "call_openai_json", fake_llm)

    out = memory_compressor.run(memory_compressor.MemoryCompressInput(n=3, scope="desk"))

    assert out["status"] == "ok"
    assert out["mode"] == "llm"
    # Первый урок — словарь из LLM, второй — нормализованная строка
    assert out["lessons"][0]["title"] == "Risk limits"
    assert out["lessons"][1]["insight"].startswith("Не усреднять")

    # Вставлено только два уникальных урока, дубль отфильтрован
    assert len(calls) == 2
    stored_payload = calls[0][0]
    assert stored_payload["title"] == "Risk limits"
    assert calls[0][1] == "desk"
    assert calls[0][2]["n"] == 3
    assert calls[0][2]["mode"] == "llm"
    assert calls[0][2]["llm_error"] is None
    assert "hash" in calls[0][2]
    assert "quality_snapshot" not in calls[0][2]


def test_run_uses_fallback_when_llm_fails(monkeypatch):
    rows = [
        {
            "run_id": "run-42",
            "created_at": "2024-01-01T12:00:00",
            "final": {
                "slot": "london",
                "regime": "trend_up",
                "e4": {"proba_up": 0.63},
                "e12": {"proba_up": 0.55},
                "risk_flags": ["volatility_spike", "news_risk"],
                "ta": {
                    "technical_sentiment": "bullish",
                    "key_observations": ["Бычий импульс от поддержки"],
                },
            },
            "outcome": {
                "4h": {"direction_correct": False, "error_pct": 0.02},
            },
        }
    ]

    def fake_tool(name, payload):
        assert name == "get_recent_run_summaries"
        return {"items": rows}

    stored: list[tuple[dict, dict]] = []

    def fake_insert(lesson, scope: str, meta: dict):
        stored.append((lesson, meta))

    metrics_calls: list[tuple[str, str, int, int, dict]] = []

    def fake_metrics(scope: str, mode: str, rows_processed: int, lessons_inserted: int, metrics: dict) -> int:
        metrics_calls.append((scope, mode, rows_processed, lessons_inserted, metrics))
        return 42

    def boom(*_args, **_kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr(memory_compressor, "_call_tool", fake_tool)
    monkeypatch.setattr(memory_compressor, "insert_agent_lesson", fake_insert)
    monkeypatch.setattr(memory_compressor, "call_openai_json", boom)
    monkeypatch.setattr(memory_compressor, "insert_agent_lesson_metrics", fake_metrics)

    out = memory_compressor.run(memory_compressor.MemoryCompressInput(n=2, scope="ops"))

    assert out["status"] == "ok"
    assert out["mode"] == "fallback"
    assert out["metrics"]["lessons_final"] == 1
    assert out["metrics"]["risk_flag_ratio"] == 1.0
    assert out["inserted"] == 1
    assert len(stored) == 1

    payload, meta = stored[0]
    assert "Режим" in payload["insight"]
    assert "Контролировать" in payload["action"]
    assert payload["risk"] == "volatility_spike, news_risk"
    assert meta["mode"] == "fallback"
    assert meta["llm_error"]
    assert "hash" in meta
    assert meta["quality_metrics_id"] == 42
    assert meta["quality_snapshot"]["rows_processed"] == 1
    assert metrics_calls == [("ops", "fallback", 1, 1, out["metrics"])]
