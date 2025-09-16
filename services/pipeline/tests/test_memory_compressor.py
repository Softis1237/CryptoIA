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

    calls: list[tuple[str, str, dict]] = []

    def fake_insert(text: str, scope: str, meta: dict):
        calls.append((text, scope, meta))

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
    # Первый урок — словарь из LLM, второй — нормализованная строка
    assert out["lessons"][0]["title"] == "Risk limits"
    assert out["lessons"][1]["insight"].startswith("Не усреднять")

    # Вставлено только два уникальных урока, дубль отфильтрован
    assert len(calls) == 2
    stored_payload = json.loads(calls[0][0])
    assert stored_payload["title"] == "Risk limits"
    assert calls[0][1] == "desk"
    assert calls[0][2]["n"] == 3
    assert "hash" in calls[0][2]
