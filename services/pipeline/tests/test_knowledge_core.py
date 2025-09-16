import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from pipeline.agents.knowledge_core import build, query, BuildInput, QueryInput  # noqa: E402


def test_build_and_query_tmpfile(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("""# Test
This is a simple technical analysis primer about bullish engulfing pattern.
It describes how to use EMA and RSI.
""", encoding="utf-8")
    out = build(BuildInput(sources=[str(p)], model="hash-128", chunk_size=200, chunk_overlap=20))
    assert out["status"] == "ok"
    assert out["inserted"] >= 1
    res = query(QueryInput(query="bullish engulfing", top_k=3))
    assert res["status"] == "ok"
    assert isinstance(res.get("top"), list)

