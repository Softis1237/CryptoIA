from __future__ import annotations

"""
Knowledge loader & search (RAG MVP).

Functions:
  - load_and_embed(paths, source): load files (txt/md), chunk, embed and upsert to DB (pgvector);
  - search(query, k): embed query and fetch top‑k similar docs via pgvector.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from ..agents.embeddings import embed_text
from ..infra.db import upsert_knowledge_doc, query_knowledge_similar


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _split_chunks(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_size)
        out.append(text[i:j])
        if j >= n:
            break
        i = max(i + chunk_size - overlap, j)
    return out


@dataclass
class LoadInput:
    paths: Iterable[str]
    source: str = "docs"


def load_and_embed(inp: LoadInput) -> int:
    count = 0
    for p in inp.paths:
        path = Path(p)
        if not path.exists() or not path.is_file():
            continue
        # only txt/md for MVP (PDF — вне скоупа без внешних зависимостей)
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        text = _read_txt(path)
        chunks = _split_chunks(text)
        for i, ch in enumerate(chunks):
            emb = embed_text(ch)
            upsert_knowledge_doc(
                doc_id=f"{path.name}:{i}",
                source=inp.source,
                title=path.stem,
                chunk_index=i,
                content=ch,
                embedding=emb,
                meta={"path": str(path)},
            )
            count += 1
    return count


def search(query: str, k: int = 5) -> List[dict]:
    emb = embed_text(query)
    return query_knowledge_similar(emb, k=k)

