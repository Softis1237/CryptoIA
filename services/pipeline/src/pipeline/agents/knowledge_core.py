from __future__ import annotations

"""Knowledge Core (RAG) skeleton.

Collects documents (books/articles/strategies), prepares embeddings and provides retrieval interface for agents.

Note: full implementation requires a table and embedding model config; here we only scaffold API.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import os
import re
import hashlib

from ..infra.db import upsert_knowledge_doc, query_knowledge_similar

# In-memory fallback store for tests / no-DB environments
_MEM_STORE: list[dict] = []


@dataclass
class BuildInput:
    sources: List[str]  # list of file paths or URLs
    model: str = "hash-128"  # hash-128 | sbert | openai
    chunk_size: int = 1200
    chunk_overlap: int = 200


def _read_source(path: str) -> Tuple[str, str]:
    """Return (source_id, text). Supports local .md/.txt and optional URL/PDF via flags."""
    allow_http = os.getenv("KNOWLEDGE_ALLOW_HTTP", "0") in {"1", "true", "True"}
    if path.startswith("http://") or path.startswith("https://"):
        if not allow_http:
            raise RuntimeError("HTTP sources disabled; set KNOWLEDGE_ALLOW_HTTP=1 to enable")
        txt: Optional[str] = None
        try:
            import requests  # type: ignore
            from bs4 import BeautifulSoup  # type: ignore
            resp = requests.get(path, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.extract()
            txt = soup.get_text(" ")
        except Exception:
            txt = None
        if not txt:
            raise RuntimeError(f"failed to fetch URL: {path}")
        return (path, txt)
    # PDF
    if path.lower().endswith(".pdf"):
        try:
            import pypdf  # type: ignore
            reader = pypdf.PdfReader(path)
            pages = [page.extract_text() or "" for page in reader.pages]
            txt = "\n".join(pages)
            return (os.path.abspath(path), txt)
        except Exception as e:
            raise RuntimeError(f"failed to read PDF {path}: {e}")
    # Fallback: plain text
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    return (os.path.abspath(path), txt)


def _clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _chunk(text: str, size: int, overlap: int) -> List[str]:
    out = []
    i = 0
    n = max(100, int(size))
    ov = max(0, int(overlap))
    while i < len(text):
        out.append(text[i : i + n])
        if i + n >= len(text):
            break
        i = i + n - ov
    return out


def _embed_hash(text: str, dim: int = 128) -> List[float]:
    vec = [0.0] * dim
    tokens = re.findall(r"[A-Za-z0-9_]+", (text or "").lower())
    for t in tokens:
        h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    import math
    s = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / s for v in vec]


def _embed_sbert(text: str, model_name: str | None = None) -> List[float]:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        m = SentenceTransformer(model_name or os.getenv("SBERT_MODEL", "all-MiniLM-L6-v2"))
        import numpy as np  # type: ignore
        v = m.encode([text or ""], normalize_embeddings=True)[0].tolist()
        return [float(x) for x in v]
    except Exception as e:
        raise RuntimeError(f"sbert embedding failed: {e}")


def _embed_openai(text: str, model: str | None = None) -> List[float]:
    try:
        import openai  # type: ignore
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        openai.api_key = api_key
        mdl = model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        # minimal client usage
        resp = openai.embeddings.create(model=mdl, input=text or "")
        v = resp.data[0].embedding
        return [float(x) for x in v]
    except Exception as e:
        raise RuntimeError(f"openai embedding failed: {e}")


def _embed(text: str, prefer: str = "hash-128") -> Tuple[List[float], Dict[str, Any]]:
    prefer = (prefer or os.getenv("KCORE_EMBEDDER", "hash-128")).lower()
    if prefer.startswith("sbert"):
        v = _embed_sbert(text, None)
        return v, {"embedder": "sbert", "model": os.getenv("SBERT_MODEL", "all-MiniLM-L6-v2")}
    if prefer.startswith("openai"):
        mdl = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        v = _embed_openai(text, mdl)
        return v, {"embedder": "openai", "model": mdl}
    # default hash
    v = _embed_hash(text, dim=128)
    return v, {"embedder": "hash-128", "model": "hash-128"}


def build(inp: BuildInput) -> Dict[str, Any]:
    inserted = 0
    for src in inp.sources:
        sid, raw = _read_source(src)
        txt = _clean_text(raw)
        chunks = _chunk(txt, inp.chunk_size, inp.chunk_overlap)
        for i, ch in enumerate(chunks):
            emb, meta = _embed(ch, prefer=inp.model)
            doc_id = f"{sid}#chunk{i}"
            try:
                upsert_knowledge_doc(
                    doc_id=doc_id,
                    source=sid,
                    title=os.path.basename(sid),
                    chunk_index=i,
                    content=ch,
                    embedding=emb,
                    meta={"embedder": meta.get("embedder"), "model": meta.get("model"), "len": len(ch)},
                )
            except Exception:
                _MEM_STORE.append({
                    "doc_id": doc_id,
                    "source": sid,
                    "title": os.path.basename(sid),
                    "chunk_index": i,
                    "content": ch,
                    "embedding": emb,
                })
            inserted += 1
    return {"status": "ok", "inserted": inserted}


@dataclass
class QueryInput:
    query: str
    top_k: int = 5


def query(inp: QueryInput) -> Dict[str, Any]:
    emb, meta = _embed(inp.query or "", prefer=os.getenv("KCORE_EMBEDDER", "hash-128"))
    rows = []
    try:
        rows = query_knowledge_similar(emb, k=max(1, int(inp.top_k)))
    except Exception:
        rows = []
    if not rows and _MEM_STORE:
        # cosine similarity in-memory
        import math
        def _cos(a, b):
            s = sum(x*y for x, y in zip(a, b))
            na = math.sqrt(sum(x*x for x in a)) or 1.0
            nb = math.sqrt(sum(y*y for y in b)) or 1.0
            return s/(na*nb)
        scored = [
            (itm, _cos(emb, itm.get("embedding") or []))
            for itm in _MEM_STORE
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        rows = [{
            "doc_id": it.get("doc_id"),
            "source": it.get("source"),
            "title": it.get("title"),
            "chunk_index": it.get("chunk_index"),
            "content": it.get("content"),
            "distance": float(1.0 - score),
        } for it, score in scored[: max(1, int(inp.top_k))]]
    # sanitize output
    out = []
    for r in rows:
        out.append({k: r.get(k) for k in ("doc_id", "source", "title", "chunk_index", "content", "distance")})
    return {"status": "ok", "top": out}
