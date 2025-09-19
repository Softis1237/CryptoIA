from __future__ import annotations

"""
Лёгкая оболочка над OpenAI Embeddings с безопасным fallback без сети.

Env:
  - OPENAI_API_KEY, OPENAI_EMBED_MODEL (default: text-embedding-3-small)
Размерность по умолчанию: 1536.
"""

import hashlib
import os
from typing import List
from ..infra.secrets import get_secret


def _hash_to_vec(text: str, dim: int = 1536) -> List[float]:
    # детерминированный псевдо-вектор (fallback)
    h = hashlib.sha256(text.encode("utf-8")).digest()
    vals = []
    for i in range(dim):
        b = h[i % len(h)]
        vals.append(((b / 255.0) - 0.5) * 2.0)  # [-1, 1]
    return vals


def embed_text(text: str, model: str | None = None) -> List[float]:
    model = model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    api_key = get_secret("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if not api_key:
        return _hash_to_vec(text)
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        out = client.embeddings.create(input=text, model=model)
        vec = out.data[0].embedding
        # нормализация не обязательна
        return [float(x) for x in vec]
    except Exception:
        return _hash_to_vec(text)
