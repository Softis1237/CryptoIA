from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable, Optional, Dict


@dataclass
class AgentResult:
    name: str
    ok: bool
    output: Any
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


@runtime_checkable
class BaseAgent(Protocol):
    """Common interface for agents used by the coordinator.

    Agents should be deterministic for the same payload and keep side effects
    explicit (e.g., returning S3 paths, not hiding writes unless documented).
    """

    name: str
    priority: int  # smaller = higher priority

    def run(self, payload: dict) -> AgentResult:  # pragma: no cover - interface only
        ...

