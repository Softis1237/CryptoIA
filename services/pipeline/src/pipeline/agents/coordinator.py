from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Tuple

from loguru import logger

from .base import BaseAgent, AgentResult


@dataclass
class Edge:
    src: str
    dst: str


class AgentCoordinator:
    """Minimal DAG-based agent coordinator.

    - Registers agents by name
    - Runs them in topological order respecting dependencies
    - Retries failed agents (simple fixed attempts)
    """

    def __init__(self, max_retries: int = 0, retry_sleep_sec: float = 0.0):
        self._agents: Dict[str, BaseAgent] = {}
        self._deps: Dict[str, List[str]] = defaultdict(list)
        self.max_retries = max_retries
        self.retry_sleep_sec = retry_sleep_sec

    def register(self, agent: BaseAgent, depends_on: Iterable[str] | None = None) -> None:
        if agent.name in self._agents:
            raise ValueError(f"Agent already registered: {agent.name}")
        self._agents[agent.name] = agent
        if depends_on:
            self._deps[agent.name].extend(depends_on)

    def _toposort(self) -> List[str]:
        indeg: Dict[str, int] = {n: 0 for n in self._agents}
        for dst, srcs in self._deps.items():
            for _ in srcs:
                indeg[dst] += 1
        q = deque([n for n, d in indeg.items() if d == 0])
        order: List[str] = []
        while q:
            n = q.popleft()
            order.append(n)
            for m, srcs in self._deps.items():
                if n in srcs:
                    indeg[m] -= 1
                    if indeg[m] == 0:
                        q.append(m)
        if len(order) != len(self._agents):
            raise RuntimeError("Cycle detected in agent dependencies")
        # Stable by priority
        order.sort(key=lambda n: getattr(self._agents[n], "priority", 1000))
        return order

    def run(self, payloads: Dict[str, dict]) -> Dict[str, AgentResult]:
        results: Dict[str, AgentResult] = {}
        order = self._toposort()
        for name in order:
            agent = self._agents[name]
            deps = self._deps.get(name, [])
            ready = all(results.get(d) and results[d].ok for d in deps)
            if not ready:
                logger.warning(f"Skip agent {name}: unmet deps {deps}")
                results[name] = AgentResult(name=name, ok=False, output=None, meta={"reason": "deps_failed"})
                continue
            attempt = 0
            payload = payloads.get(name, {})
            while True:
                started = datetime.now(timezone.utc).isoformat()
                try:
                    res = agent.run(payload)
                    res.started_at = res.started_at or started
                    res.finished_at = datetime.now(timezone.utc).isoformat()
                    results[name] = res
                    break
                except Exception as e:  # noqa: BLE001
                    attempt += 1
                    logger.exception(f"Agent {name} failed on attempt {attempt}: {e}")
                    if attempt > self.max_retries:
                        results[name] = AgentResult(name=name, ok=False, output=None, started_at=started,
                                                     finished_at=datetime.now(timezone.utc).isoformat(),
                                                     meta={"error": str(e)})
                        break
                    time.sleep(self.retry_sleep_sec)
        return results

