from __future__ import annotations

"""Заготовки для симуляционных сценариев Red Team."""

from dataclasses import dataclass
from typing import Dict


@dataclass(slots=True)
class RedTeamScenario:
    name: str
    description: str
    parameters: Dict[str, float]

    def to_payload(self) -> Dict[str, float]:
        return dict(self.parameters)

