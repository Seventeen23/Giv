"""Result types returned by an engine for a set of questions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Answer:
    name: str
    kind: str
    value: object
    probability: float
    distribution: Dict[str, float] = field(default_factory=dict)
    expected: Optional[float] = None

    def __str__(self) -> str:
        return f"{self.name}={self.value!r} (p={self.probability:.3f})"


@dataclass
class Decision:
    answers: List[Answer]
    level: str
    model: Optional[str] = None
    latency_ms: Optional[float] = None

    def __getitem__(self, name: str) -> Answer:
        for a in self.answers:
            if a.name == name:
                return a
        raise KeyError(name)

    def get(self, name: str) -> Optional[Answer]:
        try:
            return self[name]
        except KeyError:
            return None

    def as_dict(self) -> dict:
        return {
            "level": self.level,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "answers": {
                a.name: {
                    "kind": a.kind,
                    "value": a.value,
                    "probability": round(a.probability, 4),
                    "distribution": {k: round(v, 4) for k, v in a.distribution.items()},
                    **({"expected": a.expected} if a.expected is not None else {}),
                }
                for a in self.answers
            },
        }
