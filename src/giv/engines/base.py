"""Engine interface shared by all backends."""

from __future__ import annotations

from typing import Protocol, Sequence, Union

from giv.decisions import Question
from giv.results import Decision

State = Union[str, dict]


class Engine(Protocol):
    name: str

    def decide(self, state: State, questions: Sequence[Question]) -> Decision:
        ...
