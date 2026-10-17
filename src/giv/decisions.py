"""The Giv decision contract: typed questions over application state.

Three primitives mirror the System One / Jev shape:

- ``choice``: pick exactly one option from a fixed set (classification, routing)
- ``score``: place the state on an ordered rubric (urgency, risk)
- ``noul``: decide whether a statement holds (yes/no gates)

These are pure, dependency-free specs. An engine translates them into whatever
the underlying model expects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

MAX_OPTIONS = 26


class QuestionError(ValueError):
    """Raised when a question spec is malformed."""


@dataclass(frozen=True)
class Question:
    kind: str
    text: str
    options: tuple[str, ...]
    name: str
    scale: tuple[float, float] = (0.0, 1.0)
    ordered: bool = False
    centers: Optional[tuple[float, ...]] = None

    @staticmethod
    def choice(
        text: str,
        options: Sequence[str],
        name: Optional[str] = None,
    ) -> "Question":
        opts = tuple(str(o) for o in options)
        if len(opts) < 2:
            raise QuestionError("choice needs at least 2 options")
        if len(opts) > MAX_OPTIONS:
            raise QuestionError(f"choice supports at most {MAX_OPTIONS} options")
        if len(set(opts)) != len(opts):
            raise QuestionError("choice options must be unique")
        return Question("choice", text, opts, name or _slug(text))

    @staticmethod
    def noul(text: str, name: Optional[str] = None) -> "Question":
        return Question("noul", text, ("Yes", "No"), name or _slug(text))

    @staticmethod
    def score(
        text: str,
        bins: int = 5,
        scale: Sequence[float] = (0.0, 1.0),
        name: Optional[str] = None,
        levels: Optional[Sequence[str]] = None,
    ) -> "Question":
        if levels is not None:
            labels = tuple(str(x) for x in levels)
            bins = len(labels)
        if bins < 2:
            raise QuestionError("score needs at least 2 bins")
        if levels is not None:
            lo, hi = float(scale[0]), float(scale[1])
            step = (hi - lo) / (bins - 1)
            centers = tuple(lo + i * step for i in range(bins))
        else:
            labels = tuple(str(i + 1) for i in range(bins))
            centers = None
        return Question(
            "score",
            text,
            labels,
            name or _slug(text),
            (float(scale[0]), float(scale[1])),
            True,
            centers,
        )

    @property
    def is_ordered(self) -> bool:
        return self.ordered


def _slug(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in text.strip()]
    slug = "".join(keep).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "question"


def normalize_name(name: str) -> str:
    return _slug(name)
