"""Giv: turn a local LLM into a fast, CPU-friendly typed decision model.

A decision model reads typed questions (choice / score / noul) about a piece of
application state and returns probabilities, instead of generating free text.
"""

from giv.decisions import Question, QuestionError
from giv.results import Answer, Decision

__version__ = "0.0.1"

__all__ = ["Question", "QuestionError", "Answer", "Decision", "__version__"]
