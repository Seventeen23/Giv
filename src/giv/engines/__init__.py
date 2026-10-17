"""Decision-model engines.

An engine takes application state and Giv questions and returns a Decision.
Engines are imported lazily so the core package stays dependency-free.
"""

from giv.engines.base import Engine

__all__ = ["Engine", "get_engine"]


def get_engine(backend: str, **kwargs) -> Engine:
    """Return an engine by name.

    Known backends: ``anyjev`` (local LLM read as a decision model).
    """
    backend = backend.lower()
    if backend in ("anyjev", "local"):
        from giv.engines.anyjev_engine import AnyJevEngine

        return AnyJevEngine(**kwargs)
    raise ValueError(f"unknown backend: {backend!r}")
