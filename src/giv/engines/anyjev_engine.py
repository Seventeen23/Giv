"""AnyJev engine: read a local causal LLM as a Jev-style decision model.

AnyJev reads typed decisions from the model's next-token logits in a single
prefill -- no text is generated. This engine wraps AnyJev and translates Giv
questions to and from its API. Heavy imports (torch, anyjev) are lazy so the
rest of the package works without them.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Sequence

from giv.decisions import Question
from giv.results import Answer, Decision

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


class AnyJevEngine:
    name = "anyjev"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        level: str = "L0",
        dtype: str = "float32",
        quantize: bool = False,
        device: str = "cpu",
        system: Optional[str] = None,
        max_permutations: Optional[int] = None,
    ) -> None:
        self.model_name = model_name
        self.level = level
        self.dtype = dtype
        self.quantize = quantize
        self.device = device
        self.system = system
        self.max_permutations = max_permutations
        self._decider = None

    def _load(self):
        if self._decider is not None:
            return self._decider
        try:
            import anyjev
            from anyjev.backends.hf import HFBackend
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                'AnyJev is not installed. Run: pip install "giv[anyjev]"'
            ) from exc

        backend = HFBackend(self.model_name, device=self.device, dtype=self.dtype)
        if self.quantize:
            _apply_dynamic_int8(backend)

        decider_kw: Dict[str, object] = {"level": self.level}
        if self.system is not None:
            decider_kw["system"] = self.system
        if self.max_permutations is not None:
            decider_kw["max_permutations"] = self.max_permutations

        self._decider = anyjev.Decider(backend, **decider_kw)
        return self._decider

    def decide(self, state, questions: Sequence[Question]) -> Decision:
        decider = self._load()

        pairs = [(q, _to_anyjev(q)) for q in questions]
        started = time.perf_counter()
        result = decider.decide(state, [aj for _, aj in pairs])
        latency_ms = (time.perf_counter() - started) * 1000.0

        answers = [_from_anyjev(q, result[aj.id]) for q, aj in pairs]
        return Decision(
            answers=answers,
            level=str(getattr(result, "level", self.level)),
            model=self.model_name,
            latency_ms=round(latency_ms, 1),
        )


def _apply_dynamic_int8(backend) -> None:
    import torch
    from torch.ao.quantization import quantize_dynamic

    model = getattr(backend, "model", None)
    if model is not None:
        backend.model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def _to_anyjev(q: Question):
    import anyjev

    if q.kind == "choice":
        return anyjev.Question.choice(q.text, list(q.options), name=q.name)
    if q.kind == "noul":
        return anyjev.Question.noul(q.text, name=q.name)
    if q.kind == "score":
        kwargs: Dict[str, object] = {"name": q.name, "scale": q.scale}
        if q.centers is not None:
            kwargs["levels"] = list(q.options)
        else:
            kwargs["bins"] = len(q.options)
        return anyjev.Question.score(q.text, **kwargs)
    raise ValueError(f"unknown question kind: {q.kind!r}")


def _from_anyjev(q: Question, node) -> Answer:
    distribution = {str(k): float(v) for k, v in node.distribution.items()}

    if q.kind == "noul":
        p_true = float(node.p_true)
        value = "Yes" if p_true >= 0.5 else "No"
        prob = p_true if value == "Yes" else 1.0 - p_true
        return Answer(
            q.name, "noul", value, round(prob, 4),
            distribution={"Yes": round(p_true, 4), "No": round(1.0 - p_true, 4)},
        )

    if q.kind == "score":
        expected = float(node.value)
        return Answer(
            q.name, "score", expected, round(float(node.confidence), 4),
            distribution=distribution, expected=round(expected, 4),
        )

    return Answer(
        q.name, "choice", str(node.argmax), round(float(node.confidence), 4),
        distribution=distribution,
    )
