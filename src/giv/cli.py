"""Command-line interface: ``giv decide ...`` and ``giv benchmark``."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from giv.decisions import Question
from giv.engines import get_engine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="giv",
        description="Turn a local LLM into a typed decision model (CPU-friendly).",
    )
    parser.add_argument("--version", action="version", version="giv 0.0.1")
    sub = parser.add_subparsers(dest="command")

    shell = sub.add_parser("shell", help="interactive REPL (the default)")
    _add_engine_args(shell)

    decide = sub.add_parser("decide", help="classify a state into typed decisions")
    _add_engine_args(decide)
    decide.add_argument("--state", required=True, help="the text state to decide on")
    decide.add_argument(
        "-c",
        "--choice",
        action="append",
        default=[],
        metavar="NAME=a,b,c",
        help="a choice question (repeatable); NAME optional",
    )
    decide.add_argument(
        "--noul",
        action="append",
        default=[],
        metavar="NAME=question",
        help="a yes/no question (repeatable); NAME optional",
    )
    decide.add_argument(
        "--score",
        action="append",
        default=[],
        metavar="NAME=question",
        help="an ordered score question (repeatable); NAME optional",
    )
    decide.add_argument("--json", action="store_true", help="emit JSON")

    bench = sub.add_parser("benchmark", help="time a decision and report memory")
    _add_engine_args(bench)
    bench.add_argument("--state", default="I was charged $50 please refund me")
    bench.add_argument("--choice", default="billing,account_creation,account_deletion,follow_up_purchase")
    bench.add_argument("--runs", type=int, default=3)

    return parser


def _add_engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", default="anyjev")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--level", default="L0")
    p.add_argument(
        "--dtype",
        default="float32",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="float32 is fastest on CPUs without bfloat16 hardware support",
    )
    p.add_argument("--quantize", action="store_true", help="apply dynamic int8 to linear layers")
    p.add_argument(
        "--max-permutations",
        type=int,
        default=None,
        help="cap L0 option-order permutations; omit for full (most robust), 1 = fastest",
    )


def _parse_choice(spec: str) -> Question:
    if "=" in spec:
        name, rest = spec.split("=", 1)
    else:
        name, rest = "", spec
    options = [o.strip() for o in rest.split(",") if o.strip()]
    text = "Which category does this belong to?"
    return Question.choice(text, options, name=name or None)


def _parse_noul(spec: str) -> Question:
    name, text = _split_spec(spec, "Does this hold?")
    return Question.noul(text, name=name)


def _parse_score(spec: str) -> Question:
    name, text = _split_spec(spec, "Rate this on an ordered scale.")
    return Question.score(text, name=name)


def _split_spec(spec: str, default_text: str):
    if "=" in spec:
        name, text = spec.split("=", 1)
        name = name.strip()
        text = text.strip()
    else:
        name, text = "", spec.strip()
    return (name or None), (text or default_text)


def _make_engine(args):
    return get_engine(
        args.backend,
        model_name=args.model,
        level=args.level,
        dtype=args.dtype,
        quantize=getattr(args, "quantize", False),
        max_permutations=getattr(args, "max_permutations", None),
    )


def _cmd_decide(args) -> int:
    questions: List[Question] = []
    for spec in args.choice:
        questions.append(_parse_choice(spec))
    for spec in args.noul:
        questions.append(_parse_noul(spec))
    for spec in args.score:
        questions.append(_parse_score(spec))
    if not questions:
        print("error: provide at least one --choice, --noul, or --score", file=sys.stderr)
        return 2

    engine = _make_engine(args)
    decision = engine.decide(args.state, questions)

    if args.json:
        print(json.dumps(decision.as_dict(), indent=2))
        return 0

    print(f"state: {args.state}")
    print(f"level: {decision.level}   model: {decision.model}   {decision.latency_ms} ms")
    for a in decision.answers:
        bar = _bar(a.probability)
        extra = f"  expected={a.expected}" if a.expected is not None else ""
        print(f"  {a.name:<24} {str(a.value):<20} {a.probability:>6.3f} {bar}{extra}")
    return 0


def _bar(p: float, width: int = 20) -> str:
    n = max(0, min(width, int(round(p * width))))
    return "#" * n


def _cmd_benchmark(args) -> int:
    import resource

    engine = _make_engine(args)
    choice = Question.choice("Which category does this belong to?", args.choice.split(","))
    questions = [choice]

    engine.decide(args.state, questions)
    for _ in range(max(1, args.runs)):
        engine.decide(args.state, questions)

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    d = engine.decide(args.state, questions)
    print(f"model:     {d.model}")
    print(f"level:     {d.level}")
    print(f"options:   {len(choice.options)}  (L0 uses one prefill per option)")
    print(f"latency:   {d.latency_ms} ms/decision")
    print(f"peak RSS:  {peak_mb:.0f} MB")
    for a in d.answers:
        print(f"  {a.name}: {a.distribution}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "shell" or args.command is None:
        from giv.shell import main as shell_main

        return shell_main(
            backend=getattr(args, "backend", "anyjev"),
            model=getattr(args, "model", "Qwen/Qwen3-0.6B"),
            level=getattr(args, "level", "L0"),
            dtype=getattr(args, "dtype", "float32"),
            quantize=getattr(args, "quantize", False),
            max_permutations=getattr(args, "max_permutations", None),
        )
    if args.command == "decide":
        return _cmd_decide(args)
    if args.command == "benchmark":
        return _cmd_benchmark(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
