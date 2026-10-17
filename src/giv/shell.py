"""Interactive shell: an opencode-style REPL for typed decisions.

Type a state with its choices in curly braces::

    giv › I was charged $50 please refund me {billing, refund, other}

Slash-commands::

    /model <id>    change the weights/model used from here on
    /noul <text>   attach a yes/no question to every state (empty clears it)
    /json          toggle JSON output
    /help          show this help
    /clear         clear the screen
    /quit          leave (Ctrl-D also works)
"""

from __future__ import annotations

import itertools
import os
import re
import sys
import threading
import time
from typing import List, Optional, Tuple

from giv.decisions import Question
from giv.engines import get_engine

_BRACES = re.compile(r"\{([^}]*)\}")

DEFAULT_CHOICE_TEXT = "Which category does this belong to?"


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    return sys.stdout.isatty()


_COLOR = _supports_color()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def bold(t: str) -> str:
    return _c("1", t)


def dim(t: str) -> str:
    return _c("2", t)


def cyan(t: str) -> str:
    return _c("36", t)


def green(t: str) -> str:
    return _c("32", t)


def yellow(t: str) -> str:
    return _c("33", t)


def red(t: str) -> str:
    return _c("31", t)


BANNER = r"""
   ██████╗ ██╗██╗   ██╗
  ██╔════╝ ██║██║   ██║
  ██║  ███╗██║██║   ██║
  ██║   ██║██║╚██╗ ██╔╝
  ╚██████╔╝██║ ╚████╔╝
   ╚═════╝ ╚═╝  ╚═══╝
"""

TAGLINE = "typed decisions from a local LLM - no generation, just a readout"


def parse_line(line: str) -> Tuple[str, Optional[List[str]]]:
    """Split a shell line into (state, options).

    Options are the comma-separated items inside the first ``{...}``. When no
    braces are present, options is None and the whole line is the state.
    """
    m = _BRACES.search(line)
    if not m:
        return line.strip(), None
    state = line[: m.start()].strip()
    if not state:
        state = line[m.end():].strip()
    options = [o.strip() for o in m.group(1).split(",") if o.strip()]
    return state, options or None


def _bar(p: float, width: int = 24) -> str:
    n = max(0, min(width, int(round(p * width))))
    return "█" * n + "░" * (width - n)


class _Spinner:
    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, label: str = "deciding"):
        self.label = label
        self.enabled = _COLOR and sys.stderr.isatty()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "_Spinner":
        if self.enabled:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def _spin(self) -> None:
        for ch in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                break
            sys.stderr.write(f"\r\033[2m{ch} {self.label}\033[0m")
            sys.stderr.flush()
            time.sleep(0.08)

    def __exit__(self, *exc) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()


class Shell:
    def __init__(
        self,
        backend: str = "anyjev",
        model: str = "Qwen/Qwen3-0.6B",
        level: str = "L0",
        dtype: str = "float32",
        quantize: bool = False,
        max_permutations: Optional[int] = None,
    ) -> None:
        self.backend = backend
        self.model = model
        self.level = level
        self.dtype = dtype
        self.quantize = quantize
        self.max_permutations = max_permutations
        self.noul: Optional[str] = None
        self.json = False
        self._engine = None
        self._last_options: Optional[List[str]] = None

    # ---- engine lifecycle -------------------------------------------------
    def _get_engine(self):
        if self._engine is None:
            self._engine = get_engine(
                self.backend,
                model_name=self.model,
                level=self.level,
                dtype=self.dtype,
                quantize=self.quantize,
                max_permutations=self.max_permutations,
            )
        return self._engine

    def _set_model(self, name: str) -> None:
        self.model = name
        self._engine = None

    def _short_model(self) -> str:
        return self.model.rstrip("/").split("/")[-1]

    def _prompt(self) -> str:
        return f"{cyan('giv')} {dim('[' + self._short_model() + ']')} {cyan('›')} "

    # ---- commands ---------------------------------------------------------
    def _command(self, line: str) -> bool:
        """Handle a slash-command. Returns True when the shell should exit."""
        parts = line[1:].split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("quit", "exit", "q"):
            return True
        if cmd in ("help", "?"):
            print(_HELP)
        elif cmd in ("model", "weights", "w"):
            if not arg:
                print(f"  current model: {bold(self.model)}")
            else:
                self._set_model(arg)
                print(f"  {green('✓')} weights -> {bold(arg)} {dim('(loads on next decision)')}")
        elif cmd == "noul":
            self.noul = arg or None
            if self.noul:
                print(f"  {green('✓')} noul question: {self.noul!r}")
            else:
                print(f"  {green('✓')} noul question cleared")
        elif cmd == "json":
            self.json = not self.json
            print(f"  {green('✓')} json output {'on' if self.json else 'off'}")
        elif cmd == "clear":
            sys.stdout.write("\033[2J\033[H" if _COLOR else "\n" * 3)
        else:
            print(f"  {red('unknown command')} /{cmd} {dim('- try /help')}")
        return False

    # ---- deciding ---------------------------------------------------------
    def decide(self, line: str) -> None:
        state, options = parse_line(line)
        if options is not None:
            self._last_options = options
        elif self._last_options is not None:
            options = self._last_options
        else:
            print(
                f"  {yellow('no choices')} - put them in braces, "
                f"e.g. {dim('I was charged $50 {billing, refund, other}')}"
            )
            return
        if not state:
            print(f"  {yellow('empty state')} - add some text before the choices")
            return

        questions = [Question.choice(DEFAULT_CHOICE_TEXT, options, name="choice")]
        if self.noul:
            questions.append(Question.noul(self.noul, name="noul"))

        try:
            with _Spinner(f"deciding with {self._short_model()}"):
                decision = self._get_engine().decide(state, questions)
        except KeyboardInterrupt:
            print(f"\n  {dim('cancelled')}")
            return
        except Exception as exc:  # noqa: BLE001 - surface any engine failure
            print(f"  {red('error')}: {exc}")
            return

        self._render(decision)

    def _render(self, decision) -> None:
        if self.json:
            import json

            print(json.dumps(decision.as_dict(), indent=2))
            return
        print(
            f"  {dim(decision.level)}  {dim(decision.model)}  "
            f"{dim(f'{decision.latency_ms:.0f} ms')}"
        )
        for a in decision.answers:
            label = a.name
            print(f"  {bold(label)}")
            top = max(a.distribution.values()) if a.distribution else 0.0
            for opt, p in sorted(a.distribution.items(), key=lambda kv: -kv[1]):
                bar = _bar(p)
                mark = green(bar) if p == top else dim(bar)
                print(f"    {str(opt):<22} {p:>6.3f}  {mark}")
            if a.expected is not None:
                print(f"    {dim('expected')} {a.expected}")
        print()

    # ---- main loop --------------------------------------------------------
    def run(self) -> int:
        try:
            import readline  # noqa: F401 - enables history/editing in input()
        except Exception:
            pass

        print(cyan(BANNER))
        print(f"  {bold('giv')} {dim('- ' + TAGLINE)}")
        print(f"  {dim('model:')} {self.model}   {dim('type /help for commands, /quit to exit')}")
        print()
        print(f"  {dim('try:')} I was charged $50 please refund me {cyan('{billing, refund, other}')}")
        print()

        while True:
            try:
                line = input(self._prompt()).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line.startswith("/"):
                if self._command(line):
                    break
                continue
            self.decide(line)
        print(dim("bye."))
        return 0


_HELP = f"""
  {bold('input')}
    <state> {{{dim('apple, banana, cherry')}}}   classify the state into one of the choices
    <state>                        reuse the choices from the previous line

  {bold('commands')}
    /model <id>   change the weights (e.g. /model Qwen/Qwen3-1.7B)
    /noul <text>  add a yes/no question to every state ('' clears it)
    /json         toggle JSON output
    /clear        clear the screen
    /help         this help
    /quit         exit
"""


def main(
    backend: str = "anyjev",
    model: str = "Qwen/Qwen3-0.6B",
    level: str = "L0",
    dtype: str = "float32",
    quantize: bool = False,
    max_permutations: Optional[int] = None,
) -> int:
    return Shell(
        backend=backend,
        model=model,
        level=level,
        dtype=dtype,
        quantize=quantize,
        max_permutations=max_permutations,
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
