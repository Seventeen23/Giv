# Giv

Turn a local LLM into a fast, CPU-friendly **typed decision model**.

Giv takes a piece of application state (a support message, a log line, a JSON
record) and returns a probability for every option you give it — instead of
generating prose, it *reads* the model's next-token distribution. This is the
same idea as Jev's "System One" model, built on
[AnyJev](https://github.com/nokia-applied-research/AnyJev).

```
   ██████╗ ██╗██╗   ██╗
  ██╔════╝ ██║██║   ██║
  ██║  ███╗██║██║   ██║
  ██║   ██║██║╚██╗ ██╔╝
  ╚██████╔╝██║ ╚████╔╝
   ╚═════╝ ╚═╝  ╚═══╝
```

## What it does

```bash
giv
giv [Qwen3-0.6B] › I was charged $50 please refund me {billing, refund, other}

  L0  Qwen/Qwen3-0.6B  6215 ms
  choice
    billing    0.997  ████████████████████████
    refund     0.002  ░░░░░░░░░░░░░░░░░░░░░░░░
    other      0.001  ░░░░░░░░░░░░░░░░░░░░░░░░
```

## How it works

Giv is a thin typed layer over **AnyJev**, which turns an ordinary causal LLM
into a decision function. Nothing is trained or fine-tuned, and no text is
generated — the model's *next-token probabilities* are the answer.

The pipeline, top to bottom:

```
giv (CLI / shell / typed contract)          ← this repo
  └─ AnyJevEngine                            src/giv/engines/anyjev_engine.py
       └─ AnyJev Decider + HFBackend         (pip install "anyjev[hf]")
            └─ transformers AutoModelForCausalLM.from_pretrained(...)
                 └─ Qwen3-0.6B weights        ← the only source of knowledge
```

1. **Prompt building.** `anyjev.readout` renders a chat prompt:
   ```
   system: You are a decision function. ... Reply with the answer label only ...
   user:   State: <your state>
           Question: Which category does this belong to?
           Options: A. billing   B. refund   C. other
           Answer with the letter only.
   ```
   Each option is mapped to **exactly one token** (`A`,`B`,`C` for choice,
   `Yes`/`No` for noul, `1`,`2`,`3` for score). Multi-token labels are rejected,
   because the logit readout needs one number per option.

2. **One forward pass.** `HFBackend.next_token_logprobs` feeds the prompt, takes
   the logits at the **last position**, applies `log_softmax`, and reads the
   log-probability of each option token. No generation loop.

3. **Decision correction (`L0`, the default).** Two *training-free* fixes make a
   plain LLM behave like a classifier:
   - **Permutation marginalization** — the options are presented in rotated
     orders and the results averaged, cancelling the model's A/B/C position bias.
   - **Label-free prior correction** — the model's intrinsic preference for some
     labels (e.g. a habit of answering `account_deletion`) is estimated from
     content-free probes and divided out. This is what fixed the
     `account_deletion` bias we saw when *asking* the model instead of reading it.

4. **Result.** A normalized `distribution` over the options, an argmax `value`,
   and a `confidence` — all typed, never free text.

**Why not just ask the model?** Generation makes the LLM commit and keep writing,
which exposes its biases and costs one forward pass *per emitted token*. Reading
its preference among exactly the label tokens is more accurate and no slower
(measured: `read` 4/4 obvious intents vs `ask` 2/4, at comparable latency).

**The connection to your model:** the LLM provides exactly one thing — "how
likely is each next token?" Giv/AnyJev supply all the decision machinery
(prompting, option rotation, prior correction) and never touch the weights. Swap
the model (0.6B ↔ 1.7B) and nothing else changes.

> Thinking mode is always suppressed (`enable_thinking=False` in the chat
> template): a `thinking` trace would move the answer away from the single token
> AnyJev reads, breaking the readout.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[anyjev,torch,dev]"
```

CPU-only torch (smaller/faster to install):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[anyjev,dev]"
```

## Use

### Interactive shell (default)

Run `giv` with no arguments (or `giv shell`) for an opencode-style REPL:

```bash
giv            # or: giv shell
giv --model Qwen/Qwen3-1.7B
```

Type a state and put the choices in **curly braces**:

```
giv [Qwen3-0.6B] › Please delete my account and all my data {billing, account_deletion}
```

If you omit the braces, the previous line's choices are reused.

Shell commands:

| command | what it does |
|---|---|
| `/model <id>` (or `/weights`) | **change the weights** used from here on (loads on next decision) |
| `/noul <text>` | attach a yes/no question to every state (`/noul` clears it) |
| `/json` | toggle JSON output |
| `/clear` | clear the screen |
| `/help` | show help |
| `/quit` | exit (Ctrl-D also works) |

### One-shot decision

```bash
giv decide --state "I was charged $50 please refund me" \
  --choice billing,account_creation,account_deletion,follow_up_purchase \
  --noul needs_human="Does this need a human?" \
  --model Qwen/Qwen3-0.6B --json
```

`decide` accepts `--choice NAME=a,b,c`, `--noul NAME=question`, and
`--score NAME=question` (repeatable), plus `--json`.

### Benchmark

```bash
giv benchmark --choice billing,account_creation,account_deletion,follow_up_purchase
```

## Question kinds

| kind | how you ask | output |
|---|---|---|
| `choice` | `{a, b, c}` or `--choice a,b,c` | distribution over the options |
| `noul` | `/noul <text>` or `--noul` | `Yes`/`No` probability |
| `score` | `--score` | distribution over bins + `expected` value |

## Performance notes

Measured on an 8-core **AVX2-only** CPU, 13 GB RAM (no AVX512 / BF16 / VNNI):

| model | prefills | latency | example verdict |
|---|---|---|---|
| Qwen3-0.6B | full (4) | ~6.7s | billing 0.997 |
| Qwen3-0.6B | `--max-permutations 1` | ~2.8s | billing 0.906 |
| Qwen3-1.7B | full | ~24.8s | billing 1.0 |
| Qwen3-1.7B | `--max-permutations 1` | ~9.3s | billing |

- **Use `float32`** on CPUs without bfloat16 hardware support: bfloat16 is
  emulated there and ~8x slower. Dynamic int8 gives no gain without AVX512-VNNI.
- L0 runs **one prefill per option** for order debiasing; cap it with
  `--max-permutations 1` for a single-prefill path.
- `Qwen/Qwen3-0.6B` fp32 is ~2.4 GB; `Qwen/Qwen3-1.7B` is ~6.8 GB.

## Design

| file | role |
|---|---|
| `src/giv/decisions.py` | the Jev-shaped contract: `Question.choice/.noul/.score` |
| `src/giv/results.py` | `Answer` / `Decision` result types (`distribution`, `confidence`, `expected`) |
| `src/giv/engines/` | pluggable engines; `AnyJevEngine` wraps AnyJev + Qwen |
| `src/giv/shell.py` | the interactive REPL |
| `src/giv/cli.py` | `shell` / `decide` / `benchmark` commands |
| `SPEC.md` | language-agnostic contract for a future Node/TS port |
| `scripts/ask_vs_read.py` | throwaway: compare logit readout vs plain generation |

## Roadmap

- M2: batch/file input, named label sets
- M3: LLM-generated label bootstrap + human review
- M4: calibration beyond L0; TinyJev head-to-head
- M5: tiny encoder fast-path; Node/TS port

## Credits

Decision readout by [AnyJev](https://github.com/nokia-applied-research/AnyJev)
(Apache-2.0). Model: `Qwen/Qwen3-0.6B`.
