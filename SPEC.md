# Giv decision contract (portable spec)

This spec is language-agnostic. The Python implementation here is the reference;
a Node/TS port must produce the same shapes.

## Input

A **state** (string or JSON-serializable object) and one or more typed questions.

## Question primitives

| kind | fields | returns |
|------|--------|---------|
| `choice` | `text`, `options[2..26]` (unique) | distribution over options |
| `noul` | `text` | `p_true` for "Yes" |
| `score` | `text`, `bins`/`levels`, ordered | distribution over bins + expected value |

Names are derived from the question text (slugified) unless given explicitly.

## Output

```json
{
  "level": "L0",
  "model": "Qwen/Qwen3-1.7B",
  "latency_ms": 812.4,
  "answers": {
    "which_category": {
      "kind": "choice",
      "value": "billing",
      "probability": 0.81,
      "distribution": {"billing": 0.81, "account_creation": 0.06, "...": 0.13}
    },
    "needs_human": {
      "kind": "noul",
      "value": "Yes",
      "probability": 0.92,
      "distribution": {"Yes": 0.92, "No": 0.08}
    }
  }
}
```

## Invariants

1. `value` is always one of the declared options (never free text).
2. `probability` is the value's probability; `distribution` sums to ~1.
3. `level` reports the correction level used (`L0` training-free).
4. Probabilities are **not** automatically calibrated — gate on measured data.

## Reference hyperparameters

- Default model: `Qwen/Qwen3-0.6B`, `float32` (fastest on CPUs lacking bfloat16 support).
- Hypothesis/category prompt template is engine-defined; keep it identical across ports.
- `L0`: permutation marginalization + label-free prior correction (AnyJev).
  Full permutations = one prefill per option; `max_permutations=1` = single prefill.
