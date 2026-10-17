"""Compare AnyJev's logit readout ("read") against plain generation ("ask").

Interactive by default -- type a state and see both answers:

    .venv/bin/python scripts/ask_vs_read.py

Or pass states/labels directly:

    .venv/bin/python scripts/ask_vs_read.py \
        --labels billing,account_creation --state "please refund me"

In the REPL: type text to classify, `labels a,b,c` to change labels, `q` to quit.
"""

import argparse
import os
import time

import torch

torch.set_num_threads(os.cpu_count() or 8)

from anyjev.backends.hf import HFBackend
from anyjev.decider import Decider
from anyjev import Question as AJQuestion
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_LABELS = ["billing", "account_creation", "account_deletion", "follow_up_purchase"]
SYSTEM = (
    "You are a support-triage classifier. "
    "Reply with exactly one of the labels and nothing else."
)


def parse_label(text: str, labels) -> str:
    low = text.strip().lower()
    hits = [(low.find(l), l) for l in labels if low.find(l) >= 0]
    return min(hits)[1] if hits else f"<unparsed:{text.strip()[:24]!r}>"


class Lab:
    def __init__(self, model_id: str):
        print("loading model ...", flush=True)
        self.backend = HFBackend(model_id, device="cpu", dtype="float32")
        self.full = Decider(self.backend, level="L0")
        self.one = Decider(self.backend, level="L0", max_permutations=1)
        self.model = self.backend.model.eval()
        self.tok = AutoTokenizer.from_pretrained(model_id)

    def read(self, state, labels, dec):
        q = AJQuestion.choice("Which category does this belong to?", labels, name="category")
        t = time.perf_counter()
        d = dec.decide(state, [q])
        ms = (time.perf_counter() - t) * 1000
        r = d["category"]
        return r.argmax, float(r.confidence), ms

    def ask(self, state, labels):
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"State: {state}\nLabels: {', '.join(labels)}\nCategory:"},
        ]
        prompt = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        enc = self.tok(prompt, return_tensors="pt")
        t = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=8, do_sample=False, pad_token_id=self.tok.eos_token_id
            )
        ms = (time.perf_counter() - t) * 1000
        raw = self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return parse_label(raw, labels), raw, ms

    def compare(self, state, labels):
        rl, rp, rms = self.read(state, labels, self.full)
        ol, _, oms = self.read(state, labels, self.one)
        al, raw, ams = self.ask(state, labels)
        print(f"  read(full) : {rl:<20} p={rp:.2f}   {rms:6.0f} ms")
        print(f"  read(1)    : {ol:<20}          {oms:6.0f} ms")
        print(f"  ask        : {al:<20} raw={raw!r}  {ams:6.0f} ms")
        print(f"  -> read vs ask: {'AGREE' if rl == al else 'DIFFER'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    ap.add_argument("--state", action="append", default=[])
    args = ap.parse_args()

    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    lab = Lab(args.model)

    if args.state:
        for s in args.state:
            print(f"\nstate: {s}\nlabels: {labels}")
            lab.compare(s, labels)
        return

    print(f"\nlabels: {labels}")
    print("type a state to classify, `labels a,b,c` to change labels, `q` to quit.\n")
    while True:
        try:
            line = input("state> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line in ("q", "quit", "exit"):
            break
        if line.startswith("labels "):
            labels = [l.strip() for l in line[7:].split(",") if l.strip()]
            print(f"labels: {labels}")
            continue
        print()
        lab.compare(line, labels)
        print()


if __name__ == "__main__":
    main()
