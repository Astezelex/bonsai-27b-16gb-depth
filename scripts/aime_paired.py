#!/usr/bin/env python3
"""Item-level paired analysis of AIME26 across the two models and the two runs.

Same method as mmlu_paired.py: McNemar on discordant pairs, pairing keyed on a SHA1 of the
question text and asserted identical across all four runs before anything is compared.

AIME26 carries only 30 items, so this is the benchmark where a per-item view matters most.
An aggregate 0.90 against 0.60 is 27 against 18, and the whole question is whether the nine
items separating them are the same nine every time.
"""
import hashlib, json, os, sys
from math import comb

R = os.environ.get("BONSAI_ROOT", "/mnt/bigdisk/bonsai")
RUNS = {
    ("bonsai", "aug"): f"{R}/anchors-2026-08/results/bonsai-aime26/20260828_173816/reviews/bonsai/aime26_default.jsonl",
    ("qwen",   "aug"): f"{R}/anchors-2026-08/results/qwen-iq2xxs-aime26/20260828_173816/reviews/qwen-iq2xxs/aime26_default.jsonl",
    ("bonsai", "jul"): f"{R}/results/evalscope/bonsai-aime26/20260715_165009/reviews/bonsai/aime26_default.jsonl",
    ("qwen",   "jul"): f"{R}/results/evalscope/qwen-iq2xxs-aime26/20260715_165009/reviews/qwen-iq2xxs/aime26_default.jsonl",
}


def qhash(rec):
    for m in rec["messages"]:
        if m.get("role") == "user":
            return hashlib.sha1(m["content"].encode()).hexdigest()[:12]
    raise KeyError("no user message")


def load(path):
    if not os.path.exists(path):
        sys.exit(f"MISSING {path}")
    acc, tgt = {}, {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        v = rec["sample_score"]["score"]["value"]["acc"]
        if v not in (0.0, 1.0):
            sys.exit(f"non-binary acc {v} in {path}")
        h = qhash(rec)
        acc[h] = int(v)
        tgt[h] = rec.get("target")
    return acc, tgt


runs, targets = {}, {}
for k, v in RUNS.items():
    runs[k], targets[k] = load(v)
    print(f"loaded {k[0]:7} {k[1]}  {len(runs[k])} items  correct {sum(runs[k].values())}"
          f"  = {sum(runs[k].values())/len(runs[k]):.4f}")

keys = set(runs[("bonsai", "aug")])
for k, v in runs.items():
    if set(v) != keys:
        sys.exit(f"ITEM SET MISMATCH {k}: {len(set(v)-keys)} extra, {len(keys-set(v))} missing")
    for h in keys:
        if targets[k][h] != targets[("bonsai", "aug")][h]:
            sys.exit(f"TARGET MISMATCH on {h} in {k}")
print(f"\nGATE: all four runs cover the identical {len(keys)} question hashes, "
      f"and every target agrees. Pairing is verified.")
keys = sorted(keys)


def mcnemar(a, b, label):
    n01 = sum(1 for k in keys if a[k] == 0 and b[k] == 1)
    n10 = sum(1 for k in keys if a[k] == 1 and b[k] == 0)
    m = n01 + n10
    p = min(1.0, 2 * sum(comb(m, i) for i in range(min(n01, n10) + 1)) / 2 ** m) if m else 1.0
    print(f"  {label:32} a-only {n10:2d}  b-only {n01:2d}  discordant {m:2d}"
          f"  net {n10-n01:+3d}  exact p = {p:.4f}")
    return m, n10 - n01, p


print("\nMcNEMAR, bonsai vs qwen on identical items (a = bonsai, b = qwen):")
for era in ("aug", "jul"):
    mcnemar(runs[("bonsai", era)], runs[("qwen", era)], f"AIME26, {era}")

print("\nTEST-RETEST, same model across the two runs (a = Aug, b = Jul):")
for mdl in ("bonsai", "qwen"):
    mcnemar(runs[(mdl, "aug")], runs[(mdl, "jul")], f"{mdl}, Aug vs Jul")

print("\nPER-ITEM FLIP RATE, same model, two runs at temperature 0.7:")
for mdl in ("bonsai", "qwen"):
    f = sum(1 for k in keys if runs[(mdl, "aug")][k] != runs[(mdl, "jul")][k])
    print(f"  {mdl:7} {f:2d} of {len(keys)} changed verdict = {f/len(keys)*100:.1f}%")

# Is the separation the SAME items in both runs? That is the reproduction test.
print("\nREPRODUCTION: items bonsai got right and qwen got wrong, per run")
ba = {k for k in keys if runs[("bonsai","aug")][k] == 1 and runs[("qwen","aug")][k] == 0}
bj = {k for k in keys if runs[("bonsai","jul")][k] == 1 and runs[("qwen","jul")][k] == 0}
qa = {k for k in keys if runs[("bonsai","aug")][k] == 0 and runs[("qwen","aug")][k] == 1}
qj = {k for k in keys if runs[("bonsai","jul")][k] == 0 and runs[("qwen","jul")][k] == 1}
print(f"  bonsai-only: Aug {len(ba)}, Jul {len(bj)}, in BOTH {len(ba & bj)}")
print(f"  qwen-only:   Aug {len(qa)}, Jul {len(qj)}, in BOTH {len(qa & qj)}")
net_both = len(ba & bj) - len(qa & qj)
m2 = len(ba & bj) + len(qa & qj)
p2 = min(1.0, 2 * sum(comb(m2, i) for i in range(min(len(ba & bj), len(qa & qj)) + 1)) / 2 ** m2) if m2 else 1.0
print(f"  restricted to items that reproduce in BOTH runs: "
      f"bonsai {len(ba & bj)}, qwen {len(qa & qj)}, net {net_both:+d}, exact p = {p2:.4f}")

print("\nAgreement, August: both right "
      f"{sum(1 for k in keys if runs[('bonsai','aug')][k]==runs[('qwen','aug')][k]==1)}, "
      f"both wrong {sum(1 for k in keys if runs[('bonsai','aug')][k]==runs[('qwen','aug')][k]==0)}")

print("\nPOOLED across both runs (60 model-item observations per model, same 30 questions):")
for mdl in ("bonsai", "qwen"):
    c = sum(runs[(mdl,'aug')].values()) + sum(runs[(mdl,'jul')].values())
    print(f"  {mdl:7} {c}/60 = {c/60:.4f}")
