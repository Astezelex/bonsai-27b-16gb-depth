#!/usr/bin/env python3
"""Run-to-run stability of the two models, on both benchmarks.

The aggregate scores hide something the paired data shows plainly: the two systems are not
equally repeatable. This quantifies it and controls for the obvious confound, which is that a
model scoring near 0.5 has more room to flip than one scoring near 0.9.

Determinism index: if every item carried the same success probability p, two independent runs
would disagree on 2p(1-p) of them. Observed flips well below that expectation mean the items
are near 0 or near 1, so the model either knows an item or does not. Observed flips near that
expectation mean the model is sampling.
"""
import hashlib, json, glob, os, sys
from math import comb, exp, lgamma

R = os.environ.get("BONSAI_ROOT", "/mnt/bigdisk/bonsai")


def qhash(rec):
    for m in rec["messages"]:
        if m.get("role") == "user":
            return hashlib.sha1(m["content"].encode()).hexdigest()[:12]
    raise KeyError("no user message")


def load_files(paths):
    out = {}
    for p in paths:
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[qhash(rec)] = int(rec["sample_score"]["score"]["value"]["acc"])
    return out


SETS = {
 "AIME26": {
   ("bonsai","aug"): [f"{R}/anchors-2026-08/results/bonsai-aime26/20260828_173816/reviews/bonsai/aime26_default.jsonl"],
   ("qwen","aug"):   [f"{R}/anchors-2026-08/results/qwen-iq2xxs-aime26/20260828_173816/reviews/qwen-iq2xxs/aime26_default.jsonl"],
   ("bonsai","jul"): [f"{R}/results/evalscope/bonsai-aime26/20260715_165009/reviews/bonsai/aime26_default.jsonl"],
   ("qwen","jul"):   [f"{R}/results/evalscope/qwen-iq2xxs-aime26/20260715_165009/reviews/qwen-iq2xxs/aime26_default.jsonl"],
 },
 "MMLU-Redux": {
   ("bonsai","aug"): sorted(glob.glob(f"{R}/anchors-2026-08/results/bonsai-mmlu_redux/20260829_055510/reviews/bonsai/mmlu_redux_*.jsonl")),
   ("qwen","aug"):   sorted(glob.glob(f"{R}/anchors-2026-08/results/qwen-iq2xxs-mmlu_redux/20260829_055510/reviews/qwen-iq2xxs/mmlu_redux_*.jsonl")),
   ("bonsai","jul"): sorted(glob.glob(f"{R}/results/evalscope/bonsai-mmlu_redux/20260716_060917/reviews/bonsai/mmlu_redux_*.jsonl")),
   ("qwen","jul"):   sorted(glob.glob(f"{R}/results/evalscope/qwen-iq2xxs-mmlu_redux/20260716_060917/reviews/qwen-iq2xxs/mmlu_redux_*.jsonl")),
 },
}


def fisher_2x2(a, b, c, d):
    """Two-sided Fisher exact by summing tables no more probable than the observed one."""
    def lc(n, k):
        return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
    n = a + b + c + d
    r1, c1 = a + b, a + c
    def prob(x):
        return exp(lc(r1, x) + lc(n - r1, c1 - x) - lc(n, c1))
    p0 = prob(a) * (1 + 1e-9)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p0))


for bench, spec in SETS.items():
    runs = {k: load_files(v) for k, v in spec.items()}
    keys = set(runs[("bonsai", "aug")])
    for k, v in runs.items():
        if set(v) != keys:
            sys.exit(f"{bench}: item set mismatch on {k}")
    keys = sorted(keys)
    N = len(keys)
    print(f"\n===== {bench}, {N} items =====")
    stats = {}
    for mdl in ("bonsai", "qwen"):
        a, j = runs[(mdl, "aug")], runs[(mdl, "jul")]
        flips = sum(1 for k in keys if a[k] != j[k])
        pooled = (sum(a.values()) + sum(j.values())) / (2 * N)
        exp_flip = 2 * pooled * (1 - pooled) * N
        stats[mdl] = (flips, pooled, exp_flip)
        idx = flips / exp_flip if exp_flip else float("nan")
        print(f"  {mdl:7} pooled acc {pooled:.4f}   flips {flips:3d}/{N}"
              f" = {flips/N*100:5.1f}%   expected if homogeneous {exp_flip:5.1f}"
              f"   determinism index {idx:.2f}")
    fb, _, _ = stats["bonsai"]
    fq, _, _ = stats["qwen"]
    p = fisher_2x2(fb, N - fb, fq, N - fq)
    print(f"  Fisher exact on flip counts, {fb} vs {fq} of {N}: p = {p:.4f}")
    print("  (determinism index below 1.0 means the model is decisive per item;"
          " near 1.0 means it is sampling)")
