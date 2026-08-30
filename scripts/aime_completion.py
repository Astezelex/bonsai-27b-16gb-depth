#!/usr/bin/env python3
"""AIME26 separated into two questions the aggregate score conflates.

  1. Did the model finish inside the 60000-token budget and emit an answer?
  2. Given that it finished, was the answer right?

The capped set and the no-prediction set are identical item for item in three of the four
runs, so a truncated generation scores zero whether or not the model was on track. Reporting
0.90 against 0.60 without this split describes budget efficiency and reasoning quality as if
they were one number.
"""
import os
import hashlib, json
from math import comb

R = os.environ.get("BONSAI_ROOT", "/mnt/bigdisk/bonsai")
CAP_FLOOR = 58800          # 98% of the 60000 budget
RUNS = {
 ("bonsai","aug"): f"{R}/anchors-2026-08/results/bonsai-aime26/20260828_173816/reviews/bonsai/aime26_default.jsonl",
 ("qwen","aug"):   f"{R}/anchors-2026-08/results/qwen-iq2xxs-aime26/20260828_173816/reviews/qwen-iq2xxs/aime26_default.jsonl",
 ("bonsai","jul"): f"{R}/results/evalscope/bonsai-aime26/20260715_165009/reviews/bonsai/aime26_default.jsonl",
 ("qwen","jul"):   f"{R}/results/evalscope/qwen-iq2xxs-aime26/20260715_165009/reviews/qwen-iq2xxs/aime26_default.jsonl",
}

def qh(r):
    for m in r["messages"]:
        if m.get("role") == "user":
            return hashlib.sha1(m["content"].encode()).hexdigest()[:12]

def load(p):
    o = {}
    for l in open(p):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l); sc = r["sample_score"]["score"]
        tok = None
        for m in r["messages"]:
            if m.get("role") == "assistant" and m.get("perf_metrics"):
                tok = m["perf_metrics"].get("output_tokens")
        pr = sc.get("extracted_prediction")
        o[qh(r)] = {"acc": int(sc["value"]["acc"]), "tok": tok,
                    "done": pr not in (None, "", "None")}
    return o

d = {k: load(v) for k, v in RUNS.items()}
keys = sorted(d[("bonsai","aug")])

print("SPLIT: completion rate, then accuracy among completions\n")
print(f"{'run':14} {'completed':>10} {'correct':>9} {'raw acc':>9} {'acc | completed':>16} {'median tok':>11}")
for k in (("bonsai","aug"),("bonsai","jul"),("qwen","aug"),("qwen","jul")):
    v = d[k]
    comp = [x for x in keys if v[x]["done"]]
    cor = sum(v[x]["acc"] for x in keys)
    corc = sum(v[x]["acc"] for x in comp)
    toks = sorted(v[x]["tok"] for x in keys)
    print(f"{k[0]+' '+k[1]:14} {len(comp):>6}/{len(keys):<3} {cor:>9} "
          f"{cor/len(keys):>9.3f} {corc/len(comp):>16.3f} {toks[len(toks)//2]:>11}")

def mcn(a, b, field, label, ks):
    n01 = sum(1 for k in ks if not a[k][field] and b[k][field])
    n10 = sum(1 for k in ks if a[k][field] and not b[k][field])
    m = n01 + n10
    p = min(1.0, 2*sum(comb(m,i) for i in range(min(n01,n10)+1))/2**m) if m else 1.0
    print(f"  {label:38} bonsai-only {n10:2d}  qwen-only {n01:2d}  net {n10-n01:+3d}  exact p = {p:.4f}")

print("\nMcNEMAR on COMPLETION (did it emit an answer inside 60k):")
for era in ("aug","jul"):
    mcn(d[("bonsai",era)], d[("qwen",era)], "done", f"completion, {era}", keys)

print("\nMcNEMAR on CORRECTNESS, restricted to items BOTH models completed:")
for era in ("aug","jul"):
    ks = [k for k in keys if d[("bonsai",era)][k]["done"] and d[("qwen",era)][k]["done"]]
    a, b = d[("bonsai",era)], d[("qwen",era)]
    n01 = sum(1 for k in ks if a[k]["acc"]==0 and b[k]["acc"]==1)
    n10 = sum(1 for k in ks if a[k]["acc"]==1 and b[k]["acc"]==0)
    m = n01+n10
    p = min(1.0, 2*sum(comb(m,i) for i in range(min(n01,n10)+1))/2**m) if m else 1.0
    ca = sum(a[k]["acc"] for k in ks); cb = sum(b[k]["acc"] for k in ks)
    print(f"  {era}: n={len(ks):2d} both-completed   bonsai {ca}/{len(ks)}  qwen {cb}/{len(ks)}"
          f"   discordant {m}  net {n10-n01:+d}  exact p = {p:.4f}")

print("\nCOMPLETION REPRODUCIBILITY (same item, both runs):")
for mdl in ("bonsai","qwen"):
    a, j = d[(mdl,"aug")], d[(mdl,"jul")]
    ca = {k for k in keys if a[k]["done"]}; cj = {k for k in keys if j[k]["done"]}
    print(f"  {mdl:7} completed Aug {len(ca)}, Jul {len(cj)}, in BOTH {len(ca&cj)}, "
          f"in NEITHER {len(keys)-len(ca|cj)}, unstable {len(ca^cj)}")

print("\nTOKEN COST TO REACH A CORRECT ANSWER (completed and correct only):")
for k in (("bonsai","aug"),("qwen","aug")):
    v = d[k]
    ts = sorted(v[x]["tok"] for x in keys if v[x]["done"] and v[x]["acc"]==1)
    print(f"  {k[0]:7} n={len(ts):2d}  median {ts[len(ts)//2]:6d}  mean {sum(ts)//len(ts):6d}  max {ts[-1]:6d}")
