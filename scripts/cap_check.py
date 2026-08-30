#!/usr/bin/env python3
"""Does the token cap explain qwen's AIME instability?

The determinism finding says qwen flips 11 of 30 AIME items between runs and bonsai flips 1.
An intermittent 60000-token cap would produce exactly that pattern for a completely different
reason: a truncated generation emits no ANSWER line and scores zero regardless of whether the
model knew the answer. Cap-rate is first-class evidence and has to be ruled in or out before
the flip rate is attributed to sampling.

Reports, per model per run: output-token distribution, count at or near the cap, count with
no extracted prediction, and the overlap between capped items and flipped items.
"""
import os
import hashlib, json, sys

R = os.environ.get("BONSAI_ROOT", "/mnt/bigdisk/bonsai")
CAP = 60000
RUNS = {
 ("bonsai","aug"): f"{R}/anchors-2026-08/results/bonsai-aime26/20260828_173816/reviews/bonsai/aime26_default.jsonl",
 ("qwen","aug"):   f"{R}/anchors-2026-08/results/qwen-iq2xxs-aime26/20260828_173816/reviews/qwen-iq2xxs/aime26_default.jsonl",
 ("bonsai","jul"): f"{R}/results/evalscope/bonsai-aime26/20260715_165009/reviews/bonsai/aime26_default.jsonl",
 ("qwen","jul"):   f"{R}/results/evalscope/qwen-iq2xxs-aime26/20260715_165009/reviews/qwen-iq2xxs/aime26_default.jsonl",
}

def qhash(rec):
    for m in rec["messages"]:
        if m.get("role") == "user":
            return hashlib.sha1(m["content"].encode()).hexdigest()[:12]
    raise KeyError

def load(p):
    out = {}
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        sc = rec["sample_score"]["score"]
        tok = None
        for m in rec["messages"]:
            if m.get("role") == "assistant" and m.get("perf_metrics"):
                tok = m["perf_metrics"].get("output_tokens")
        pred = sc.get("extracted_prediction")
        out[qhash(rec)] = {
            "acc": int(sc["value"]["acc"]),
            "tok": tok,
            "pred": pred,
            "nopred": pred in (None, "", "None"),
        }
    return out

d = {k: load(v) for k, v in RUNS.items()}
keys = sorted(d[("bonsai","aug")])
print(f"{len(keys)} items, cap = {CAP}\n")
for k, v in d.items():
    toks = sorted(x["tok"] for x in v.values() if x["tok"] is not None)
    if not toks:
        sys.exit(f"NO output_tokens recorded for {k}. Cap analysis impossible, do not claim it.")
    atcap = sum(1 for t in toks if t >= CAP * 0.98)
    nopred = sum(1 for x in v.values() if x["nopred"])
    n = len(toks)
    print(f"{k[0]:7} {k[1]}  n={n}  median {toks[n//2]:6d}  max {toks[-1]:6d}  "
          f"mean {sum(toks)//n:6d}   >=98% of cap: {atcap}   no prediction extracted: {nopred}")

print("\nPer-item, qwen: did the flipped items differ in token use?")
qa, qj = d[("qwen","aug")], d[("qwen","jul")]
flipped = [k for k in keys if qa[k]["acc"] != qj[k]["acc"]]
stable  = [k for k in keys if qa[k]["acc"] == qj[k]["acc"]]
for label, ks in (("flipped", flipped), ("stable", stable)):
    ta = [qa[k]["tok"] for k in ks if qa[k]["tok"]]
    tj = [qj[k]["tok"] for k in ks if qj[k]["tok"]]
    both = ta + tj
    print(f"  {label:8} n={len(ks):2d}  mean output tokens {sum(both)//len(both) if both else 0:6d}"
          f"  max {max(both) if both else 0:6d}"
          f"  capped {sum(1 for t in both if t >= CAP*0.98)}"
          f"  nopred {sum(1 for k in ks for r in (qa[k], qj[k]) if r['nopred'])}")

print("\nSame split for bonsai:")
ba, bj = d[("bonsai","aug")], d[("bonsai","jul")]
fb = [k for k in keys if ba[k]["acc"] != bj[k]["acc"]]
sb = [k for k in keys if ba[k]["acc"] == bj[k]["acc"]]
for label, ks in (("flipped", fb), ("stable", sb)):
    both = [r["tok"] for k in ks for r in (ba[k], bj[k]) if r["tok"]]
    print(f"  {label:8} n={len(ks):2d}  mean output tokens {sum(both)//len(both) if both else 0:6d}"
          f"  max {max(both) if both else 0:6d}"
          f"  capped {sum(1 for t in both if t >= CAP*0.98)}")

print("\nToken use, correct vs wrong (August):")
for mdl in ("bonsai","qwen"):
    v = d[(mdl,"aug")]
    for want in (1, 0):
        ts = [v[k]["tok"] for k in keys if v[k]["acc"] == want and v[k]["tok"]]
        if ts:
            print(f"  {mdl:7} acc={want}  n={len(ts):2d}  mean {sum(ts)//len(ts):6d}  max {max(ts):6d}")
