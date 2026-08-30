#!/usr/bin/env python3
"""Check each claimed number in the deliverables against the artefact it came from.
A claim with no artefact behind it is a finding, not a rounding question."""
import json, glob, os, re, sys

R = os.environ.get("BONSAI_ROOT", "/mnt/bigdisk/bonsai")
fails, oks = [], []

def chk(label, got, want, tol=0.0):
    try:
        g, w = float(got), float(want)
        ok = abs(g - w) <= tol
    except (TypeError, ValueError):
        g, w = got, want; ok = (got == want)
    (oks if ok else fails).append(f"{label}: claimed {want}, artefact {got}")

def report(path):
    d = json.load(open(path))
    m = d["metrics"][0]
    n = sum(s.get("num", 0) for c in m.get("categories", []) for s in c.get("subsets", []))
    return round(d["score"] * n), n

# --- anchors, August
A = f"{R}/anchors-2026-08/results"
for label, path, want_c, want_n in [
    ("AIME bonsai Aug",  f"{A}/bonsai-aime26/20260828_173816/reports/bonsai/aime26.json", 27, 30),
    ("AIME qwen Aug",    f"{A}/qwen-iq2xxs-aime26/20260828_173816/reports/qwen-iq2xxs/aime26.json", 18, 30),
    ("MMLU bonsai Aug",  f"{A}/bonsai-mmlu_redux/20260829_055510/reports/bonsai/mmlu_redux.json", 297, 342),
    ("MMLU qwen Aug",    f"{A}/qwen-iq2xxs-mmlu_redux/20260829_055510/reports/qwen-iq2xxs/mmlu_redux.json", 291, 342),
    ("LCB bonsai Aug",   f"{A}/bonsai-live_code_bench/20260829_091833/reports/bonsai/live_code_bench.json", 30, 50),
    ("LCB qwen Aug",     f"{A}/qwen-iq2xxs-live_code_bench/20260829_102802/reports/qwen-iq2xxs/live_code_bench.json", 12, 50),
]:
    if not os.path.exists(path):
        fails.append(f"{label}: ARTEFACT MISSING {path}"); continue
    c, n = report(path)
    chk(f"{label} correct", c, want_c); chk(f"{label} n", n, want_n)

# --- phase 3
p3 = glob.glob(f"{A}/bonsai-dspark-aime26/*/reports/bonsai-dspark/aime26.json")
if p3:
    c, n = report(p3[0]); chk("phase3 drafter correct", c, 26); chk("phase3 n", n, 30)
else:
    fails.append("phase3: report artefact MISSING")

# --- model file sizes (PQ2_0 claim)
for label, path, want in [
    ("Q2_g64 bytes", f"{R}/rerun-2026-08/models/Ternary-Bonsai-27B-Q2_g64.gguf", 7585330240),
    ("PQ2_0 bytes",  f"{R}/rerun-2026-08/models/Ternary-Bonsai-27B-PQ2_0.gguf", 7165121600),
    ("drafter bytes",f"{R}/rerun-2026-08/models/dflash-slim-Q4_0.gguf", 631712480),
    ("bias bytes",   f"{R}/rerun-2026-08/kv-bias-q4_0.gguf", 66400),
]:
    chk(label, os.path.getsize(path) if os.path.exists(path) else "MISSING", want)
pct = (7585330240 - 7165121600) / 7585330240 * 100
chk("PQ2_0 pct smaller", round(pct, 2), 5.54, 0.01)

# --- host RAM arithmetic
chk("perplexity reserve at 262144 GB", round(262144*151936*4/1e9, 1), 159.3, 0.05)
chk("perplexity reserve at 32768 GB",  round(32768*151936*4/1e9, 1), 19.9, 0.05)

# --- skew arithmetic from the bench files
def bench_tg(p):
    for line in open(p):
        if "tg128" in line and "|" in line:
            return float(line.rsplit("|", 2)[1].split("±")[0].strip())
E = sorted(glob.glob(f"{R}/earlybench-*/"))[-1]
fk = [bench_tg(f"{E}bench-fork-iq2xxs-fork-p{i}.txt") for i in (1, 2)]
mn = [bench_tg(f"{E}bench-main-iq2xxs-main-p{i}.txt") for i in (1, 2)]
if all(fk) and all(mn):
    fm, mm = sum(fk)/2, sum(mn)/2
    chk("fork mean tg128", round(fm, 2), 34.42, 0.01)
    chk("mainline mean tg128", round(mm, 2), 35.28, 0.02)
    chk("decode skew pct", round((mm-fm)/mm*100, 2), 2.42, 0.02)
else:
    fails.append("skew: could not parse one of the bench files")

print(f"PASS {len(oks)}   FAIL {len(fails)}")
for f in fails: print("  FAIL " + f)
if not fails: print("  every checked claim matches its artefact")
