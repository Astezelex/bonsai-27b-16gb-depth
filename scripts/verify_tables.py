#!/usr/bin/env python3
"""Verify every table value in the deliverables against the run artefacts."""
import csv, glob, json, os, statistics, sys

R = os.environ.get("BONSAI_ROOT", "/mnt/bigdisk/bonsai")
fails, oks = [], []
def chk(label, got, want, tol=0.0):
    try:
        ok = abs(float(got) - float(want)) <= tol
    except (TypeError, ValueError):
        ok = got == want
    (oks if ok else fails).append(f"{label}: doc says {want}, artefact {got}")

def gate(run, budget):
    """accuracy, completed, capped, acc|completed straight from reviews."""
    n = cor = cap = comp = compcor = 0
    floor = int(budget * 0.98)
    for f in glob.glob(os.path.join(run, "reviews", "*", "*.jsonl")):
        for line in open(f):
            line = line.strip()
            if not line: continue
            rec = json.loads(line); sc = rec["sample_score"]["score"]
            acc = int(sc["value"]["acc"]); tok = None
            for m in rec["messages"]:
                if m.get("role") == "assistant" and m.get("perf_metrics"):
                    tok = m["perf_metrics"].get("output_tokens")
            done = sc.get("extracted_prediction") not in (None, "", "None")
            n += 1; cor += acc
            if tok is not None and tok >= floor: cap += 1
            if done: comp += 1; compcor += acc
    return n, cor, cap, (compcor/comp if comp else 0)

A = f"{R}/anchors-2026-08/results"
J = f"{R}/results/evalscope"
CELLS = [
 ("AIME bonsai Jul", f"{J}/bonsai-aime26/20260715_165009", 60000, 3, 0.9630),
 ("AIME qwen Jul",   f"{J}/qwen-iq2xxs-aime26/20260715_165009", 60000, 11, 1.0000),
 ("MMLU bonsai Jul", f"{J}/bonsai-mmlu_redux/20260716_060917", 8000, 6, 0.8817),
 ("MMLU qwen Jul",   f"{J}/qwen-iq2xxs-mmlu_redux/20260716_060917", 8000, 18, 0.9074),
 ("LCB bonsai Jul",  f"{J}/bonsai-live_code_bench/20260716_150714", 14000, 23, 0.9286),
 ("LCB qwen Jul",    f"{J}/qwen-iq2xxs-live_code_bench/20260716_150714", 14000, 35, 0.9375),
 ("AIME bonsai Aug", f"{A}/bonsai-aime26/20260828_173816", 60000, 3, 0.9643),
 ("AIME qwen Aug",   f"{A}/qwen-iq2xxs-aime26/20260828_173816", 60000, 11, 0.9474),
 ("MMLU bonsai Aug", f"{A}/bonsai-mmlu_redux/20260829_055510", 8000, 6, 0.8787),
 ("MMLU qwen Aug",   f"{A}/qwen-iq2xxs-mmlu_redux/20260829_055510", 8000, 20, 0.9009),
 ("LCB bonsai Aug",  f"{A}/bonsai-live_code_bench/20260829_091833", 14000, 19, 0.9375),
 ("LCB qwen Aug",    f"{A}/qwen-iq2xxs-live_code_bench/20260829_102802", 14000, 38, 1.0000),
]
for label, run, budget, want_cap, want_accc in CELLS:
    if not os.path.isdir(run): fails.append(f"{label}: RUN DIR MISSING"); continue
    n, cor, cap, accc = gate(run, budget)
    chk(f"{label} cap", cap, want_cap)
    chk(f"{label} acc|conv", round(accc, 4), want_accc, 0.0002)

p3 = glob.glob(f"{A}/bonsai-dspark-aime26/*/")
if p3:
    n, cor, cap, accc = gate(p3[0], 60000)
    chk("phase3 correct", cor, 26); chk("phase3 cap", cap, 3)
    chk("phase3 acc|conv", round(accc, 4), 0.9630, 0.0002)

# ---- drafter depth table, from the two natural-filler sweeps
def med(tsv, depth, center, draft):
    v = [float(r["decode_tps"]) for r in csv.DictReader(open(tsv), delimiter="\t")
         if int(r["depth_target"]) == depth and r["center"] == center
         and r["draft"] == draft and r["rep"] not in ("LOAD_FAIL",)
         and r["decode_tps"] not in ("ERROR", "")]
    return round(statistics.median(v), 2) if v else None
def vram(tsv, depth, center, draft):
    v = [r["vram_mib"] for r in csv.DictReader(open(tsv), delimiter="\t")
         if int(r["depth_target"]) == depth and r["center"] == center
         and r["draft"] == draft and r["vram_mib"]]
    return int(v[0]) if v else None

T1 = f"{R}/pasha-depth-20260829-162441/results.tsv"
T2 = f"{R}/pasha-depth-20260829-190835/results.tsv"
for tsv, d, nd, wd, vn, vd in [
    (T1, 0,      42.11, 77.88,  7893, 10629),
    (T1, 8192,   32.02, 65.38,  7975, 10769),
    (T2, 16384,  25.50, 35.87,  8159, 11049),
    (T1, 32768,  18.25, 18.32,  8527, 11609),
    (T1, 131072,  6.78, 10.18, 10735, 14969),
]:
    chk(f"depth {d} no-draft t/s", med(tsv, d, "off", "off"), nd, 0.02)
    chk(f"depth {d} drafted t/s",  med(tsv, d, "off", "on"),  wd, 0.02)
    chk(f"depth {d} VRAM no-draft", vram(tsv, d, "off", "off"), vn)
    chk(f"depth {d} VRAM drafted",  vram(tsv, d, "off", "on"),  vd)
chk("depth 262144 no-draft t/s", med(T1, 262144, "off", "off"), 3.69, 0.02)
chk("depth 262144 VRAM no-draft", vram(T1, 262144, "off", "off"), 13679)
chk("centering VRAM at 262144", vram(T1, 262144, "on", "off"), 13681)
chk("centering t/s at depth 0", med(T1, 0, "on", "off"), 42.10, 0.02)

# ---- KLD ladder
K = {}
for f in glob.glob(f"{R}/pasha-kld-*/kld.tsv"):
    for r in csv.DictReader(open(f), delimiter="\t"):
        K[(r["stage"], r["ctx"], r["corpus"], r["arm"])] = r["value"]
for key, want in [
    (("control","512","orig","uncentered"), "0.000646"),
    (("control","512","orig","centered"),   "0.000487"),
    (("isolate","512","orig","uncentered"), "0.001019"),
    (("isolate","512","orig","centered"),   "0.000945"),
    (("isolate","512","wiki","uncentered"), "0.001360"),
    (("isolate","512","wiki","centered"),   "0.001231"),
    (("ladder","8192","wiki","uncentered"), "0.002145"),
    (("ladder","8192","wiki","centered"),   "0.002130"),
    (("ladder","16384","wiki","uncentered"),"0.002312"),
    (("ladder","16384","wiki","centered"),  "0.002329"),
]:
    chk("KLD " + " ".join(key[1:]), K.get(key, "MISSING"), want)

print(f"PASS {len(oks)}   FAIL {len(fails)}")
for f in fails: print("  FAIL " + f)
if not fails: print("  every table value matches its artefact")
