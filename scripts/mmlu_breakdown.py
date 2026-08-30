#!/usr/bin/env python3
"""Per-subject MMLU-Redux breakdown for the Part 2 anchors.

Four report artefacts: bonsai and qwen-iq2xxs, July and August. Each subject carries n=6,
so a single subject's score moves in steps of 1/6 = 16.7 points. A per-subject table read
naively will therefore invent differences that are not there. Two guards against that:

  1. The July run is an independent draw at the same temperature, so the per-subject
     |August - July| spread for the SAME model is a measured noise floor, not a modelled one.
  2. A model difference is only called out when it reproduces in BOTH runs and clears that
     floor. One-run gaps are reported in the table and nowhere else.

Writes a full markdown table to disk and prints a digest.
"""
import json, os, sys

R = os.environ.get("BONSAI_ROOT", "/mnt/bigdisk/bonsai")
SRC = {
    ("bonsai", "aug"): f"{R}/anchors-2026-08/results/bonsai-mmlu_redux/20260829_055510/reports/bonsai/mmlu_redux.json",
    ("qwen",   "aug"): f"{R}/anchors-2026-08/results/qwen-iq2xxs-mmlu_redux/20260829_055510/reports/qwen-iq2xxs/mmlu_redux.json",
    ("bonsai", "jul"): f"{R}/results/evalscope/bonsai-mmlu_redux/20260716_060917/reports/bonsai/mmlu_redux.json",
    ("qwen",   "jul"): f"{R}/results/evalscope/qwen-iq2xxs-mmlu_redux/20260716_060917/reports/qwen-iq2xxs/mmlu_redux.json",
}

GROUP = {}
for g, names in {
 "STEM": """abstract_algebra anatomy astronomy college_biology college_chemistry
   college_computer_science college_mathematics college_physics computer_security
   conceptual_physics electrical_engineering elementary_mathematics high_school_biology
   high_school_chemistry high_school_computer_science high_school_mathematics
   high_school_physics high_school_statistics machine_learning""",
 "Humanities": """formal_logic high_school_european_history high_school_us_history
   high_school_world_history international_law jurisprudence logical_fallacies
   moral_disputes moral_scenarios philosophy prehistory professional_law world_religions""",
 "Social sciences": """econometrics high_school_geography high_school_government_and_politics
   high_school_macroeconomics high_school_microeconomics high_school_psychology
   human_sexuality professional_psychology public_relations security_studies sociology
   us_foreign_policy""",
 "Other": """business_ethics clinical_knowledge college_medicine global_facts human_aging
   management marketing medical_genetics miscellaneous nutrition professional_accounting
   professional_medicine virology""",
}.items():
    for nm in names.split():
        GROUP[nm] = g


def load(path):
    """subject -> (correct, n). Refuses to guess: a missing file is fatal, not skipped."""
    if not os.path.exists(path):
        sys.exit(f"MISSING ARTEFACT {path}")
    d = json.load(open(path))
    out = {}
    for m in d["metrics"]:
        if m.get("name") != "mean_acc":
            continue
        for c in m.get("categories", []):
            for s in c.get("subsets", []):
                if s.get("is_aggregate"):
                    continue
                n = s["num"]
                out[s["name"]] = (round(s["score"] * n), n)
    if not out:
        sys.exit(f"NO SUBSETS PARSED from {path}")
    return out


data = {k: load(v) for k, v in SRC.items()}

subjects = sorted(data[("bonsai", "aug")])
for k, v in data.items():
    if sorted(v) != subjects:
        sys.exit(f"SUBJECT SET MISMATCH in {k}: {len(v)} vs {len(subjects)}")
ns = {n for v in data.values() for (_, n) in v.values()}
if ns != {6}:
    sys.exit(f"UNEXPECTED per-subject n: {ns}")
print(f"GATE: 4 artefacts, {len(subjects)} subjects each, n=6 per subject, sets identical.")
for k, v in data.items():
    tot = sum(c for c, _ in v.values())
    print(f"  {k[0]:7} {k[1]}  {tot}/{len(subjects)*6} = {tot/(len(subjects)*6):.4f}")

# ---- measured noise floor: same model, two independent runs -------------------
retest = []
for mdl in ("bonsai", "qwen"):
    for s in subjects:
        retest.append(abs(data[(mdl, "aug")][s][0] - data[(mdl, "jul")][s][0]))
n_ret = len(retest)
mean_ret = sum(retest) / n_ret
dist = {d: retest.count(d) for d in sorted(set(retest))}
print(f"\nNOISE FLOOR, test-retest |Aug - Jul| on the SAME model, {n_ret} subject-observations:")
print(f"  mean {mean_ret:.2f} items of 6")
for d in sorted(dist):
    print(f"  delta {d}/6 : {dist[d]:3d}  ({dist[d]/n_ret*100:4.1f}%)")
ge = {d: sum(v for k, v in dist.items() if k >= d) / n_ret for d in sorted(dist)}
for d in sorted(ge):
    print(f"  P(|delta| >= {d}/6) = {ge[d]*100:.1f}%")

# ---- group aggregates ---------------------------------------------------------
print("\nGROUP AGGREGATES (bonsai vs qwen, August), correct/items:")
rows = []
for g in ("STEM", "Humanities", "Social sciences", "Other"):
    subs = [s for s in subjects if GROUP.get(s) == g]
    r = {}
    for mdl in ("bonsai", "qwen"):
        for era in ("aug", "jul"):
            r[(mdl, era)] = sum(data[(mdl, era)][s][0] for s in subs)
    tot = len(subs) * 6
    rows.append((g, len(subs), tot, r))
    ba, qa = r[("bonsai", "aug")], r[("qwen", "aug")]
    bj, qj = r[("bonsai", "jul")], r[("qwen", "jul")]
    print(f"  {g:16} n={tot:3d}  bonsai {ba:3d} ({ba/tot:.3f}) vs qwen {qa:3d} ({qa/tot:.3f})"
          f"  diff {ba-qa:+3d}   [July diff {bj-qj:+3d}]")

# ---- paired sign test across subjects, August ---------------------------------
w = l = t = 0
for s in subjects:
    a, b = data[("bonsai", "aug")][s][0], data[("qwen", "aug")][s][0]
    if a > b: w += 1
    elif a < b: l += 1
    else: t += 1
print(f"\nPAIRED SIGN TEST, August, bonsai vs qwen across {len(subjects)} subjects:")
print(f"  bonsai higher {w}, qwen higher {l}, tied {t}")
try:
    from math import comb
    m = w + l
    p = sum(comb(m, k) for k in range(min(w, l) + 1)) / 2 ** m * 2
    print(f"  two-sided sign-test p = {min(p,1.0):.3f} on the {m} untied subjects")
except Exception as e:
    print(f"  sign test unavailable: {e}")

# ---- differences that REPRODUCE across both runs -------------------------------
print("\nREPRODUCIBLE per-subject differences (same sign in July AND August, both >= 2/6):")
hits = []
for s in subjects:
    da = data[("bonsai", "aug")][s][0] - data[("qwen", "aug")][s][0]
    dj = data[("bonsai", "jul")][s][0] - data[("qwen", "jul")][s][0]
    if da * dj > 0 and abs(da) >= 2 and abs(dj) >= 2:
        hits.append((min(abs(da), abs(dj)), s, da, dj))
for _, s, da, dj in sorted(hits, key=lambda r: -r[0]):
    who = "bonsai" if da > 0 else "qwen"
    print(f"  {s:36} Aug {da:+d}/6  Jul {dj:+d}/6   favours {who}")
print(f"  {len(hits)} of {len(subjects)} subjects")

# ---- full table to disk --------------------------------------------------------
out = f"{R}/anchors-2026-08/MMLU-PER-SUBJECT.md"
with open(out, "w") as fh:
    fh.write("# MMLU-Redux per subject, correct of 6\n\n")
    fh.write(f"Generated from the four report artefacts, {len(subjects)} subjects, n=6 each.\n")
    fh.write("A single item is 16.7 points, so column-to-column gaps below 2 items carry no\n")
    fh.write("information. The July columns are an independent draw at the same temperature\n")
    fh.write("and serve as the measured noise floor.\n\n")
    fh.write("| subject | group | bonsai Aug | qwen Aug | bonsai Jul | qwen Jul | Aug diff | Jul diff |\n")
    fh.write("|---|---|---|---|---|---|---|---|\n")
    for s in subjects:
        ba = data[("bonsai", "aug")][s][0]; qa = data[("qwen", "aug")][s][0]
        bj = data[("bonsai", "jul")][s][0]; qj = data[("qwen", "jul")][s][0]
        fh.write(f"| {s} | {GROUP.get(s,'?')} | {ba} | {qa} | {bj} | {qj} | {ba-qa:+d} | {bj-qj:+d} |\n")
    fh.write("\n## Group aggregates\n\n")
    fh.write("| group | subjects | items | bonsai Aug | qwen Aug | Aug diff | Jul diff |\n|---|---|---|---|---|---|---|\n")
    for g, nsub, tot, r in rows:
        fh.write(f"| {g} | {nsub} | {tot} | {r[('bonsai','aug')]} ({r[('bonsai','aug')]/tot:.3f}) "
                 f"| {r[('qwen','aug')]} ({r[('qwen','aug')]/tot:.3f}) "
                 f"| {r[('bonsai','aug')]-r[('qwen','aug')]:+d} "
                 f"| {r[('bonsai','jul')]-r[('qwen','jul')]:+d} |\n")
print(f"\nWROTE {out}  ({os.path.getsize(out)} B)")
