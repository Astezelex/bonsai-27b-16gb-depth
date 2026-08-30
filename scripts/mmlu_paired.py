#!/usr/bin/env python3
"""Item-level paired analysis of MMLU-Redux across the two models and the two runs.

The report JSON only carries per-subject aggregates, so the earlier group comparison used an
unpaired two-proportion test. That throws away the strongest fact about this design: both
models answered the SAME 342 questions. McNemar on the discordant pairs is the correct test
and is far more powerful.

Pairing is keyed on a SHA1 of the question text, never on the row index. Index-based pairing
would silently compare different questions if evalscope ever reordered a subset, and the
whole result would look fine. Every pairing is asserted, and a mismatch is fatal.
"""
import glob, hashlib, json, os, sys
from math import comb

R = os.environ.get("BONSAI_ROOT", "/mnt/bigdisk/bonsai")
RUNS = {
    ("bonsai", "aug"): f"{R}/anchors-2026-08/results/bonsai-mmlu_redux/20260829_055510/reviews/bonsai",
    ("qwen",   "aug"): f"{R}/anchors-2026-08/results/qwen-iq2xxs-mmlu_redux/20260829_055510/reviews/qwen-iq2xxs",
    ("bonsai", "jul"): f"{R}/results/evalscope/bonsai-mmlu_redux/20260716_060917/reviews/bonsai",
    ("qwen",   "jul"): f"{R}/results/evalscope/qwen-iq2xxs-mmlu_redux/20260716_060917/reviews/qwen-iq2xxs",
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


def qhash(rec):
    """SHA1 of the first user turn, which is the question as the model saw it."""
    for m in rec["messages"]:
        if m.get("role") == "user":
            return hashlib.sha1(m["content"].encode()).hexdigest()[:12]
    raise KeyError("no user message")


def load(dirpath):
    if not os.path.isdir(dirpath):
        sys.exit(f"MISSING reviews dir {dirpath}")
    out = {}
    files = sorted(glob.glob(os.path.join(dirpath, "mmlu_redux_*.jsonl")))
    if not files:
        sys.exit(f"NO review jsonl under {dirpath}")
    for f in files:
        subj = os.path.basename(f)[len("mmlu_redux_"):-len(".jsonl")]
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            acc = rec["sample_score"]["score"]["value"]["acc"]
            if acc not in (0.0, 1.0):
                sys.exit(f"non-binary acc {acc} in {f}")
            out[(subj, qhash(rec))] = int(acc)
    return out


runs = {k: load(v) for k, v in RUNS.items()}
for k, v in runs.items():
    print(f"loaded {k[0]:7} {k[1]}  {len(v)} items  correct {sum(v.values())}")

keys = set(runs[("bonsai", "aug")])
for k, v in runs.items():
    if set(v) != keys:
        d1 = len(set(v) - keys); d2 = len(keys - set(v))
        sys.exit(f"ITEM SET MISMATCH {k}: {d1} extra, {d2} missing. Pairing refused.")
print(f"\nGATE: all four runs cover the identical {len(keys)} question hashes. Pairing is verified.")

keys = sorted(keys)


def mcnemar(a, b, label):
    """a, b: dicts key->0/1. Exact two-sided binomial on discordant pairs."""
    n01 = sum(1 for k in keys if a[k] == 0 and b[k] == 1)   # b right, a wrong
    n10 = sum(1 for k in keys if a[k] == 1 and b[k] == 0)   # a right, b wrong
    m = n01 + n10
    if m == 0:
        print(f"  {label:34} no discordant pairs")
        return
    p = min(1.0, 2 * sum(comb(m, i) for i in range(min(n01, n10) + 1)) / 2 ** m)
    print(f"  {label:34} a-only {n10:3d}  b-only {n01:3d}  discordant {m:3d}"
          f"  net {n10-n01:+3d}  exact p = {p:.3f}")


print("\nMcNEMAR, bonsai vs qwen on identical items (a = bonsai, b = qwen):")
for era in ("aug", "jul"):
    mcnemar(runs[("bonsai", era)], runs[("qwen", era)], f"all 342 items, {era}")

print("\nMcNEMAR by group, August:")
for g in ("STEM", "Humanities", "Social sciences", "Other"):
    sub = [k for k in keys if GROUP.get(k[0]) == g]
    a, b = runs[("bonsai", "aug")], runs[("qwen", "aug")]
    n01 = sum(1 for k in sub if a[k] == 0 and b[k] == 1)
    n10 = sum(1 for k in sub if a[k] == 1 and b[k] == 0)
    m = n01 + n10
    p = min(1.0, 2 * sum(comb(m, i) for i in range(min(n01, n10) + 1)) / 2 ** m) if m else 1.0
    print(f"  {g:16} n={len(sub):3d}  bonsai-only {n10:3d}  qwen-only {n01:3d}"
          f"  discordant {m:3d}  net {n10-n01:+3d}  exact p = {p:.3f}")

print("\nTEST-RETEST, same model across the two runs (a = Aug, b = Jul):")
for mdl in ("bonsai", "qwen"):
    mcnemar(runs[(mdl, "aug")], runs[(mdl, "jul")], f"{mdl}, Aug vs Jul")

print("\nPER-ITEM FLIP RATE, same model, two runs at temperature 0.7:")
for mdl in ("bonsai", "qwen"):
    flips = sum(1 for k in keys if runs[(mdl, "aug")][k] != runs[(mdl, "jul")][k])
    print(f"  {mdl:7} {flips:3d} of {len(keys)} items changed verdict = {flips/len(keys)*100:.1f}%")

both = sum(1 for k in keys if runs[("bonsai","aug")][k] == runs[("qwen","aug")][k] == 1)
neither = sum(1 for k in keys if runs[("bonsai","aug")][k] == runs[("qwen","aug")][k] == 0)
print(f"\nAgreement, August: both correct {both}, both wrong {neither}, "
      f"agree {both+neither}/{len(keys)} = {(both+neither)/len(keys)*100:.1f}%")
