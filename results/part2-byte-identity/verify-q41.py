#!/usr/bin/env python3
"""Direct Q4_1-vs-bf16 comparison on the ARCHIVED July artefacts.

Part 1 published: "Q4_1 and bf16 outputs are byte-identical at temp 0 (5/5 prompts)".
phase3b_drafter.sh never tested that. It compared each drafter against the BASELINE and
reported two verdicts; the q41-vs-bf16 comparison was inferred from the two verdicts
matching. Two strings can both differ from a third and differ from each other.

The generations were kept, so the claim can be checked directly, offline, right now.

Captures BOTH content and reasoning_content and reports them separately, because the
July harness used `content or reasoning_content`, which silently prefers one. If a
generation carries text in both fields, the two methods compare different strings.
"""
import glob, hashlib, json, os, sys

R = "/mnt/bigdisk/bonsai/results"
PROMPTS = ["r1", "r2", "c1", "c2", "r3"]
TAGS = {"baseline": "baseline", "q41": "dspark-q41", "bf16": "dspark-bf16"}
EMPTY = hashlib.sha256(b"").hexdigest()


def load(tag, pid):
    fs = sorted(glob.glob(f"{R}/draft-{tag}-t0-r1-{pid}-*.json"))
    if not fs:
        return None
    d = json.load(open(fs[-1]))
    ch = d["choices"][0]
    m = ch["message"]
    c = m.get("content") or ""
    rc = m.get("reasoning_content") or ""
    return {
        "file": os.path.basename(fs[-1]),
        "content": c,
        "reasoning": rc,
        "july": (c or rc or ""),          # exactly what phase3b compared
        "both": (rc + c),                 # the fixed method: nothing discarded
        "finish": ch.get("finish_reason"),
    }


def h(s):
    return hashlib.sha256(s.encode()).hexdigest()[:12]


rows, bad = [], []
for pid in PROMPTS:
    got = {k: load(v, pid) for k, v in TAGS.items()}
    if any(g is None for g in got.values()):
        bad.append(f"{pid}: missing artefact for " +
                   ",".join(k for k, g in got.items() if g is None))
        continue
    if any(h(g["july"]) == EMPTY[:12] or not g["july"] for g in got.values()):
        bad.append(f"{pid}: an empty capture, refusing to compare")
        continue
    b, q, f = got["baseline"], got["q41"], got["bf16"]
    rows.append({
        "pid": pid,
        "chars": len(b["july"]),
        "q_vs_b": q["july"] == b["july"],
        "f_vs_b": f["july"] == b["july"],
        "q_vs_f_july": q["july"] == f["july"],
        "q_vs_f_both": q["both"] == f["both"],
        "rc_used": bool(not b["content"] and b["reasoning"]),
        "finish": (b["finish"], q["finish"], f["finish"]),
    })

if bad:
    print("PROBLEMS:")
    for x in bad:
        print("  " + x)

if not rows:
    print("NO COMPARABLE ROWS. Claim cannot be checked from these artefacts.")
    sys.exit(2)

print(f"{'pid':4} {'chars':>6} {'q41==base':>10} {'bf16==base':>11} "
      f"{'q41==bf16':>10} {'q41==bf16(+rc)':>15} {'reasoning_only':>15}  finish(b,q,f)")
for r in rows:
    print(f"{r['pid']:4} {r['chars']:6d} {str(r['q_vs_b']):>10} {str(r['f_vs_b']):>11} "
          f"{str(r['q_vs_f_july']):>10} {str(r['q_vs_f_both']):>15} "
          f"{str(r['rc_used']):>15}  {r['finish']}")

n = len(rows)
agree = sum(r["q_vs_f_july"] for r in rows)
agree_both = sum(r["q_vs_f_both"] for r in rows)
qb = sum(r["q_vs_b"] for r in rows)
print()
print(f"PUBLISHED CLAIM: Q4_1 == bf16, 5/5 at temp 0")
print(f"  directly measured now: {agree}/{n} (July capture method)")
print(f"  with reasoning+content: {agree_both}/{n}")
print(f"  (drafted vs undrafted, for context: {qb}/{n} identical)")
