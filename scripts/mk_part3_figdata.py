#!/usr/bin/env python3
"""Build figdata-part3.json from the run artefacts. Figures never hardcode a number."""
import csv, glob, json, os, re, statistics

R = os.environ.get("BONSAI_ROOT", "/mnt/bigdisk/bonsai")
out = {}

# ---- depth sweep: natural filler. Two runs cover the five depths. -----------------------
SWEEPS = [f"{R}/pasha-depth-20260829-162441/results.tsv",
          f"{R}/pasha-depth-20260829-190835/results.tsv"]
cells = {}
for tsv in SWEEPS:
    src = os.path.dirname(tsv)
    for r in csv.DictReader(open(tsv), delimiter="\t"):
        k = (int(r["depth_target"]), r["center"], r["draft"])
        c = cells.setdefault(k, {"tps": [], "vram": None, "actual": None,
                                 "fail": False, "src": src})
        if r["rep"] == "LOAD_FAIL" or r["decode_tps"] == "ERROR":
            c["fail"] = True
            continue
        c["tps"].append(float(r["decode_tps"]))
        c["vram"] = int(r["vram_mib"]) if r["vram_mib"] else c["vram"]
        c["actual"] = int(r["depth_actual"]) if r["depth_actual"] else c["actual"]

ACC = re.compile(r"= ([0-9.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([0-9.]+)")
def acceptance(src, depth, center):
    p = f"{src}/acceptance-d{depth}-c{center}.txt"
    if not os.path.exists(p):
        return None
    m = ACC.search(open(p).read())
    return {"acceptance": float(m.group(1)), "accepted": int(m.group(2)),
            "generated": int(m.group(3)), "mean_len": float(m.group(4))} if m else None

depths = sorted({k[0] for k in cells})
rows = []
for d in depths:
    nd, wd = cells.get((d, "off", "off")), cells.get((d, "off", "on"))
    row = {"depth_target": d,
           "depth_actual": (nd or {}).get("actual"),
           "nodraft_tps": round(statistics.median(nd["tps"]), 3) if nd and nd["tps"] else None,
           "draft_tps":   round(statistics.median(wd["tps"]), 3) if wd and wd["tps"] else None,
           "nodraft_vram": (nd or {}).get("vram"),
           "draft_vram": (wd or {}).get("vram"),
           "draft_load_failed": bool(wd and wd["fail"] and not wd["tps"]),
           "acceptance": acceptance((wd or nd or {}).get("src", ""), d, "off")}
    # centering arms, for chapter 5's speed claim
    cn, cw = cells.get((d, "on", "off")), cells.get((d, "on", "on"))
    row["center_nodraft_tps"] = round(statistics.median(cn["tps"]), 3) if cn and cn["tps"] else None
    row["center_nodraft_vram"] = (cn or {}).get("vram")
    row["center_draft_tps"] = round(statistics.median(cw["tps"]), 3) if cw and cw["tps"] else None
    rows.append(row)
out["depth_sweep"] = rows

# ---- real-workload acceptance, from the phase-3 runner log ------------------------------
log = f"{R}/anchors-phase23-20260828-230834.log"
out["workload_acceptance"] = [
    {"acceptance": float(m.group(1)), "accepted": int(m.group(2)),
     "generated": int(m.group(3)), "mean_len": float(m.group(4))}
    for m in (ACC.search(l) for l in open(log)) if m
] if os.path.exists(log) else []

# ---- KLD ladder -------------------------------------------------------------------------
kld = {}
for f in glob.glob(f"{R}/pasha-kld-*/kld.tsv"):
    for r in csv.DictReader(open(f), delimiter="\t"):
        kld[(r["stage"], int(r["ctx"]), r["corpus"], r["arm"])] = r["value"]
lad = []
for stage, ctx, corpus, vtype in [("control", 512, "orig", "f16"),
                                  ("isolate", 512, "orig", "q4_0"),
                                  ("isolate", 512, "wiki", "q4_0"),
                                  ("ladder", 8192, "wiki", "q4_0"),
                                  ("ladder", 16384, "wiki", "q4_0")]:
    u = kld.get((stage, ctx, corpus, "uncentered"))
    c = kld.get((stage, ctx, corpus, "centered"))
    if u and c and u not in ("PARSE_FAIL",) and c not in ("PARSE_FAIL",):
        u, c = float(u), float(c)
        lad.append({"n_ctx": ctx, "corpus": corpus, "v_cache": vtype,
                    "uncentered": u, "centered": c,
                    "delta_pct": round((c - u) / u * 100, 2)})
out["kld_ladder"] = lad

# ---- host-RAM ceiling -------------------------------------------------------------------
VOCAB = 151936
out["perplexity_reserve"] = [{"n_ctx": c, "gb": round(c * VOCAB * 4 / 1e9, 2)}
                             for c in (512, 8192, 16384, 32768, 65536, 131072, 262144)]
out["host_ram_gb"] = 31
out["card_vram_mib"] = 15888
out["drafter_file_bytes"] = 631712480
out["draft_n_ctx_train"] = 4096

print(json.dumps(out, indent=1))
