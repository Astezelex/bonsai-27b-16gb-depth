#!/usr/bin/env python3
"""Median decode per cell from a sweep TSV. ERROR and LOAD_FAIL cells stay visible."""
import csv, statistics, collections, sys

rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
g, meta, bad = collections.defaultdict(list), {}, {}
for r in rows:
    k = (int(r["depth_target"]), r["center"], r["draft"])
    if r["rep"] == "LOAD_FAIL":
        bad[k] = "LOAD_FAIL"; continue
    if r["decode_tps"] == "ERROR":
        bad[k] = "ERROR: " + r["ttft_ms"][:60]; continue
    g[k].append(float(r["decode_tps"]))
    meta[k] = (r["depth_actual"], r["vram_mib"])

hdr = ("depth", "cen", "dft", "median t/s", "actual", "VRAM")
print("%8s %4s %4s %11s %9s %7s" % hdr)
for k in sorted(set(list(g) + list(bad))):
    d, c, dr = k
    if k in bad:
        print("%8d %4s %4s %11s" % (d, c, dr, bad[k])); continue
    a, v = meta[k]
    print("%8d %4s %4s %11.2f %9s %7s" % (d, c, dr, statistics.median(g[k]), a, v))

print("\nuplift from the drafter, per depth (median of medians):")
for d in sorted({k[0] for k in list(g) + list(bad)}):
    for c in ("off", "on"):
        nd, wd = (d, c, "off"), (d, c, "on")
        if nd in g and wd in g:
            a, b = statistics.median(g[nd]), statistics.median(g[wd])
            print("  depth %7d center=%-3s  %6.2f -> %6.2f  = %+.1f%%" % (d, c, a, b, (b/a-1)*100))
        elif nd in g and wd in bad:
            print("  depth %7d center=%-3s  %6.2f -> %s" % (d, c, statistics.median(g[nd]), bad[wd]))
