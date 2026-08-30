#!/usr/bin/env python3
"""Refuse an evalscope run whose item count is not what the author thinks it is.

Exists because `--limit` is PER SUBSET, not per run. July hit this twice
(mmlu_redux 57 subsets, live_code_bench 28 subsets) and the anchor rerun of
2026-08-28 hit it a third time by copying the pre-fix July script.

Usage: item_gate.py <dataset> <limit> <budget> [pinned_subset,...]
Prints the resolved arithmetic and exits non-zero if it exceeds the budget.
The budget must be typed out by the caller, so "--limit N" can never silently
be read as "N items".
"""
import sys

from evalscope.api.registry import get_benchmark


def main():
    if len(sys.argv) < 4:
        print("usage: item_gate.py <dataset> <limit> <budget> [pinned,...]")
        return 2
    ds, limit, budget = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    pinned = [s for s in sys.argv[4].split(",") if s] if len(sys.argv) > 4 else []

    bench = get_benchmark(ds)
    all_subsets = list(getattr(bench, "subset_list", []) or [])
    if not all_subsets:
        print(f"GATE FAIL {ds}: could not resolve subset_list from the installed evalscope")
        return 1

    if pinned:
        unknown = [s for s in pinned if s not in all_subsets]
        if unknown:
            print(f"GATE FAIL {ds}: pinned subsets not in the registry: {unknown}")
            return 1
        n = len(pinned)
        how = f"{n} pinned of {len(all_subsets)} available"
    else:
        n = len(all_subsets)
        how = f"all {n} subsets, NOT pinned"

    worst = n * limit
    print(f"GATE {ds}: {how}; --limit {limit} is PER SUBSET -> "
          f"upper bound {n} x {limit} = {worst} items; budget {budget}")
    if worst > budget:
        print(f"GATE FAIL {ds}: {worst} exceeds the declared budget of {budget}. "
              f"Either pin subsets or lower --limit.")
        return 1
    print(f"GATE OK {ds}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
