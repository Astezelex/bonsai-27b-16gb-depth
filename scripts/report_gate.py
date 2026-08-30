#!/usr/bin/env python3
"""Refuse to report an evalscope accuracy figure without its cap rate.

WHY: on 2026-08-29 an AIME26 result of 0.90 against 0.60 was written up as a reasoning-quality
gap. It was a token cap. Qwen IQ2_XXS hit the 60000-token budget on 11 of 30 items and emitted
no answer at all; on the 18 items both models finished, both scored 18/18. Part 1 already
stated "never report accuracy without its cap rate" as a rule, in prose, and the rule was
broken anyway. This is the same rule as a precondition instead of a sentence.

The gate prints accuracy, completion rate, cap rate and accuracy-given-completion as one
block, and exits 1 when the cap rate is above the threshold, so a bare accuracy number cannot
be quoted without the caveat travelling with it.

Usage: report_gate.py <run-dir> <max_tokens> [cap-rate-threshold, default 0.05]
  <run-dir> is the timestamped evalscope directory holding reviews/<model>/*.jsonl
"""
import glob, json, os, sys


def main():
    if len(sys.argv) < 3:
        print(__doc__.strip())
        return 2
    run = sys.argv[1]
    budget = int(sys.argv[2])
    thresh = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    floor = int(budget * 0.98)

    files = sorted(glob.glob(os.path.join(run, "reviews", "*", "*.jsonl")))
    if not files:
        print(f"GATE FAIL: no review jsonl under {run}/reviews/*/")
        return 1

    n = correct = capped = nopred = 0
    completed_correct = completed = 0
    no_tok = 0
    for f in files:
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sc = rec["sample_score"]["score"]
            acc = int(sc["value"]["acc"])
            tok = None
            for m in rec["messages"]:
                if m.get("role") == "assistant" and m.get("perf_metrics"):
                    tok = m["perf_metrics"].get("output_tokens")
            pred = sc.get("extracted_prediction")
            done = pred not in (None, "", "None")
            n += 1
            correct += acc
            if tok is None:
                no_tok += 1
            elif tok >= floor:
                capped += 1
            if not done:
                nopred += 1
            else:
                completed += 1
                completed_correct += acc

    if n == 0:
        print(f"GATE FAIL: {run} parsed 0 items.")
        return 1
    if no_tok == n:
        print(f"GATE FAIL: {run} records no output_tokens, so the cap rate is unknown. "
              f"An accuracy figure from this run is not reportable.")
        return 1

    cap_rate = capped / n
    print(f"RUN            {run}")
    print(f"budget         {budget} tokens, cap floor {floor}")
    print(f"items          {n}")
    print(f"accuracy       {correct}/{n} = {correct/n:.4f}   <- never quote this line alone")
    print(f"completed      {completed}/{n} = {completed/n:.4f}   (emitted an extractable answer)")
    print(f"capped         {capped}/{n} = {cap_rate:.4f}   (output_tokens >= {floor})")
    print(f"no prediction  {nopred}/{n}")
    if completed:
        print(f"acc|completed  {completed_correct}/{completed} = {completed_correct/completed:.4f}")
    if no_tok:
        print(f"WARNING        {no_tok} item(s) carried no output_tokens and are excluded from the cap count")

    if cap_rate > thresh:
        print(f"\nGATE FAIL: cap rate {cap_rate:.1%} exceeds {thresh:.1%}. The accuracy figure "
              f"for this run measures budget as much as capability. Report the split, or "
              f"re-run with a larger budget, before quoting {correct/n:.4f} anywhere.")
        return 1
    print(f"\nGATE OK: cap rate {cap_rate:.1%} is at or below {thresh:.1%}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
