# 9. The box, the rules, and how to redo any of it

## The box

Deliberately unimpressive, as in Part 1, because the question is what these files do on
hardware people actually own.

```
host     Intel i5-4670K, 4 cores, 31 GB DDR3-1333
cards    2x NVIDIA RTX 5060 Ti, 16,311 MiB physical each, 15,888 MiB usable at init
         PCIe 3.0 x8 per card. Every measurement in this part is SINGLE CARD.
engine   prism b10658, commit 4725def, linux-cuda-12.8-x64 prebuilt release asset
mainline ghcr.io/ggml-org/llama.cpp:full-cuda, build 10588, commit 70adb1b4c
```

Every speed number here was taken on one card with `CUDA_VISIBLE_DEVICES` pinned, and the
harness refuses to start if more than one device is visible. Splitting a 27B model across two
cards over PCIe 3.0 x8 moves activations at every layer boundary, so a cross-card throughput
figure is not comparable to the single-card numbers Part 1 published, and it answers a question
nobody asked: the entire 16 GB story is one card.

## The files

| file | bytes | role |
|---|---|---|
| `Ternary-Bonsai-27B-Q2_g64.gguf` | 7,585,330,240 | the model under test |
| `Ternary-Bonsai-27B-PQ2_0.gguf` | 7,165,121,600 | the fork's re-encoding, chapter 7 |
| `dflash-slim-Q4_0.gguf` | 631,712,480 | the slim drafter, `n_ctx_train` 4,096 |
| `kv-bias-q4_0.gguf` | 66,400 | centering bias, calibrated `-ctk q4_0 -c 512 --chunks 200` |
| `Qwen3.6-27B-UD-IQ2_XXS.gguf` | 9,388,779,744 | the baseline, chapter 2 |
| `en-calib.txt` | 2,027,937 | Wikipedia, calibration half |
| `en-heldout.txt` | 9,028,371 | Wikipedia, evaluation half, disjoint |

The Wikipedia corpora are CC BY-SA and were built by streaming a dump and stopping at a byte
target, with the two halves drawn from disjoint article ranges. The builder is in `scripts/`.
The original self-generated corpus from Part 2 is kept because the control rung in chapter 5
needs it to reproduce the published number.

## Honesty rules, continued from Part 1

Part 1's five rules stand unchanged. Budgets are part of the result, raw JSONs go in the PR,
same sampling for both models, per-card numbers never averaged, negative results are
contributions. This part adds four, and every one of them exists because it was broken during
this work before it was written down.

- **Depth is part of the result.** Never report a long-context feature's effect without the
  context depth it was measured at. This is chapter 1 and it is the whole point.
- **Synthetic padding is valid only for content-independent metrics.** Before padding a
  context, name the metric and say whether it depends on what the tokens say. VRAM, fit and
  allocation do not. Acceptance, perplexity, KL divergence and accuracy do. The first depth
  sweep here padded with 28 unique words shuffled at random and produced an acceptance curve
  that was not attributable to depth at all.
- **Prove the smallest cell before running the sweep, and state the ETA out loud.** The depth
  sweep now refuses to start without a recorded single-cell run beside it, and prints its own
  per-cell arithmetic against a budget typed by the caller. The n=1 run caught two harness
  bugs that would each have produced a confident false finding about the fork.
- **Parse against a real log, never against a remembered format.** `Final estimate: PPL =
  1.6671` is prefixed by a dotted timestamp, so a generic "first float on the line" extractor
  returns `0.13`. A loose `VRAM: (\d+) MiB` matches `Total VRAM: 31777 MiB` and reports two
  cards as one. Every extractor here was run against a kept log before it was used.
- **When a data file and the code reading it disagree, never fix it in the data file.** After
  any change to a generator and consumer pair, re-run the documented path into a scratch
  directory and diff its output against the committed artefact. An earlier revision of this
  repo hand-patched `figdata-part3.json` and the renderer while leaving the emitter alone. Every
  figure looked correct, because the figure was correct. The published Quickstart produced one
  figure and a `KeyError`. Looking at an output cannot see a broken input path.

Two mechanical gates enforce the older rules that prose did not. `report_gate.py` refuses to
print an accuracy figure without its cap rate and exits non-zero above 5%. `item_gate.py`
resolves a benchmark's real subset count from the installed evalscope and refuses to launch
when `n_subsets x --limit` exceeds a declared budget, because `--limit` is per subset and
getting that wrong is a 57x error on MMLU-Redux.

## Redoing any of it

```
scripts/pasha_depth_sweep.sh      chapters 3 and 4: depth x centering x drafter, single card
scripts/pasha_kld_ladder.sh       chapter 5: the KLD ladder and the isolation rungs
scripts/report_gate.py            accuracy with its cap rate, or a non-zero exit
scripts/mk_part3_figdata.py       figdata-part3.json from the raw artefacts
scripts/make_figures_part3.py     figures 8 to 11 from figdata-part3.json
```

The sweep takes about 100 minutes on one card and prints its own estimate before it starts. The
ladder takes about two hours and aborts if its control rung fails to reproduce the published
-24.6% within tolerance, which is the check that says the harness is wired correctly before
anything new is measured.

No figure in this part contains a hardcoded number. `figdata-part3.json` is generated from the
review JSONLs, the sweep TSVs and the ladder output, and the figure script reads only that.

## What would be most useful from someone else

A card with more than 16 GB, so the drafter and a 262k context can be measured together at all.
A host with 64 GB or more, so the KLD ladder reaches past 16,384. And chapter 8's item 1, a bias
recalibrated at each rung, which needs no special hardware and would settle whether
mean-centering is short-context by nature or only by default.

## Provenance and disclosure

Run design, scripts and analysis were AI-orchestrated (Claude Opus 5) and human-directed,
reviewed and operated. Every number is reproducible from the scripts and raw artefacts here.
Models tested are the vendors' published GGUF artifacts. This part was produced after the
maintainers asked for a re-test on the rebased branch; they did not review it before
publication.

No affiliation with, and no funding from, PrismML, Alibaba/Qwen or any vendor.

License: MIT.
