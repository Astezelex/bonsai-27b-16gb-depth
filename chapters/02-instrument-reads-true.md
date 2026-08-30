# 2. The instrument still reads true

Every claim in the chapters that follow is a difference: this feature changes that number by
this much at this depth. A difference is only worth reading if the thing doing the measuring
did not move. Between Part 1 and this work the engine was rebased onto current mainline, the
release line changed, and the weight file was replaced. Any of those could have shifted the
baseline, and a shifted baseline would be indistinguishable from a feature effect.

So the quality anchors were re-run first, over the identical item sets, before anything else
was measured.

## The anchors replicate

| benchmark | budget | model | July | August | capped July | capped August |
|---|---|---|---|---|---|---|
| AIME26, n=30 | 60k | Bonsai ternary | 0.867 | 0.900 | 3 | **3** |
| AIME26, n=30 | 60k | Qwen IQ2_XXS | 0.633 | 0.600 | 11 | **11** |
| MMLU-Redux, n=342 | 8k | Bonsai ternary | 0.871 | 0.868 | 6 | **6** |
| MMLU-Redux, n=342 | 8k | Qwen IQ2_XXS | 0.860 | 0.851 | 18 | 20 |
| LiveCodeBench, n=50 | 14k | Bonsai ternary | 0.520 | 0.600 | 23 | 19 |
| LiveCodeBench, n=50 | 14k | Qwen IQ2_XXS | 0.300 | 0.240 | 35 | 38 |

Three of the six cap counts are identical across a build change and a weight-file change. The
two AIME cells, which carry the headline, are identical in both arms.

Part 1's central finding survives the rebase intact. The gap between these two models is a
convergence gap and not a knowledge gap, and the least intuitive part of it reproduces as well:
on MMLU-Redux, accuracy restricted to items the model actually finished **flips to the 2-bit
quant**, 0.901 against 0.879 in August and 0.907 against 0.882 in July. On the problems it
completes, IQ2_XXS is the stronger of the two. It completes fewer of them.

## What the item-level pairing adds

Part 1 counted caps. It did not test them item by item. Both models answered the same
questions, so McNemar on the discordant pairs is the right test. Pairing was keyed on a SHA-1
of the question text as the model received it, never on a row index, and all four runs were
asserted to cover the identical question hashes before any comparison ran.

| test | Bonsai only | Qwen only | net | exact p |
|---|---|---|---|---|
| AIME26 convergence, August | 10 | 1 | +9 | **0.0117** |
| AIME26 convergence, July | 9 | 1 | +8 | **0.0215** |
| AIME26 correctness, items both finished, August | 0 | 0 | 0 | 1.0000 |
| AIME26 correctness, items both finished, July | 0 | 0 | 0 | 1.0000 |

On the 18 AIME items both models finished, both scored 18 of 18, in both runs, with zero
discordant pairs. Convergence separates these models at p = 0.012. Correctness, on the items
where correctness can be observed at all, does not separate them.

**MMLU-Redux should be reported as a tie.** McNemar gives p = 0.451 in August and p = 0.672 in
July. No subject group is significant, and the two smallest p values are both 0.289 pointing in
opposite directions. Subjects showing a difference of at least two items in the same direction
in both independent runs: **0 of 57**.

That last number is worth keeping for a different reason. The two runs are independent draws
over the same questions, so disagreement between them is measured run-to-run variance and not a
modelled one: **9.1% of items change verdict for Bonsai and 9.6% for the quant**, and 59.6% of
the 57 subjects score identically twice. Anyone reading a per-subject table at `--limit 6`
should discount differences below three items entirely.

## Two harness details that a reproducer needs

**`--limit` in evalscope is per subset, not per run.** `mmlu_redux` carries 57 subsets, so
`--limit 6` queues 342 items and `--limit 300` queues about 17,100. The difference is roughly
65 hours per arm on this hardware. The runner here resolves the benchmark's real subset count
from the installed evalscope before launching and refuses to start when the product exceeds a
budget typed out by the caller.

**Accuracy is never reported without its cap rate.** Part 1 stated this as a rule. It is now a
gate: a script that reads a run directory prints accuracy, completion rate, cap rate and
accuracy-given-completion as one block, and exits non-zero above a 5% cap rate. On this data it
fails both AIME arms at 36.7% and 10.0%, fails LiveCodeBench at 38.0% and 76.0%, fails the
MMLU quant arm at 5.8%, and passes MMLU Bonsai at 1.8%. The rule existed in Part 1 as prose and
was still broken during this work before the gate was built.

## Two cautions on the table above

**The LiveCodeBench movement is not a capability change.** Bonsai's 0.520 to 0.600 is a
cap-rate movement, 23 capped down to 19. Accuracy when converged went 0.929 to 0.938, which is
flat inside the noise on n=28 and n=32.

**The weight file is not the same file.** July ran `Q2_0`. That file is refused by b10658, so
the August arm necessarily used `Q2_g64`. Every August number carries that substitution, and it
is why resident VRAM rose 0.37 GiB at matched context. The comparison is model-plus-packaging
and not model alone.

With that established, the baseline is stable enough to attribute what follows to the features
themselves.
