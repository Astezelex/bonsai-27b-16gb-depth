# 8. What did not resolve

Everything below was measured and left open. It is collected in one place because a reader
deciding how much weight to put on the rest of this part is entitled to know where it stops,
and because two of these items are worth more to whoever runs them next than anything already
settled here.

## 1. The centering bias was fitted at 512 tokens, and nobody knows if that is the problem

The single most valuable follow-up in this part. Chapter 6 shows the centering benefit going
-9.5%, -0.7%, +0.7% across n_ctx 512, 8,192 and 16,384. Every one of those rungs uses a bias
calibrated at n_ctx 512, which is `llama-kv-mean-center`'s default and the value in its own
usage line.

So the decay could be either of two very different things. The correction itself might not
generalise across sequence length, in which case the feature is short-context by nature. Or the
per-channel means might simply drift with position, in which case a bias fitted at the serving
context would work fine and only the **default** is wrong.

Those have opposite implications for the maintainers and the measurement that separates them is
cheap: recalibrate at each rung and re-run the ladder, roughly two GPU-hours. It was not run.

## 2. A 50% speedup with zero draft acceptance

At a 131,050-token prefix the drafted server produced 300 tokens in 31 seconds against 45
seconds undrafted, a real speedup confirmed by wall clock between repetitions, and reproduced
in two independent runs with different prefix content. Over the same generations the server's
own counter reported `0 accepted / 1186 generated, mean len = 1.00`.

Both numbers are reproducible. Neither explains the other: with no accepted draft tokens the
drafted path should be slower, since it pays for draft forwards and gains nothing.

Server logs were captured for both arms and show no configuration difference beyond the
expected draft-model lines. The candidates not eliminated are that the acceptance counter is
wrong at long context, that `predicted_per_second` is computed over a different denominator in
the drafted path, or that batched verification is genuinely cheaper per token at 131k for a
reason unrelated to acceptance. **The +50.0% figure is drawn in the figure and marked as
unresolved, and it should not be quoted as a drafter benefit.**

## 3. Prefix composition against draft length were not separated

Chapter 3 reports 0.0% acceptance at a 131k Wikipedia prefix and 21.1% at 130k tokens of
self-generated reasoning, and attributes the difference to what the context contains. Two
things differed between those measurements: the composition, and `--spec-draft-n-max`, which
was 7 in the depth sweep and 4 in the workload run.

Both are clamped to the trained block size of 4 by the loader, which the server log confirms,
so the flag difference should be inert. It was not tested. Composition is stated as the
probable cause and not the isolated one.

## 4. PQ2_0 as an exact re-encoding

The size claim holds at 5.54% and the speed is better than claimed. "Same weights, exact
re-encoding" is a behavioural claim that predicts identical output at temperature 0, and this
repository has the identity harness that would test it. It was not run.

## 5. Quality at the headline context

Not an open question so much as a closed door, and chapter 7 is about it. Quality at 262,144
tokens cannot be measured with `llama-perplexity` below roughly 160 GB of host RAM. The ladder
stops at 16,384. Every quality statement in this part is bounded by that.

## 6. The remaining fork and mainline gap

Chapter 8 measures 2.42% decode skew where the maintainers expect zero. The measurement is
solid, the cause is not identified, and mainline b10588 against fork b10658 is seventy builds
apart. A fork built from its own merge base would settle whether the residual is real or an
artefact of that distance.

## 7. The quantised draft cache

Chapter 4 fits the drafter's cost at 2,647 MiB fixed plus 11.75 MiB per 1,000 tokens, with the
draft cache at its f16 default. `-ctkd` and `-ctvd` exist and would cut the slope. The
conclusion that the drafter and a 262k context do not fit together on 16 GB survives without
that measurement, because the fixed term alone puts the total over the card. The measurement
would still sharpen the recommendation at every context below 262k, and it was not made.

## What this list is for

Five of the seven items are experiments that cost less than a day of GPU time between them.
They are written out with their commands in `scripts/` so that anyone with the same card, or a
different one, can close them. That is the same bargain Part 1 made: the interesting part of a
home-lab benchmark is not the numbers, it is that somebody else can produce them.
