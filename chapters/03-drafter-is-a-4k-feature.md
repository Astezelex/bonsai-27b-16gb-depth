# 3. Speculative decoding is a 4,096-token feature

The slim DSpark drafter is the headline change since Part 1. It is 3.1x smaller than the
`Q4_1` drafter it replaces, 631,712,480 bytes against 1.95 GB, and its published decode uplift
is +73% to +93% on CUDA. That figure reproduces here on the first attempt: **+85.0%**, 42.11
tokens/s to 77.88, measured the way it was published, with a short prompt and an empty cache.

The question this chapter asks is what happens to that number when the cache is not empty,
because a drafter that returns VRAM so it can be spent on context is a long-context feature by
its own description.

![Draft acceptance and decode speed against context depth](../figures/f8-drafter-acceptance-vs-depth.png)

## The measurement

Decode was read from `llama-server`'s own timings at temperature 0 over 300 tokens, three
repetitions, `-fa on -np 1`, single card, which is the method that produced the published
figure. The only change is a prefix in front of the question, sized to a target depth and
verified through the server's `/tokenize` endpoint. Achieved depths land within 0.6% of target.
Wall-clock intervals between repetitions were checked against the server's reported rate and
agree at every depth.

| depth | multiple of 4,096 | acceptance | mean accepted run | no drafter | drafted | uplift |
|---|---|---|---|---|---|---|
| 0 | 0 | 59.2% | 3.36 | 42.11 | 77.88 | **+85.0%** |
| 8,208 | 2.0x | 49.5% | 2.98 | 32.02 | 65.38 | **+104.2%** |
| 16,488 | 4.0x | 18.1% | 1.72 | 25.50 | 35.87 | +40.7% |
| 32,918 | 8.0x | 0.085% | 1.00 | 18.25 | 18.32 | +0.4% |
| 131,050 | 32x | 0.0% | 1.00 | 6.78 | 10.18 | +50.0%, unresolved |
| 261,329 | 64x | n/a | n/a | 3.69 | does not load | n/a |

At 8k the drafter does better than its own specification, +104.2%. At four times the training
window the uplift is +40.7%. At eight times it is gone.

## The cause is in the server's own log

```
llama_context: n_ctx_seq (135168) > n_ctx_train (4096) -- possible training context overflow
common_speculative_impl_draft_dflash: block_size=4, n_extract=5, sample_from_anchor=true
requested draft size (n_max=7) exceeds the trained block size 4 -- clamping to 4
```

`dflash-slim-Q4_0.gguf` carries `n_ctx_train = 4096`. The decay in the table is not mysterious
and it is not a property of speculative decoding in general. It is a 4k-trained draft model
being asked to predict a 27B model's next token from a context sixty-four times longer than
anything it saw in training.

This is worth stating carefully, because it is the difference between a defect and a
limitation. Nothing here says the drafter is broken. It says the drafter's advertised number
and the configuration it is advertised for do not overlap.

## The prefix content matters as much as its length

The table above puts foreign text, held-out Wikipedia, in front of a short question. That is
one composition of a long context and it is not the common one. In a real workload the long
context is usually the model's own output growing under it.

Measured on the same drafter, same card, same build, during the AIME26 quality run:

| generated tokens | acceptance | mean accepted run |
|---|---|---|
| 59,976 | 30.7% | 2.23 |
| 61,212 | 30.6% | 2.22 |
| 71,820 | 27.2% | 2.09 |
| 123,332 | 18.2% | 1.73 |
| 130,039 | 21.1% | 1.85 |

At 130,039 tokens of self-generated reasoning the drafter holds **21.1%** acceptance. At
131,050 tokens of Wikipedia it holds **0.0%**. Same drafter, same hardware, same order of
magnitude of context, and the two numbers are not close.

So the honest statement is not "the drafter dies past 8k". It is that **draft acceptance
depends on what the long context contains at least as much as on how long it is**, and the
model's own continuation is far easier to draft for than an unfamiliar prefix. That reading is
more favourable to the feature than the depth table alone, and it is the one the data supports.

One variable was not held constant between those two measurements: the depth sweep used
`--spec-draft-n-max 7` and the quality run used `4`. Both are clamped to the trained block size
of 4 by the loader, which is visible in the log line above, so the effect of that difference
should be small. It was not isolated, and the claim is stated as composition being the probable
cause and not the proven one.

## The drafter does not cost quality

Speculative decoding is meant to preserve the output distribution. On this stack it does.
AIME26 was run again with the drafter attached, same budget, same items:

| | correct | capped | accuracy when converged |
|---|---|---|---|
| Bonsai, no drafter | 27/30 | 3 | 0.9643 |
| Bonsai with the slim drafter | 26/30 | 3 | **0.9630** |

Identical cap counts, one item apart on raw accuracy, and accuracy-when-converged matching to
a tenth of a point. At temperature 0.7 that is sampling noise. Whatever the drafter costs, it
is not answer quality.

## What a 16 GB owner should take from this

Attach the drafter for interactive work, short prompts, and chat, where it roughly doubles
decode. Attach it for long self-generated reasoning, where it still pays 18% to 31%
acceptance. Do not budget for it at a long foreign context, such as a large document dropped
into the prompt, because there it costs the drafter's own VRAM and returns nothing. Chapter 4
measures what that VRAM actually is.
