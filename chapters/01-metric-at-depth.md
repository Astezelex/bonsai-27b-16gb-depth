# 1. The claim: report metric@depth, not metric

Part 1 of this benchmark made one methodological argument. A benchmark score published without
the token budget it was measured at describes the budget as much as it describes the model, and
the same weights can be presented as a collapse or as a win by moving the cap: 0.48 against 0.28
at a 28k budget, 0.87 against 0.63 at 60k, same two models, same thirty problems. The proposal
was that rows for thinking-mode quants carry accuracy and cap-rate together at a stated budget.

This part makes the same argument on a different axis, because the axis has moved. The features
that shipped on this stack since July are long-context features. K-cache mean-centering exists
so that a q4_0 K cache stays usable. The slim DSpark drafter is presented as returning VRAM that
becomes context. The headline for the whole configuration is 262,144 tokens on one 16 GB card.

Each of those is published as a single number. Every one of those numbers was measured with an
empty or nearly empty KV cache.

| feature | the published number | the same thing, measured at depth |
|---|---|---|
| DSpark speculative decoding | +73% to +93% decode | **+85.0%** at depth 0, **+104.2%** at 8,208, **+0.4%** at 32,918 |
| K-cache mean-centering | -24.6% mean KL divergence | **-24.6%** at n_ctx 512, **-0.7%** at 8,192, **+0.7%** at 16,384 |
| the drafter's returned VRAM | "basically free context" | costs **2,736 MiB** at 8k, **4,234 MiB** at 135k, **does not load** at 262k |

None of the published numbers is wrong. Each reproduced here on the first attempt, on the
hardware and at the depth it was measured at. The drafter's +85.0% at depth 0 sits inside the
maintainers' stated band. The centering result reproduced to the digit: uncentered mean KLD
0.000646, centered 0.000487.

The problem is that all three are depth-conditioned quantities reported as scalars, and the
depth they were measured at is the one depth nobody deploys at. A reader on a 16 GB card who
adopts these features to reach 262k context is reading numbers taken at a context of roughly
zero.

## Why the depth-0 number is the only one anyone publishes

This is the part that makes the argument structural instead of a complaint. The tooling that
ships alongside these features cannot produce a depth-N number.

`llama-perplexity`, which is how quality is measured, allocates a host-memory buffer of
`n_ctx x n_vocab x 4` bytes. For this model that is 5.0 GB at n_ctx 8,192, 19.9 GB at 32,768,
and **159.3 GB at 262,144**. A 32 GB host is killed by the kernel at the 32k rung. Quality at
the context this feature was built for is not measurable on any machine without roughly 160 GB
of RAM.

`llama-bench`, which is how speed is measured, does have a depth flag, `-d`. It has no flag for
the centering bias and no flag for a draft model. So the two features whose depth behaviour
matters most are exactly the two that the depth-capable tool cannot exercise. Measuring them at
depth requires driving `llama-server` and reading its own timings, which is what was done here.

An ecosystem that publishes depth-0 scalars for long-context features is not being careless. It
is publishing the only numbers its instruments return.

## What this claim does not say

- **Not that the features are useless.** Speculative decoding is worth +104% at 8k, which is
  better than its own specification, and mean-centering costs at most 0.24% of decode across every
  depth measured, and exactly 2 MiB of VRAM at every context from 8,192 to 266,240. Both findings are in this part too.
- **Not that the maintainers measured badly.** Every number they published reproduced. The
  disagreement is about what a scalar covers, not about whether it is true.
- **Not that depth is the only hidden variable.** Chapter 3 shows that draft acceptance depends
  on what the long context *contains*, not only how long it is, and the difference between a
  foreign prefix and self-generated reasoning is larger than the difference between 8k and 33k.
- **Not a general claim about llama.cpp.** Everything here is one model family, one fork, two
  cards, one host. Chapter 10 states the box and the limits.

## The shape of the rest

Chapter 2 establishes that the instrument still reads true: Part 1's quality anchors replicate
on the rebased engine with identical cap counts, and the rebase moved decode by 0.8%, so any
change measured afterwards belongs to the features and not to the harness. Chapters 3 and 4 take
the drafter to depth and then weigh what it costs. Chapter 5 is the one finding here that is not
about depth at all, a documented byte-identity property that does not hold. Chapter 6 does the
same for mean-centering. Chapter 7 measures the tooling ceiling that explains why none of this
had been done. Chapters 8 through 10 cover engine parity and packaging, the seven things that
did not resolve, and how to redo any of it.
