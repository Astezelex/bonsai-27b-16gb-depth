# 6. Why nobody had measured this

Three chapters have now said the same thing about three different features: the published
number is a short-context number, and the feature behaves differently at the context it is sold
for. Stated that way it sounds like an accusation of carelessness. It is not, and this chapter
is the reason.

**The depth-0 number is, for one of the four quantities that matter, the only number the shipped
tooling can produce.**

![What can be measured at the target context](../figures/f11-what-can-be-measured.png)

## The quality instrument has a hard ceiling, and it is host RAM

`llama-perplexity` is how quality is measured in this ecosystem, and it is what produces the
KL-divergence numbers in chapter 5. Before it scores anything it allocates a buffer of
`n_ctx x n_vocab x 4` bytes. For a 151,936-token vocabulary that is:

| n_ctx | host memory reserved |
|---|---|
| 512 | 0.31 GB |
| 8,192 | 4.98 GB |
| 16,384 | 9.96 GB |
| 32,768 | **19.91 GB** |
| 65,536 | 39.83 GB |
| 131,072 | 79.66 GB |
| 262,144 | **159.32 GB** |

This box has 31 GB. The 16,384 rung completes. The 32,768 rung was killed by the kernel:

```
Out of memory: Killed process 2358839 (llama-perplexit)
total-vm:109215068kB, anon-rss:25539436kB
```

A plain perplexity run at `n_ctx 262144` died the same way, `std::bad_alloc` thrown from
`std::vector<float>::reserve`.

The ceiling is a property of the tool and the vocabulary, not of the graphics card. A 16 GB
card can hold the model and a 262k KV cache comfortably, as chapter 4 measured at 13,679 MiB.
It is the host that cannot hold the logits. **Validating this feature at its own headline
context requires roughly 160 GB of system RAM**, which is not a home-lab machine and is not
what the people adopting a 16 GB configuration have.

## The speed instrument cannot exercise the features

`llama-bench` is how throughput is measured, and it does have a depth flag, `-d`. Reading its
help on the build under test:

- `-d, --n-depth` is present
- `-ctk`, `-ctv` are present
- there is **no** flag for a mean-centering bias
- there is **no** flag for a draft model

So the one tool that can put tokens in the cache before it measures cannot switch on either of
the two features whose depth behaviour this part is about. Everything in chapters 3 and 5 had
to be driven through `llama-server` and read from its own timings and counters, which is a
different instrument with different overheads and needs a prefix built and verified by hand.

That is not a small inconvenience. It is the difference between a number anyone can produce
with one command and a number that needs a harness.

## What that adds up to

For a 16 GB owner deciding whether to adopt these features at 262k context, four things are
worth knowing. Three of them are measurable on this hardware and one is not:

| quantity | instrument | reaches 262k | why not |
|---|---|---|---|
| resident VRAM | NVML with the server up | yes | |
| decode speed | `llama-server` timings | yes | |
| draft acceptance | `llama-server` draft counters | to 131k | the drafted server does not load on 16 GB |
| **quality** | `llama-perplexity` | **to 16k** | 159 GB of host RAM at the target |

The one that cannot be reached is the one the K-cache feature exists to protect. An ecosystem
that publishes a short-context quality figure for a long-context feature is not being lazy. It
is publishing the number its instrument returns.

## The fix is small and it is upstream

None of this requires new science. `llama-perplexity` reserves the full logit buffer because
`--kl-divergence-base` writes every logit to disk, a format that also costs 247,358 bytes per
scored token here. A streaming or chunked path that accumulates the divergence statistics
without holding all logits at once would move the ceiling from 16k to wherever the KV cache
runs out, which on this card is 262k. A depth flag on the bias and draft-model options in
`llama-bench` would remove the need for a bespoke harness entirely.

Until one of those exists, the honest thing for anyone publishing a long-context feature is to
say at what context the number was taken. That is the whole of this part's methodological
claim, and chapter 1 stated it before any of the measurements: **report metric@depth, not
metric**. Not because the depth-0 number is dishonest, but because it is the only one available,
and a reader deploying at 262k has no way to know that from the number alone.
