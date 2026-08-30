# 4. The returned VRAM is re-spent, not returned

The slim DSpark conversion is real and it is a good piece of work. The drafter file goes from
1.95 GB to **631,712,480 bytes**, 602 MiB, which is the 3.1x reduction the maintainers claim.
The sentence this chapter tests is the one that follows it: *"on a 16 GB card the 1.3 GB of
VRAM the slim drafter returns is basically free context."*

The file shrank. The resident cost did not follow it, because a draft model is not only a file.
It gets loaded, it gets compute buffers, and it gets **its own KV cache**, sized to the same
context the main model is serving.

![Resident VRAM against allocated context, with and without the drafter](../figures/f9-drafter-vram-vs-context.png)

## Measured

Resident VRAM read from NVML with the server up and a slot allocated, single card, `-ctk q4_0
-ctv q4_0 -fa on`, which is the configuration a 262k deployment uses.

| allocated context | no drafter | with drafter | drafter costs |
|---|---|---|---|
| 8,192 | 7,893 MiB | 10,629 MiB | 2,736 MiB |
| 12,288 | 7,975 | 10,769 | 2,794 |
| 20,480 | 8,159 | 11,049 | 2,890 |
| 36,864 | 8,527 | 11,609 | 3,082 |
| 135,168 | 10,735 | 14,969 | **4,234** |
| 266,240 | 13,679 | **fails to load** | n/a |

Five points is enough to fit the cost, and it fits well:

> **drafter cost = 2,647 MiB fixed + 11.75 MiB per 1,000 tokens of context**

Residuals across the five measurements are within 7 MiB. Two things in that line deserve
attention. The fixed term is **4.4x the drafter's own file size**, so most of the standing cost
is runtime and compute buffers and not weights. The slope is the draft model's KV cache, held
at its f16 default here.

## What that costs in the currency the claim is denominated in

The claim is about context, so convert both sides into context. The main model's KV costs a
measured 22.46 MiB per 1,000 tokens at `q4_0` KV.

- The slim conversion gives back 1,257 MiB against the old `Q4_1` drafter. That is **56,000
  tokens** of context.
- At a 135,168-token deployment the drafter is holding 4,234 MiB. That is **188,500 tokens** of
  context.

The saving is real and the spend is 3.4 times larger. At that context the drafter is net
**132,500 tokens** of context in the red.

## At the headline context it does not load at all

```
common_fit_params: failed to fit params to free device memory:
n_gpu_layers already set by user to 999, abort
```

Applying the fitted cost at 266,240 tokens gives 13,679 + 5,775 = **19,454 MiB against a
15,888 MiB card**, which is 3,566 MiB over. The extrapolation is offered only to say by how
much; the load failure itself is measured.

That conclusion survives the obvious objection. The draft cache was left at its f16 default,
and `-ctkd` and `-ctvd` exist, so quantising it should cut the 11.75 MiB per 1k slope
substantially. It cannot touch the 2,647 MiB fixed term. Even if the draft KV cost nothing at
all, 13,679 + 2,647 = 16,326 MiB, still over the card. **The slim drafter and a 262k context do
not fit together on 16 GB, and no cache setting changes that.**

Measuring the quantised draft cache is the obvious next experiment and it was not run here.

## What this means beside chapter 3

Chapter 3 showed the drafter's acceptance goes to zero on a long foreign prefix. This chapter
shows that at that same context it is also the single largest consumer of VRAM after the model
itself. Those two facts point the same way and they are the practical recommendation of this
part:

- **Short context, or long self-generated context:** attach the drafter. It roughly doubles
  decode at 8k and still pays 18% to 31% acceptance on the model's own reasoning.
- **Long foreign context, a large document in the prompt:** do not. It returns nothing and it
  costs 2.6 GiB plus 11.75 MiB per 1k.
- **262k on a 16 GB card:** the choice is already made. The drafter does not load.

None of this contradicts the 3.1x file reduction, which is the thing the conversion actually
did. It contradicts the sentence about free context that was attached to it.
