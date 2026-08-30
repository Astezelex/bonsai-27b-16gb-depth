# 5. The byte-identical property does not hold

This is the one finding in this part that is not about depth, and it is the one most worth a
maintainer's attention.

The documented behaviour is that with the slim dflash drafter, output stays byte-identical to
undrafted decoding at temperature 0 "by construction". Speculative decoding is supposed to be
an optimisation with no semantic effect: the drafter proposes, the full model verifies, and
anything the verifier rejects is discarded, so the greedy path should be reproduced exactly.

**That does not reproduce here.**

## The experiment

Five fixed prompts, `temperature 0`, `top_k 1`, `seed 1234`, thinking disabled, and
`max_tokens 900` so generations finish naturally instead of being cut off. Outputs are
compared by SHA-256 of the exact string, with `reasoning_content` and `content` concatenated,
because this model emits into the former. No normalisation of any kind: a speculative decoder
that is correct must reproduce the greedy path byte for byte.

## Controls first, because a divergence result without them is worthless

| control | result | what it rules out |
|---|---|---|
| greedy against greedy, same server process | **5/5 identical** | engine nondeterminism |
| greedy against greedy, fresh server process | **5/5 identical** | state carried across restarts |
| drafted against drafted, same process, every block size | **5/5 identical** | a flaky drafter |

The engine is deterministic and each drafted configuration is deterministic. Any difference
between the drafted and undrafted arms is therefore systematic and reproducible.

## Result, swept across block size

| `--spec-draft-n-max` | effective | self-consistent | identical to greedy | tok/s |
|---|---|---|---|---|
| 1 | 1 | 5/5 | **3/5** | 63.68 |
| 2 | 2 | 5/5 | **3/5** | 77.99 |
| 4 | 4 | 5/5 | **2/5** | 81.37 |
| 7 | **clamped to 4** | 5/5 | 2/5 | 81.14 |
| greedy baseline | | | | 44.51 |

The drafter's trained `block_size` is 4, so a requested 7 is clamped and those two rows are the
same configuration. Their divergence maps are identical, which confirms it. Effective values
were read back from the server's own log, never assumed.

Per prompt, where `.` is bit-exact and `X` diverges:

```
prompt   n=1  n=2  n=4
   1      X    X    X
   2      .    .    .
   3      .    .    X
   4      .    .    .
   5      X    X    X
```

**Divergence occurs even at block size 1**, and that is the informative part. If it appeared
only at larger blocks, accumulated drift across a block would be the natural suspect. At a
block size of one, the suspect is the verification and acceptance step itself.

The divergences are fluent alternatives, not corruption:

```
p1 @char  129  "propagation of splits upward"   ->  "propagation of splits up the tree"
p3 @char 1487  "| Step | Flags Sent | Key ..."  ->  "| Step | Flag(s) Set | Direction |"
p5 @char   61  "...only if it exceeds a size threshold"
               "...if it exceeds a given size threshold"
```

A divergence at character 1487 shows this is not confined to the opening tokens.

## Acceptance points at the rejection path

| n-max | overall acceptance | on diverging prompts | on identical prompts | gap | mean run |
|---|---|---|---|---|---|
| 1 | 0.8311 | 0.7831 | 0.8632 | +0.0801 | 1.83 |
| 2 | 0.7416 | 0.6795 | 0.7829 | +0.1034 | 2.49 |
| 4 | 0.5781 | 0.5063 | 0.6858 | +0.1795 | 3.31 |

At all three block sizes the prompts that diverge have **lower** draft acceptance than the ones
that stay bit-exact, and the gap widens with block size. Divergence tracks rejections, not
acceptances. Where the drafter guesses well the greedy path is preserved; where it guesses
badly and the verifier must reject and roll back more often, the output drifts. That points at
the rejection and rollback path.

The mean accepted run rising 1.83, 2.49, 3.31 independently confirms the block-size flag was
honoured, agreeing with the server's own configuration log.

## What this does not establish

- **Not a rate.** 3 of 5 has a Wilson 95% interval of [23.1%, 88.2%]. The claim is that the
  property does not hold, not how often it fails. Nobody should quote a percentage from n=5.
- **Not a significance test.** The acceptance correlation is a paired observation over five
  prompts at three settings. A 3-of-3 ordering arises by chance with probability 0.125. It is a
  strong hint and not a proven correlation.
- **Not a quality claim.** Different is not worse. Both outputs read as competent and neither
  was scored.

## A correction to Part 1, and a correction to that correction

Part 1 reported that the older `Q4_1` drafter produced outputs byte-identical at temperature 0
on 5 of 5 prompts.

An earlier draft of this work flagged that claim as unverified, assuming it came from a harness
bug in which five empty `content` fields hashed to the same value and reported a perfect score.
**That assumption was wrong, and it is corrected here instead of being quietly removed.**

The July generations were kept, so the check was made against them directly. Part 1's harness
read `content or reasoning_content`, so it captured the text this model actually emits: the
baselines run 1,213 to 2,002 characters, and every one of the fifteen generations carried its
output in `reasoning_content` with `content` empty. Part 1 was never comparing empty strings.

The July method did have a real defect, a different one. Its script loaded the baseline, the
`Q4_1` output and the `bf16` output, then printed only `q41 == baseline` and `bf16 == baseline`.
**The two drafters were never compared with each other.** The published pair claim was inferred
from the two verdicts agreeing, which is sound for the three prompts where both equal the
baseline, since equality is transitive, and unsound for the two where both merely differ from
it. So three of the five were proven and two were asserted.

Re-measured directly from the archived generations:

| prompt | chars | q41 == base | bf16 == base | **q41 == bf16** |
|---|---|---|---|---|
| r1 | 1,329 | yes | yes | **yes** |
| r2 | 1,530 | yes | yes | **yes** |
| c1 | 2,002 | yes | yes | **yes** |
| c2 | 1,793 | no | no | **yes** |
| r3 | 1,213 | no | no | **yes** |

**The claim holds, 5 of 5.** The two prompts where both drafters leave the undrafted path land
on the same output as each other, which is the interesting case and the one that had only been
inferred.

Script and raw output for that re-measurement: `results/part2-byte-identity/verify-q41.py` and
`verify-q41-output.txt`. The identity harness itself is `scripts/identity.py`.

One limit the original posts did not state, and this one does: all fifteen generations returned
`finish_reason=length` at `max_tokens=512`. That is byte-identity **over the first 512 tokens**,
not over complete outputs. The divergence measured above appears as late as character 1,487, so
the old claim and the new one do not contradict each other. The old drafter was never tested at
the length where this property is stressed.
