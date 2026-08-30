# 7. Engine parity, packaging, and one missing metadata key

Three claims in this chapter are not about depth. They are checks on statements the maintainers
made about the engine and the files, and they belong here because a reader deciding what to
download needs them. None of the three needs a figure.

## The fork and mainline gap is halved, not gone

Part 1 measured the fork running standard, non-ternary quants 4.5% slower than mainline. The
maintainers state that the rebase removes this "by construction", since the non-ternary paths
are now identical to mainline.

The test is one IQ2_XXS file, one card, identical flags, run A/B/A/B so that any thermal or
clock drift falls on both builds equally. Mainline is the prebuilt `full-cuda` image at build
10588, commit `70adb1b4c`; the fork is prism b10658.

| build | pass | prefill, pp512 | decode, tg128 |
|---|---|---|---|
| fork b10658 | 1 | 970.06 | 34.47 |
| mainline b10588 | 1 | 974.29 | 35.32 |
| fork b10658 | 2 | 961.27 | 34.37 |
| mainline b10588 | 2 | 971.49 | 35.23 |

Means: fork 965.67 and 34.42, mainline 972.89 and 35.28.

**Decode skew is 2.42%, down from 4.47%. Prefill skew is 0.74%.** The gap halved and it did not
close. The separation is many standard deviations wide: the two mainline decode figures differ
from each other by 0.09 tok/s and the two fork figures by 0.10, against a between-build gap of
0.86.

One detail is worth more than the headline. The narrowing came mostly from mainline moving
**down**, 35.8 to 35.28, and not from the fork moving up, 34.2 to 34.42. Whatever closed half
the gap, it was not the fork getting faster.

Caveat that ships with this: mainline 10588 against fork 10658 is seventy builds apart. It is
close to like-for-like and it is not exactly like-for-like. A fork built from its own merge base
would settle the remaining 2.42% and was not built here.

## PQ2_0 is smaller than claimed and also faster

The maintainers describe `PQ2_0` as "about 6% smaller, same weights, an exact re-encoding". Two
thirds of that checks out and the third part was not tested.

| file | bytes | prefill, pp512 | decode, tg128 |
|---|---|---|---|
| `Ternary-Bonsai-27B-Q2_g64.gguf` | 7,585,330,240 | 1,027.57 | 44.35 |
| `Ternary-Bonsai-27B-PQ2_0.gguf` | 7,165,121,600 | **1,040.45** | **46.02** |

**5.54% smaller**, matching "about 6%". It is also 3.8% faster at decode and 1.3% faster at
prefill, which the maintainers do not claim and which is worth knowing.

For a 16 GB owner the 400.7 MiB it frees is, at the measured 22.46 MiB per 1,000 tokens of KV,
about **17,800 more tokens of context**. That is a real gain for a re-download.

What was not tested is the behavioural half of the claim. "Same weights, exact re-encoding"
predicts identical output at temperature 0, and this repository already has an identity harness
that would answer it. It was not run. Chapter 9 lists it with the other open items.

## Two July-era defects are gone

Part 1 could not use two of the three weight files. Both now work, and each was checked for
non-empty generated output and not merely for a zero exit code.

| file | July | August |
|---|---|---|
| `Ternary-Bonsai-27B-Q2_0.gguf` | the file Part 1 used | **refused by v7** |
| `Ternary-Bonsai-27B-Q2_g64.gguf` | reported as gibberish upstream | **works, 1,745 coherent characters** |
| `Ternary-Bonsai-27B-PQ2_0.gguf` | loaded nowhere, `invalid ggml type 142` | **works, 1,747 coherent characters** |

The old `Q2_0` being refused is a deliberate migration and not a regression, and it is the one
change that breaks existing setups. Anyone pinned to that file has to re-download. The cost of
the newer file is 0.42 GiB, which is why resident VRAM at a 262k context moved from 12.9 to
13.27 GiB between the two parts.

## Packaging and build, both as described

`Q2_g64` loads on b10658 and the old `Q2_0` is refused with a pointer to the replacement, as
stated. The prebuilt CUDA release asset initialises on sm_120 with no source build:

```
version: 0.2.0-dev (build 10658, commit 4725def0a)
Device 0: NVIDIA GeForce RTX 5060 Ti, compute capability 12.0, VMM: yes, 15888 MiB
```

For a Blackwell owner that is the single biggest practical change in the whole diff. Part 1 had
to carry a warning about building inside a CUDA devel container with the GPU attached at
configure time, because a GPU-less configure caches `CUDA_DRIVER=NOTFOUND` and the build then
fails in a way that does not name its cause. That entire step is gone.

The one part of the build claim that does not hold is the CUDA version. The maintainers wrote
that native CUDA 13.3 builds for Linux and Windows "are landing in the release matrix". The
matrix carries **12.4 and 12.8**. Everything here runs on 12.8, so nothing was blocked by it,
and a reader planning around a 13.3 asset should not.

## A mismatched bias is accepted, and the format is why

The maintainers state that calibrating the centering bias with `-ctk q4_0` is required "so the
K-rotation state matches the serving config", and that "a mismatched bias is rejected at load
with a clear message".

The first half is enforced. The second half is not, and an earlier draft of this work repeated
the maintainers' sentence without testing it. It is corrected here, not dropped.

A **structurally corrupt** bias is rejected, cleanly and with a good message:

```
load_kv_mean_center: failed to load K-cache mean-centering bias file from bias-truncated.gguf
llama_init_from_model: failed to initialize the context: failed to load K-cache mean-centering bias file
```

A bias **calibrated at `-ctk q8_0` and then served at `-ctk q4_0`** loads without complaint.
That is the mismatch the sentence describes. The two files are genuinely different: different
SHA-256, and 35,409 of 66,400 bytes changed.

Dumping the metadata shows why no loader could catch it. A bias GGUF carries exactly two keys:

```
general.type          str  = kv-mean-center
kv_mean_center.k_rot  bool = 1
```

There is no field recording the cache type the bias was calibrated with, so there is nothing
for the loader to compare against the serving configuration. The rejection cannot be
implemented on this format as it stands, which makes this a format gap and not a missing `if`.

The fix is one key. Writing `kv_mean_center.cache_type_k` at calibration time and checking it at
load would close it, and would turn a silent wrong-rotation-state into the clear message the
documentation already promises. That is worth doing: a user who follows the instructions and
calibrates against the wrong cache type currently gets a model that runs, produces plausible
text, and is quietly using a bias fitted to a different quantisation.
