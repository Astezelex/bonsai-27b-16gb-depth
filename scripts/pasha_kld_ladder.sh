#!/bin/bash
# Pasha #1: mean-center quality against uncentered q4 KV, as a function of context.
#
# His words: "how did the q4_0-K + mean-center calibration work out ... in terms of quality
# (vs uncentered q4 KV at your 262k target)". The published answer is a single point at
# n_ctx=512, which is 0.2% of the target. This produces a ladder, and a direct comparison at
# the target itself.
#
# TWO INSTRUMENTS, because one of them cannot reach 262k:
#   A. KLD against an F16 reference. The stronger instrument. Measured today: F16 V alone
#      already fails to allocate at 262k on a 16 GB card (cudaMalloc 10496 MiB), and the base
#      file stores full logits at 247,358 B per scored token. So the KLD ladder stops where
#      the reference fits on one card, and the script finds that point by trying.
#   B. Raw perplexity, centered against uncentered, both at -ctk q4_0 -ctv q4_0, at n_ctx
#      262144. Both arms fit on ONE card at 13587 MiB, no F16 reference needed. Weaker
#      instrument, reaches the actual target, answers the question he asked.
#
# PROVE SMALL THEN SCALE, enforced not remembered:
#   The 512 rung runs FIRST on the original corpus and must reproduce the published
#   -24.6% mean KLD within tolerance. If it does not, the harness is wrong and the ladder is
#   abandoned. Its measured wall time then sets the per-token rate that gates every rung
#   above it. Ledger 2026-08-29T14:30.
set -u
set -o pipefail

D=/mnt/bigdisk/bonsai
BIN=$D/prism-b10658/llama-prism-b10658-4725def
MODELS=$D/rerun-2026-08/models
IMG=ghcr.io/ggml-org/llama.cpp:full-cuda
TERN=$(cat "$D/rerun-2026-08/TERNARY.txt")
W=$D/rerun-2026-08
BIAS=$W/kv-bias-q4_0.gguf
CARD=${CARD:-1}
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=$D/pasha-kld-$STAMP
LOG=$OUT/run.log
TSV=$OUT/kld.tsv
KLDDIR=${KLDDIR:-$D/kldbase}          # big .kld files land here
BYTES_PER_TOKEN=247358                 # MEASURED: 1,519,767,476 B / 6144 tokens
BUDGET_MIN=${BUDGET_MIN:-240}

mkdir -p "$OUT" "$KLDDIR"
say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

case "$CARD" in *,*) say "REFUSING: CARD='$CARD'. Fit and timing here assume one card."; exit 2;; esac
for f in "$BIAS" "$W/eval-heldout.txt" "$D/kld-corpus/en-heldout.txt"; do
  [ -s "$f" ] || { say "REFUSING: missing $f"; exit 2; }
done

ppx(){ # $1 tag, rest: llama-perplexity args. Captures the whole run, never a tail.
  local tag=$1; shift
  local out="$OUT/ppx-$tag.log"
  say "  RUN $tag"
  docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES="$CARD" \
    -v "$BIN":/prism:ro -v "$MODELS":/models -v "$W":/w:ro \
    -v "$D/kld-corpus":/corpus:ro -v "$KLDDIR":/kld -v "$OUT":/out \
    --entrypoint /prism/llama-perplexity "$IMG" \
    -m "/models/$TERN" -ngl 999 "$@" > "$out" 2>&1
  local rc=$?
  echo "  exit=$rc  log=$out  ($(wc -l < "$out") lines)" | tee -a "$LOG"
  return $rc
}

# Targeted extractors. A generic "first float on the line" grabber returns 0.13 from
# "0.13.291.205 I Final estimate: PPL = 1.6671", because llama.cpp prefixes stdout with a
# dotted timestamp. Both functions therefore anchor on the label and take what follows it.
# Verified against the real July log: "Mean    KLD:   0.000646 +/- 0.000025" and
# "0.13.291.205 I Final estimate: PPL = 1.6671 +/- 0.03043".
grab_kld(){ # $1 log -> mean KLD, or PARSE_FAIL
  local v
  v=$(grep -E "^Mean[[:space:]]+KLD:" "$1" | tail -1 \
      | sed -n 's/^Mean[[:space:]]*KLD:[[:space:]]*\([-+]\?[0-9.]\+\).*/\1/p')
  [ -n "$v" ] && echo "$v" || echo "PARSE_FAIL"
}
grab_ppl(){ # $1 log -> final perplexity, or PARSE_FAIL
  local v
  v=$(grep -E "Final estimate: PPL" "$1" | tail -1 \
      | sed -n 's/.*PPL[[:space:]]*=[[:space:]]*\([0-9.]\+\).*/\1/p')
  [ -n "$v" ] && echo "$v" || echo "PARSE_FAIL"
}

printf 'stage\tctx\tchunks\ttokens\tcorpus\tarm\tmetric\tvalue\twall_s\n' > "$TSV"
row(){ printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$TSV"; }

df_free_gb(){ df -BG --output=avail "$KLDDIR" | tail -1 | tr -dc 0-9; }

# ---------------------------------------------------------------- STAGE 0: the control rung
say "### STAGE 0  n_ctx=512 on the ORIGINAL corpus, V left at f16 exactly as July ran it."
say "  target, read from rerun-2026-08/logs/p1_kld.log: uncentered 0.000646, centered 0.000487, -24.6%"
T0=$(date +%s)
ppx base512orig -f /w/eval-heldout.txt -c 512 --chunks 12 --kl-divergence-base /kld/base512-orig.kld
ppx unc512orig  -f /w/eval-heldout.txt -c 512 --chunks 12 -ctk q4_0 \
      --kl-divergence-base /kld/base512-orig.kld --kl-divergence
ppx cen512orig  -f /w/eval-heldout.txt -c 512 --chunks 12 -ctk q4_0 \
      --kv-mean-center /w/kv-bias-q4_0.gguf \
      --kl-divergence-base /kld/base512-orig.kld --kl-divergence
T1=$(date +%s)
UNC=$(grab_kld "$OUT/ppx-unc512orig.log")
CEN=$(grab_kld "$OUT/ppx-cen512orig.log")
say "  uncentered mean KLD = $UNC ; centered = $CEN ; wall $((T1-T0))s"
row control 512 12 6144 orig uncentered mean_kld "$UNC" $((T1-T0))
row control 512 12 6144 orig centered   mean_kld "$CEN" $((T1-T0))

if [ "$UNC" = PARSE_FAIL ] || [ "$CEN" = PARSE_FAIL ]; then
  say "ABORT: could not parse a mean KLD from the control rung. The ladder is not trustworthy."
  say "  Inspect $OUT/ppx-unc512orig.log by eye before changing anything."
  exit 3
fi
DELTA=$(python3 -c "u=$UNC; c=$CEN; print(f'{(c-u)/u*100:.1f}')")
say "  control delta = ${DELTA}% (published: -24.6%)"
row control 512 12 6144 orig delta pct_change "$DELTA" $((T1-T0))
OK=$(python3 -c "print(1 if abs($DELTA - -24.6) <= 6.0 else 0)")
if [ "$OK" != 1 ]; then
  say "ABORT: control rung gives ${DELTA}%, more than 6 points from the published -24.6%."
  say "  Something differs from the July harness. Find it before extending the ladder."
  exit 3
fi
say "  CONTROL PASSED. Per-token rate: $(python3 -c "print(f'{6144/max($((T1-T0)),1):.1f}')") tok/s across 3 arms."
RATE=$(python3 -c "print(max(1.0, 6144/max($((T1-T0)),1)))")

# ---------------------------------------------------------------- ladder plan and gates
# Token counts chosen so every rung carries MORE scored tokens than the published 6144.
LADDER=${LADDER:-"8192:4 32768:2"}
total_tok=0
# ---------------------------------------------------- STAGE 0b: isolate the two deviations
# The ladder differs from July in TWO ways at once: V moves from f16 to q4_0 (because f16 V
# collapses 200x at depth on this model, measured today) and the corpus moves from 25 KB of
# self-generated text to Wikipedia held-out. Changing both at once makes any difference
# unattributable, so each is measured alone at ctx 512 where it is cheap.
say "### STAGE 0b  isolate the V-type change and the corpus change, both at ctx 512"
for iso in "orig:q4v:/w/eval-heldout.txt" "wiki:q4v:/corpus/en-heldout.txt"; do
  tag=${iso%%:*}; rest=${iso#*:}; corpus=${rest#*:}
  t0=$(date +%s)
  ppx "base512-$tag-q4v" -f "$corpus" -c 512 --chunks 12 --kl-divergence-base "/kld/b512$tag.kld"
  ppx "unc512-$tag-q4v"  -f "$corpus" -c 512 --chunks 12 -ctk q4_0 -ctv q4_0 \
        --kl-divergence-base "/kld/b512$tag.kld" --kl-divergence
  ppx "cen512-$tag-q4v"  -f "$corpus" -c 512 --chunks 12 -ctk q4_0 -ctv q4_0 \
        --kv-mean-center /w/kv-bias-q4_0.gguf \
        --kl-divergence-base "/kld/b512$tag.kld" --kl-divergence
  t1=$(date +%s)
  iu=$(grab_kld "$OUT/ppx-unc512-$tag-q4v.log"); ik=$(grab_kld "$OUT/ppx-cen512-$tag-q4v.log")
  idl="PARSE_FAIL"
  [ "$iu" != PARSE_FAIL ] && [ "$ik" != PARSE_FAIL ] && \
    idl=$(python3 -c "print(f'{($ik-$iu)/$iu*100:.1f}')")
  say "  512 $tag q4_0-V: uncentered $iu  centered $ik  delta ${idl}%"
  row isolate 512 12 6144 "$tag" uncentered mean_kld "$iu" $((t1-t0))
  row isolate 512 12 6144 "$tag" centered   mean_kld "$ik" $((t1-t0))
  row isolate 512 12 6144 "$tag" delta      pct_change "$idl" $((t1-t0))
  rm -f "$KLDDIR/b512$tag.kld"
done

say "### ETA and disk gate for the ladder [$LADDER] on the Wikipedia corpus"
for rung in $LADDER; do
  c=${rung%%:*}; n=${rung##*:}; tok=$(( c * n ))
  gb=$(python3 -c "print(f'{$tok*$BYTES_PER_TOKEN/1e9:.1f}')")
  secs=$(python3 -c "print(int($tok*3/$RATE))")
  say "  ctx=$c chunks=$n tokens=$tok  base .kld ~${gb} GB  ~$((secs/60)) min for 3 arms"
  total_tok=$(( total_tok + tok ))
done
NEED_GB=$(python3 -c "print(int($total_tok*$BYTES_PER_TOKEN/1e9)+5)")
FREE_GB=$(df_free_gb)
say "  total .kld need ~${NEED_GB} GB, free on $KLDDIR = ${FREE_GB} GB"
if [ "$NEED_GB" -ge "$FREE_GB" ]; then
  say "REFUSING: the base logit files do not fit. Point KLDDIR at /mnt/archive or cut LADDER."
  exit 3
fi
EST_MIN=$(python3 -c "print(int($total_tok*3/$RATE/60))")
say "  ladder ETA ${EST_MIN} min, budget ${BUDGET_MIN} min"
[ "$EST_MIN" -gt "$BUDGET_MIN" ] && { say "REFUSING: over budget."; exit 3; }

# ---------------------------------------------------------------- STAGE 1: the ladder
for rung in $LADDER; do
  c=${rung%%:*}; n=${rung##*:}; tok=$(( c * n ))
  say "### LADDER ctx=$c chunks=$n tokens=$tok (Wikipedia held-out, CC BY-SA)"
  t0=$(date +%s)
  if ! ppx "base${c}" -f /corpus/en-heldout.txt -c "$c" --chunks "$n" \
        --kl-divergence-base "/kld/base-$c.kld"; then
    say "  F16 REFERENCE FAILED TO LOAD OR RUN at ctx=$c. That is the ladder's ceiling."
    grep -iE "cudaMalloc|out of memory|failed to allocate" "$OUT/ppx-base${c}.log" | tail -4 | tee -a "$LOG"
    row ladder "$c" "$n" "$tok" wiki f16_base fit FAILED 0
    break
  fi
  ppx "unc${c}" -f /corpus/en-heldout.txt -c "$c" --chunks "$n" -ctk q4_0 -ctv q4_0 \
        --kl-divergence-base "/kld/base-$c.kld" --kl-divergence
  ppx "cen${c}" -f /corpus/en-heldout.txt -c "$c" --chunks "$n" -ctk q4_0 -ctv q4_0 \
        --kv-mean-center /w/kv-bias-q4_0.gguf \
        --kl-divergence-base "/kld/base-$c.kld" --kl-divergence
  t1=$(date +%s)
  u=$(grab_kld "$OUT/ppx-unc${c}.log")
  k=$(grab_kld "$OUT/ppx-cen${c}.log")
  d="PARSE_FAIL"
  [ "$u" != PARSE_FAIL ] && [ "$k" != PARSE_FAIL ] && \
    d=$(python3 -c "print(f'{($k-$u)/$u*100:.1f}')")
  say "  ctx=$c  uncentered $u  centered $k  delta ${d}%  wall $((t1-t0))s"
  row ladder "$c" "$n" "$tok" wiki uncentered mean_kld "$u" $((t1-t0))
  row ladder "$c" "$n" "$tok" wiki centered   mean_kld "$k" $((t1-t0))
  row ladder "$c" "$n" "$tok" wiki delta      pct_change "$d" $((t1-t0))
  rm -f "$KLDDIR/base-$c.kld"      # freed only after both arms have consumed it
  say "  freed $KLDDIR/base-$c.kld"
done

# ---------------------------------------------------------- STAGE 2: the target itself
# No F16 reference, so no .kld file and no fit problem. Both arms are q4_0 KV and both fit
# on one card at the measured 13587 MiB.
CTX262=${CTX262:-262144}; N262=${N262:-2}
say "### STAGE 2  raw perplexity at n_ctx=$CTX262, centered against uncentered, one card"
say "  no F16 reference: the KLD instrument cannot reach this context on 16 GB (measured)"
t0=$(date +%s)
ppx "ppl262-unc" -f /corpus/en-heldout.txt -c "$CTX262" --chunks "$N262" -ctk q4_0 -ctv q4_0
ppx "ppl262-cen" -f /corpus/en-heldout.txt -c "$CTX262" --chunks "$N262" -ctk q4_0 -ctv q4_0 \
      --kv-mean-center /w/kv-bias-q4_0.gguf
t1=$(date +%s)
pu=$(grab_ppl "$OUT/ppx-ppl262-unc.log")
pc=$(grab_ppl "$OUT/ppx-ppl262-cen.log")
say "  ctx=$CTX262  uncentered PPL $pu  centered PPL $pc  wall $((t1-t0))s"
row target "$CTX262" "$N262" $(( CTX262 * N262 )) wiki uncentered ppl "$pu" $((t1-t0))
row target "$CTX262" "$N262" $(( CTX262 * N262 )) wiki centered   ppl "$pc" $((t1-t0))

say "=== KLD LADDER COMPLETE ==="
column -t -s $'\t' "$TSV" | tee -a "$LOG"
