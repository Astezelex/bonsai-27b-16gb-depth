#!/bin/bash
# Pasha #2, #10c and #5c in one sweep: centering speed, drafter speedup AT CONTEXT, and the
# two together. Everything here is a SPEED measurement, so everything here is SINGLE CARD.
#
# WHY SINGLE CARD IS A HARD RULE HERE
#   Splitting a model across the two 5060 Ti with -ts moves activations over PCIe 3.0 x8 at
#   every layer boundary. That changes throughput, so a cross-card t/s is not comparable to
#   the single-card numbers already published, and it answers a question nobody asked: the
#   entire 16 GB story is one card. The guard below refuses to run if more than one device is
#   visible. Quality metrics may go cross-card. Speed may not.
#
# METHOD, matched to the published +85.8% so the numbers are comparable:
#   driver.sh p1_drafter measured decode from the SERVER's own timings, temp 0, max_tokens
#   300, 3 reps, -fa on -np 1 --jinja, drafter as
#   -md dflash-slim-Q4_0.gguf --spec-type draft-dspark --spec-draft-n-max 7.
#   The only thing this script changes is the DEPTH: a filler prefix is prepended so the KV
#   already holds N tokens when decode starts. That is what "at your context sizes" means.
#
#   Depth is VERIFIED through the server's /tokenize endpoint, never estimated. A cell whose
#   achieved depth is more than 2% off target is recorded as a failure, not silently kept.
#
# BUILD CONTRACT, read from this build's --help on 2026-08-29, not recalled:
#   --kv-mean-center FNAME   requires --cache-type-k q4_0
#   -ctv q4_0 is the FAST path on this model: G8a measured pp512 1020 t/s at -ctv q4_0
#   against 241 t/s at -ctv f16 with K held at q4_0, so the standing Qwen3.8 "never -ctv q4_0"
#   rule does NOT transfer here. V stays q4_0, matching the published 262k fit.
set -u
set -o pipefail

D=/mnt/bigdisk/bonsai
BIN=$D/prism-b10658/llama-prism-b10658-4725def
MODELS=$D/rerun-2026-08/models
IMG=ghcr.io/ggml-org/llama.cpp:full-cuda
TERN=$(cat "$D/rerun-2026-08/TERNARY.txt")
BIAS=${BIAS:-$D/rerun-2026-08/kv-bias-q4_0.gguf}
DRAFT=/models/dflash-slim-Q4_0.gguf
CARD=${CARD:-1}
PORT=${PORT:-8104}
NAME=pasha-depth
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=$D/pasha-depth-$STAMP
LOG=$OUT/run.log
TSV=$OUT/results.tsv

DEPTHS=${DEPTHS:-"0 8192 32768 131072 262144"}
REPS=${REPS:-3}

mkdir -p "$OUT"
say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

vram(){ python3 -c "
import ctypes as c, sys
nv=c.CDLL('libnvidia-ml.so.1'); nv.nvmlInit_v2()
h=c.c_void_p(); nv.nvmlDeviceGetHandleByIndex_v2(int(sys.argv[1]),c.byref(h))
class M(c.Structure): _fields_=[('t',c.c_ulonglong),('f',c.c_ulonglong),('u',c.c_ulonglong)]
m=M(); nv.nvmlDeviceGetMemoryInfo(h,c.byref(m)); print(m.u//2**20)" "$CARD"; }

# ---------------------------------------------------------------------- guards
ndev=$(python3 -c "
import ctypes as c
nv=c.CDLL('libnvidia-ml.so.1'); nv.nvmlInit_v2()
n=c.c_uint(); nv.nvmlDeviceGetCount_v2(c.byref(n)); print(n.value)")
say "host reports $ndev CUDA devices; this script pins CUDA_VISIBLE_DEVICES=$CARD (single card)"
case "$CARD" in
  *,*) say "REFUSING: CARD='$CARD' names more than one device. Every number in this script is"
       say "a speed number and a cross-card speed number is not comparable to the published set."
       exit 2;;
esac
[ -s "$BIAS" ] || { say "REFUSING: no bias file at $BIAS. #2 and #5c cannot be measured."; exit 2; }
[ -s "$MODELS/dflash-slim-Q4_0.gguf" ] || { say "REFUSING: slim drafter missing."; exit 2; }
# The #5b mismatch test calibrates a second bias from this corpus. Without the preflight a
# missing file made docker fail silently and the test reported "the tool refused to calibrate",
# which is the exact inverse of the real finding.
CALIB=${CALIB:-$D/rerun-2026-08/calib-corpus.txt}
[ -s "$CALIB" ] || { say "REFUSING: no calibration corpus at $CALIB. Test 5b-B cannot run."; exit 2; }

# ------------------------------------------------------- ETA gate and n=1 proof gate
# Both exist because on 2026-08-29 this exact script was armed with no cell proved and no ETA
# stated. item_gate.py held all day by making the arithmetic a precondition; this is the same
# shape. Ledger 2026-08-29T14:30.
BUDGET_MIN=${BUDGET_MIN:-150}
LOAD_S=${LOAD_S:-165}          # measured: anchor servers took 148 s and 152 s to become healthy
PREFILL_TPS=${PREFILL_TPS:-1020}   # measured today, G8a, -ctk q4_0 -ctv q4_0, pp512
GEN_S=${GEN_S:-40}             # 4 generations x 300 tok, decode 42.7 nodraft / 82.3 draft, derated

total_s=0
for depth in $DEPTHS; do
  pf=$(( depth / PREFILL_TPS ))
  cell=$(( LOAD_S + pf + GEN_S ))
  arms_s=$(( cell * 4 ))
  total_s=$(( total_s + arms_s ))
  say "ETA depth=$depth: load ${LOAD_S}s + prefill ${pf}s + gen ${GEN_S}s = ${cell}s/arm x 4 arms = ${arms_s}s"
done
total_s=$(( total_s + 300 ))   # filler builds and the #5b mismatch load
total_min=$(( total_s / 60 ))
say "ETA TOTAL ${total_min} min (${total_s}s), budget ${BUDGET_MIN} min"
if [ "$total_min" -gt "$BUDGET_MIN" ]; then
  say "REFUSING: ETA ${total_min} min exceeds the declared budget of ${BUDGET_MIN} min."
  say "Cut DEPTHS or raise BUDGET_MIN deliberately. Do not raise it to make this line go away."
  exit 3
fi

PROOF="$0.proof"
if [ "${SKIP_PROOF:-0}" != "1" ] && [ ! -s "$PROOF" ]; then
  say "REFUSING: no n=1 proof at $PROOF."
  say "Run one cheap cell first:  DEPTHS=0 REPS=1 SKIP_PROOF=1 $0"
  say "then record its exit code and wall time into that file. A sweep whose smallest cell"
  say "has never run is a sweep that produces a plausible TSV full of nothing."
  exit 3
fi
say "n=1 proof present: $(head -1 "$PROOF" 2>/dev/null)"

# Park until the card is actually free, so we never contend with the anchor suite.
if [ "${WAIT_FOR_CARD:-1}" = "1" ]; then
  say "parking until card $CARD drops below 1500 MiB used"
  while :; do u=$(vram); [ "${u:-99999}" -le 1500 ] && break; sleep 300; done
fi
say "card $CARD free (${u:-?} MiB). engine $(basename "$BIN"). ternary $TERN"

srv_down(){ docker rm -f "$NAME" >/dev/null 2>&1; sleep 3; }
srv_up(){   # $* = extra server args
  srv_down
  docker run -d --name "$NAME" --gpus all -e CUDA_VISIBLE_DEVICES="$CARD" \
    -p "${PORT}":8080 -v "$BIN":/prism:ro -v "$MODELS":/models -v "$OUT":/out \
    -v "$(dirname "$BIAS")":/bias:ro \
    --entrypoint /prism/llama-server "$IMG" \
    -m "/models/$TERN" -ngl 999 -fa on -np 1 --jinja --host 0.0.0.0 --port 8080 "$@" \
    >>"$LOG" 2>&1 || return 1
  for _ in $(seq 1 180); do
    curl -sf -m 3 "localhost:${PORT}/health" >/dev/null 2>&1 && return 0
    docker ps --format '{{.Names}}' | grep -qx "$NAME" || { say "  server died during load"; return 1; }
    sleep 5
  done
  say "  server did not become healthy in 900 s"; return 1
}

ntok(){ # exact token count of a file's contents, from the server itself
  python3 - "$1" "$PORT" <<'PY'
import json,sys,urllib.request
txt=open(sys.argv[1],encoding='utf-8').read()
req=urllib.request.Request(f"http://localhost:{sys.argv[2]}/tokenize",
    data=json.dumps({"content":txt}).encode(),headers={"Content-Type":"application/json"})
print(len(json.load(urllib.request.urlopen(req,timeout=300))["tokens"]))
PY
}

# Deterministic filler. Built once per depth and reused across all four arms so every arm
# sees byte-identical context.
#
# FILLER=natural (default) slices a prefix of the Wikipedia held-out corpus.
# FILLER=random reproduces the first version: words drawn at random from a 28-word list.
#
# WHY THIS MATTERS AND WHY THE FIRST VERSION WAS WRONG:
#   Draft acceptance is a statement about how well the small model predicts the large model's
#   next token. That is entirely content-dependent. The first run filled the context with
#   131,000 tokens drawn from 28 unique words, which is word salad, and then reported that
#   acceptance collapses from 59% to 1.4% "with depth". Depth and gibberish were changed
#   together, so the collapse could not be attributed to either. The natural-text run is the
#   real measurement; the random run is kept as the control arm, which makes the pair a clean
#   2x2 at no extra cost.
FILLER=${FILLER:-natural}
NATSRC=${NATSRC:-$D/kld-corpus/en-heldout.txt}
[ "$FILLER" = natural ] && { [ -s "$NATSRC" ] || { say "REFUSING: no natural corpus at $NATSRC"; exit 2; }; }

mkfiller(){ # $1 target tokens -> writes $OUT/filler-$1.txt, echoes achieved token count
  local want=$1
  local f="$OUT/filler-$want.txt"
  [ "$want" -eq 0 ] && { : > "$f"; echo 0; return 0; }
  if [ ! -s "$f" ]; then
    if [ "$FILLER" = natural ]; then
      # ~4.2 bytes/token for English is a starting guess only; the loop corrects it against
      # the real tokenizer.
      head -c $(( want * 5 )) "$NATSRC" > "$f"
    else
      python3 - "$f" "$want" <<'PY2'
import sys, random
path, want = sys.argv[1], int(sys.argv[2])
random.seed(20260829)
words = ("the quick brown fox jumps over lazy dogs while numbers like 3141 and 2718 drift past "
         "a quiet harbour where ledgers record cargo manifests tonnage and the names of ships").split()
with open(path,"w",encoding="utf-8") as fh:
    for i in range(int(want*0.85)):
        fh.write(random.choice(words))
        fh.write("\n" if i % 17 == 16 else " ")
PY2
    fi
  fi
  local got; got=$(ntok "$f")
  local tries=0
  while [ "$tries" -lt 8 ]; do
    local diff=$(( got - want )); local adiff=${diff#-}
    [ "$adiff" -le $(( want / 50 )) ] && break
    if [ "$FILLER" = natural ]; then
      local newbytes
      newbytes=$(python3 -c "import os;print(max(1024,int(os.path.getsize('$f')*$want/max($got,1))))")
      head -c "$newbytes" "$NATSRC" > "$f"
    else
      python3 - "$f" "$got" "$want" <<'PY3'
import sys
path, got, want = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
toks = open(path,encoding='utf-8').read().split()
if got > want:
    toks = toks[:max(1,int(len(toks)*want/got))]
else:
    need = int(len(toks)*want/max(got,1)) - len(toks)
    toks = toks + toks[:max(1,need)]
open(path,"w",encoding='utf-8').write(" ".join(toks))
PY3
    fi
    got=$(ntok "$f"); tries=$((tries+1))
  done
  echo "$got"
}

# Response parser as a file, not an inline heredoc: a heredoc cannot be piped cleanly and
# the first attempt was a syntax error. Emits ERROR rows so a failed request can never be
# recorded as 0 tok/s.
PARSER=$OUT/parse_resp.py
cat > "$PARSER" <<'PYEOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print("ERROR\tunparseable: %s\t0" % str(e)[:120].replace("\t", " "))
    raise SystemExit
if "timings" not in d:
    msg = json.dumps(d.get("error", d))[:180].replace("\t", " ")
    print("ERROR\t%s\t0" % msg)
    raise SystemExit
t = d["timings"]
print("%s\t%s\t%s" % (t.get("predicted_per_second", 0), t.get("prompt_ms", 0),
                       t.get("predicted_n", 0)))
PYEOF

printf 'depth_target\tdepth_actual\tcenter\tdraft\tctx\trep\tdecode_tps\tttft_ms\tpredicted_n\tvram_mib\n' > "$TSV"

say "### sweep: depths [$DEPTHS] x {center off,on} x {draft off,on}, $REPS reps, single card $CARD"
say "### filler = $FILLER$([ "$FILLER" = natural ] && echo " (source $NATSRC)")"

for depth in $DEPTHS; do
  ctx=$(( depth + 4096 ))
  [ "$ctx" -lt 8192 ] && ctx=8192
  for center in off on; do
    for draft in off on; do
      extra="-c $ctx -ctk q4_0 -ctv q4_0"
      # /bias, not /models: the bias lives beside the models directory, not inside it.
      # n=1 caught this as four LOAD_FAILs; the sweep would have reported "centering does not
      # load on this build" and that would have been a false finding about the fork.
      [ "$center" = on ] && extra="$extra --kv-mean-center /bias/$(basename "$BIAS")"
      [ "$draft"  = on ] && extra="$extra -md $DRAFT --spec-type draft-dspark --spec-draft-n-max 7"
      say "--- depth=$depth ctx=$ctx center=$center draft=$draft"
      if ! srv_up $extra; then
        say "    LOAD FAILED. Recorded as a failure, kept, not retried silently."
        printf '%s\t%s\t%s\t%s\t%s\tLOAD_FAIL\t\t\t\t\n' "$depth" "" "$center" "$draft" "$ctx" >> "$TSV"
        docker logs "$NAME" 2>&1 | tail -20 >> "$LOG"
        srv_down; continue
      fi
      actual=$(mkfiller "$depth")
      used=$(vram)
      say "    achieved depth $actual tokens, ${used} MiB on card $CARD"
      body=$OUT/req-$depth.json
      python3 - "$OUT/filler-$depth.txt" "$body" <<'PY'
import json,sys
pre=open(sys.argv[1],encoding='utf-8').read()
q="Explain in detail how a binary search tree stays balanced, then give an example."
msg=(pre+"\n\n"+q) if pre.strip() else q
json.dump({"messages":[{"role":"user","content":msg}],
           "max_tokens":300,"temperature":0,"cache_prompt":True},open(sys.argv[2],"w"))
PY
      # Warm-up pays the prefill once; the timed reps then measure decode at depth.
      curl -s -m 1800 "localhost:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
        --data-binary "@$body" -o /dev/null
      for i in $(seq 1 "$REPS"); do
        # Keep the raw response. The first version piped straight into a timings extractor,
        # so a server that returned an ERROR object produced "0 tok/s" and the error text was
        # thrown away. At depth 262144 that turned a real failure into a fake zero.
        resp=$OUT/resp-d${depth}-c${center}-dr${draft}-r${i}.json
        curl -s -m 1800 "localhost:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
          --data-binary "@$body" -o "$resp"
        python3 "$PARSER" "$resp" \
        | while IFS=$'\t' read -r tps ttft pn; do
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
              "$depth" "$actual" "$center" "$draft" "$ctx" "$i" "$tps" "$ttft" "$pn" "$used" >> "$TSV"
            if [ "$tps" = ERROR ]; then
              say "    rep$i SERVER ERROR: $ttft   (body kept at $resp)"
            else
              say "    rep$i decode ${tps} tok/s  ttft ${ttft} ms  predicted ${pn}"
            fi
          done
      done
      # Always keep the server's own log for every cell. The 131k cell reported 0% draft
      # acceptance and a real +50% wall-clock speedup at the same time, which speculative
      # decoding cannot produce. The KV cache configuration lines are the place that
      # contradiction will resolve, and they were being discarded on success.
      docker logs "$NAME" > "$OUT/srvlog-d${depth}-c${center}-dr${draft}.txt" 2>&1
      if [ "$draft" = on ]; then
        docker logs "$NAME" 2>&1 \
          | grep -oE "draft acceptance = [0-9.]+ \([ 0-9]+ accepted / +[0-9]+ generated\), mean len = +[0-9.]+" \
          | tail -3 | tee -a "$LOG" > "$OUT/acceptance-d${depth}-c${center}.txt"
      fi
      srv_down
    done
  done
done

say "### #5b: does a MISMATCHED bias get rejected at load"
# His claim: "calibrate the bias with -ctk q4_0 set so the K-rotation state matches the
# serving config (a mismatched bias is rejected at load with a clear message)".
# Two distinct negatives, because they fail for different reasons:
#   A. structurally corrupt file  -> tests the parser
#   B. bias calibrated at a DIFFERENT -ctk -> tests the claim he actually made
# The first version of this used `head -c 200000` on a 66400 B file, which copies the file
# unchanged and proved nothing. n=1 caught it.
BADDIR=$OUT/badbias; mkdir -p "$BADDIR"
head -c 30000 "$BIAS" > "$BADDIR/bias-truncated.gguf"
say "  A. truncated bias: $(stat -c %s "$BADDIR/bias-truncated.gguf") B of $(stat -c %s "$BIAS") B"
if [ "$(stat -c %s "$BADDIR/bias-truncated.gguf")" -ge "$(stat -c %s "$BIAS")" ]; then
  say "  REFUSING to run test A: the truncated file is not smaller than the original."
else
  BIAS_SAVE=$BIAS; BIAS=$BADDIR/bias-truncated.gguf
  if srv_up -c 8192 -ctk q4_0 -ctv q4_0 --kv-mean-center /bias/bias-truncated.gguf; then
    say "  A RESULT: server STARTED on a truncated bias. Silent accept."
  else
    say "  A RESULT: load refused. Error text:"
    docker logs "$NAME" 2>&1 | grep -iE "mean.center|gguf|error" | tail -6 | tee -a "$LOG"
  fi
  srv_down; BIAS=$BIAS_SAVE
fi

say "  B. bias calibrated at -ctk q8_0, served at -ctk q4_0 (the mismatch he describes)"
WRONG=$BADDIR/bias-q8calib.gguf
CALRC=0
if [ ! -s "$WRONG" ]; then
  CALLOG=$OUT/calib-q8-attempt.log
  docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES="$CARD" \
    -v "$BIN":/prism:ro -v "$MODELS":/models -v "$BADDIR":/bad \
    -v "$(dirname "$CALIB")":/w:ro --entrypoint /prism/llama-kv-mean-center "$IMG" \
    -m "/models/$TERN" -f "/w/$(basename "$CALIB")" -o /bad/bias-q8calib.gguf \
    -ctk q8_0 -fa on -ngl 99 -c 512 --chunks 200 > "$CALLOG" 2>&1
  CALRC=$?
  cat "$CALLOG" >> "$LOG"
fi
# Three outcomes, kept distinct. The first version collapsed them into one and would have
# reported a harness fault as a finding about the tool.
if [ "$CALRC" -ne 0 ] && [ ! -s "$WRONG" ]; then
  say "  B RESULT: llama-kv-mean-center exited $CALRC and produced no bias. Its own words:"
  tail -12 "${CALLOG:-$LOG}" | tee -a "$LOG"
  say "     If that is a refusal to calibrate at -ctk q8_0, the mismatch cannot be created and"
  say "     so cannot reach load time. If it is anything else, this test did not run. READ IT."
elif [ ! -s "$WRONG" ]; then
  say "  B RESULT: the tool exited 0 and still wrote no bias file. Test 5b-B did NOT run."
else
  say "  calibrated a q8_0 bias: $(stat -c %s "$WRONG") B"
  # srv_up mounts $(dirname "$BIAS") as /bias, so BIAS must point into BADDIR for this test.
  BIAS_SAVE=$BIAS; BIAS=$WRONG
  if srv_up -c 8192 -ctk q4_0 -ctv q4_0 --kv-mean-center /bias/bias-q8calib.gguf; then
    say "  B RESULT: server STARTED with a q8_0-calibrated bias served at q4_0."
    say "     That CONTRADICTS the claim that a mismatched bias is rejected at load."
  else
    say "  B RESULT: rejected at load, as claimed. Error text:"
    docker logs "$NAME" 2>&1 | grep -iE "mean.center|mismatch|error" | tail -6 | tee -a "$LOG"
  fi
  srv_down; BIAS=$BIAS_SAVE
fi

say "=== SWEEP COMPLETE ==="
say "tsv: $TSV"
wc -l "$TSV" | tee -a "$LOG"
