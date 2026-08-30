#!/usr/bin/env python3
"""Temp-0 byte-identity check for speculative decoding.

The claim under test (PrismML, 2026-08): with the slim dflash drafter, output stays
byte-identical to no-draft decoding at temperature 0 "by construction". Part 1 of our
benchmark verified this 5/5 for the older Q4_1 drafter. The slim Q4_0 drafter has not
been checked by anyone outside the vendor.

Two modes:
  identity.py <port> <out.json>              capture outputs from a running server
  identity.py --compare <a.json> <b.json>    compare two captures byte for byte

Deliberately strict: compares the exact string, not a normalised form. A speculative
decoder that is correct must reproduce the greedy path exactly.
"""
import hashlib
import json
import os
import sys
import urllib.request

PROMPTS = [
    "Explain how a B-tree keeps its height balanced during insertion.",
    "Write a Python function that merges overlapping intervals. Include comments.",
    "List the steps of the TCP three-way handshake and what each step establishes.",
    "A train leaves at 14:05 and arrives at 17:40. How long is the journey? Show working.",
    "Write a bash function that rotates a log file only if it exceeds a size threshold.",
]
MAXTOK = int(os.environ.get("MAXTOK", "200"))


def capture(port, out_path):
    results = []
    for i, p in enumerate(PROMPTS, 1):
        body = json.dumps({
            "messages": [{"role": "user", "content": p}],
            "max_tokens": MAXTOK,
            "temperature": 0,
            "top_k": 1,
            "seed": 1234,
            # Bonsai is a thinking model. A first version of this script captured only
            # `content` with a 200-token budget: every generation spent its whole budget
            # in `reasoning_content`, `content` came back empty, and the comparison
            # reported "5/5 identical" because it was hashing the empty string against
            # itself. Same class as the 4000-token Cold-Fusion cap. Thinking is disabled
            # here so the answer fits the budget, and BOTH channels are captured anyway.
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     data=body, headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=600).read())
        msg = r["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        finish = r["choices"][0].get("finish_reason")
        combined = reasoning + "\x00" + content      # compare the WHOLE emitted stream
        tim = r.get("timings", {}) or {}
        results.append({
            "prompt": p,
            "text": content,
            "reasoning": reasoning,
            "finish_reason": finish,
            "sha256": hashlib.sha256(combined.encode()).hexdigest(),
            "n_predicted": tim.get("predicted_n"),
            "tok_per_s": round(tim.get("predicted_per_second", 0), 2),
        })
        print(f"  prompt {i}/{len(PROMPTS)}: content={len(content)} reasoning={len(reasoning)} "
              f"chars, {tim.get('predicted_n')} tok, finish={finish}, "
              f"{tim.get('predicted_per_second', 0):.2f} tok/s", flush=True)
        if not combined.strip("\x00"):
            print("      REFUSING: generation produced no text in either channel. "
                  "A hash of nothing is not evidence.", flush=True)
            return 2
    json.dump(results, open(out_path, "w"), indent=1)
    print(f"captured {len(results)} -> {out_path}")
    return 0


def compare(a_path, b_path):
    a = json.load(open(a_path))
    b = json.load(open(b_path))
    if len(a) != len(b):
        print(f"MISMATCH: different prompt counts {len(a)} vs {len(b)}")
        return 1
    # Guard against the failure that produced a false "5/5 identical" on the first run:
    # two empty captures hash the same. An empty comparison is not a pass.
    EMPTY = hashlib.sha256(b"\x00").hexdigest()
    empties = sum(1 for x in a + b if x["sha256"] == EMPTY or not (x.get("text") or x.get("reasoning")))
    if empties:
        print(f"INVALID: {empties} capture(s) contain no generated text. "
              f"Comparing empty against empty is not evidence of identity.")
        return 2
    capped = [x for x in a + b if x.get("finish_reason") == "length"]
    if capped:
        print(f"WARNING: {len(capped)} generation(s) hit the token budget "
              f"(finish_reason=length). Identity across truncated outputs proves less.")

    identical = 0
    for i, (x, y) in enumerate(zip(a, b), 1):
        same = x["sha256"] == y["sha256"]
        identical += same
        mark = "IDENTICAL" if same else "DIFFERS"
        print(f"  {i}. {mark}  nodraft={x['sha256'][:12]} dspark={y['sha256'][:12]}  "
              f"{x['tok_per_s']} -> {y['tok_per_s']} tok/s")
        if not same:
            # show the first divergence so the failure is diagnosable, not just flagged
            for j, (ca, cb) in enumerate(zip(x["text"], y["text"])):
                if ca != cb:
                    print(f"      first divergence at char {j}")
                    print(f"        nodraft: ...{x['text'][max(0,j-40):j+40]!r}")
                    print(f"        dspark : ...{y['text'][max(0,j-40):j+40]!r}")
                    break
            else:
                print(f"      one is a prefix of the other: "
                      f"{len(x['text'])} vs {len(y['text'])} chars")
    print(f"\nRESULT: {identical}/{len(a)} byte-identical at temperature 0")
    return 0 if identical == len(a) else 1


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    sys.exit(capture(sys.argv[1], sys.argv[2]))
