#!/usr/bin/env python3
"""Build a calibration + held-out prose corpus from a Wikipedia pages-articles dump.

Why this exists: the kv-mean-center KLD numbers in the changelog were measured at
n_ctx=512 on an 88 KB model-generated corpus. Pasha asked for the comparison "at your
262k target". A ladder up to 65k-token chunks needs roughly 3 MB of held-out text, which
is two orders of magnitude more than we have, and generating it from the model would cost
GPU hours we do not have while the anchors run.

Wikipedia is CC-BY-SA, it is real prose instead of model output, and both dumps are already
on /mnt/archive with verified md5s.

Design notes:
  - Streams the bz2 and stops as soon as the byte targets are met. It never decompresses
    25 GB.
  - calib and heldout are DISJOINT ARTICLE SETS, alternating, so neither split can contain
    a paraphrase of the other's content from the same page.
  - Refuses to write a corpus that is over-compressible, the same degeneracy guard
    mkcorpus.py uses, because a corpus of navigation boilerplate would silently flatter
    every perplexity number computed on it.

stdlib only: the container that runs the llama.cpp tools has no jq and no pip.
"""
import bz2
import gzip
import html
import os
import re
import sys

RE_PAGE_END = re.compile(rb"</page>")
RE_TEXT = re.compile(r"<text\b[^>]*>(.*?)</text>", re.S)
RE_TITLE = re.compile(r"<title>(.*?)</title>", re.S)
RE_NS = re.compile(r"<ns>(\d+)</ns>")

RE_COMMENT = re.compile(r"<!--.*?-->", re.S)
RE_REF = re.compile(r"<ref\b[^>]*?/>|<ref\b.*?</ref>", re.S | re.I)
RE_TAG = re.compile(r"<[^>]+>")
RE_TABLE = re.compile(r"\{\|.*?\|\}", re.S)
RE_DROP_PREFIX = re.compile(r"^\s*(?:File|Image|Category|Plik|Grafika|Kategoria)\s*:", re.I)
RE_EXTLINK = re.compile(r"\[https?://\S+\s([^\]]*)\]")
RE_BOLDIT = re.compile(r"'{2,5}")
RE_HEADING = re.compile(r"^=+\s*(.*?)\s*=+\s*$", re.M)
RE_WS = re.compile(r"[ \t]+")
RE_BLANK = re.compile(r"\n{3,}")

# Sections that are lists of links, not prose.
STOP_SECTIONS = ("See also", "References", "External links", "Further reading",
                 "Bibliography", "Notes", "Zobacz też", "Przypisy", "Bibliografia",
                 "Linki zewnętrzne")


def strip_templates(text):
    """Remove {{...}} including nested ones. Regex cannot nest, so scan."""
    out = []
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("{{", i):
            depth += 1
            i += 2
        elif text.startswith("}}", i):
            if depth:
                depth -= 1
            i += 2
        else:
            if not depth:
                out.append(text[i])
            i += 1
    return "".join(out)


def strip_links(text):
    r"""Resolve [[...]] with nesting.

    A regex cannot do this. `[[File:x.jpg|thumb|a [[log]] burning]]` has a nested link in
    its caption, so `[^\]]*` stops at the inner `]` and leaves a dangling `]]` in the
    output. That leak was visible in the first smoke run of this script: a caption
    fragment ending in `]]` had survived into the corpus.

    File/Image/Category groups are dropped whole, including their captions. Everything
    else keeps its display text, which is the part after the last pipe.
    """
    out = []
    stack = []          # start index in `out` for each open group
    i, n = 0, len(text)
    while i < n:
        if text.startswith("[[", i):
            stack.append(len(out))
            out.append("")
            i += 2
        elif text.startswith("]]", i) and stack:
            start = stack.pop()
            inner = "".join(out[start:])
            del out[start:]
            if RE_DROP_PREFIX.match(inner):
                pass                      # a media or category link, drop it entirely
            else:
                out.append(inner.split("|")[-1] if "|" in inner else inner)
            i += 2
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def clean(wikitext):
    t = wikitext
    t = RE_COMMENT.sub(" ", t)
    t = RE_REF.sub(" ", t)
    t = strip_templates(t)
    for _ in range(3):
        new = RE_TABLE.sub(" ", t)
        if new == t:
            break
        t = new
    t = strip_links(t)
    t = RE_EXTLINK.sub(r"\1", t)
    t = RE_TAG.sub(" ", t)
    t = RE_BOLDIT.sub("", t)

    # Cut everything from the first appendix-style heading onwards.
    for sec in STOP_SECTIONS:
        m = re.search(r"^=+\s*" + re.escape(sec) + r"\s*=+\s*$", t, re.M | re.I)
        if m:
            t = t[:m.start()]
    t = RE_HEADING.sub(r"\1", t)
    t = html.unescape(t)

    keep = []
    for line in t.split("\n"):
        s = line.strip()
        if not s:
            keep.append("")
            continue
        # Drop residual markup lines and list scaffolding.
        if s[0] in "|!*#:;{}=" or s.startswith("http"):
            continue
        if "[[" in s or "]]" in s or "{{" in s or "}}" in s:
            continue
        if len(s) < 40:
            continue
        keep.append(RE_WS.sub(" ", s))
    t = "\n".join(keep)
    t = RE_BLANK.sub("\n\n", t)
    return t.strip()


def pages(path):
    """Yield decoded <page> blocks, streaming."""
    dec = bz2.BZ2Decompressor()
    buf = b""
    with open(path, "rb") as fh:
        while True:
            raw = fh.read(1 << 20)
            if not raw:
                return
            try:
                buf += dec.decompress(raw)
            except EOFError:
                return
            while True:
                m = RE_PAGE_END.search(buf)
                if not m:
                    break
                block, buf = buf[:m.end()], buf[m.end():]
                yield block.decode("utf-8", "replace")


def main():
    if len(sys.argv) < 5:
        print("usage: wiki_corpus.py <dump.bz2> <out_calib> <out_heldout> "
              "<calib_MB> <heldout_MB> [skip_articles]")
        return 2
    dump, out_cal, out_eval = sys.argv[1], sys.argv[2], sys.argv[3]
    want_cal = int(float(sys.argv[4]) * 1_000_000)
    want_eval = int(float(sys.argv[5]) * 1_000_000)
    # The first articles in the dump are the flagship pages (Anarchism, Aristotle, ...):
    # the longest and most heavily edited text on the site, and not representative of the
    # prose a model actually serves. Skip past them so the corpus is ordinary articles.
    skip = int(sys.argv[6]) if len(sys.argv) > 6 else 3000

    cal, ev = [], []
    n_cal = n_ev = 0
    kept = seen = 0
    for block in pages(dump):
        seen += 1
        if "<redirect" in block:
            continue
        mns = RE_NS.search(block)
        if not mns or mns.group(1) != "0":
            continue
        mt = RE_TEXT.search(block)
        if not mt:
            continue
        title = RE_TITLE.search(block)
        title = title.group(1) if title else ""
        if "(disambiguation)" in title or title.endswith("(ujednoznacznienie)"):
            continue
        body = clean(html.unescape(mt.group(1)))
        if len(body) < 1500:
            continue
        kept += 1
        if kept <= skip:
            continue
        chunk = body + "\n\n"
        # Alternate whole articles between the two splits: disjoint by construction.
        if n_cal < want_cal and (kept % 3 == 1 or n_ev >= want_eval):
            cal.append(chunk)
            n_cal += len(chunk)
        elif n_ev < want_eval:
            ev.append(chunk)
            n_ev += len(chunk)
        if n_cal >= want_cal and n_ev >= want_eval:
            break

    if n_cal < want_cal * 0.9 or n_ev < want_eval * 0.9:
        print(f"FAIL: dump exhausted early. calib={n_cal} heldout={n_ev} "
              f"(wanted {want_cal}/{want_eval})", file=sys.stderr)
        return 1

    for path, parts in ((out_cal, cal), (out_eval, ev)):
        data = "".join(parts)
        raw = data.encode()
        ratio = len(gzip.compress(raw)) / max(len(raw), 1)
        if ratio < 0.25:
            print(f"FAIL: {path} is over-compressible (gzip ratio {ratio:.3f}), "
                  f"that means boilerplate, not prose", file=sys.stderr)
            return 1
        open(path, "w").write(data)
        print(f"{path}: {os.path.getsize(path)} B, gzip ratio {ratio:.3f}")

    print(f"pages scanned {seen}, qualifying articles {kept} (first {skip} skipped), "
          f"splits are disjoint article sets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
