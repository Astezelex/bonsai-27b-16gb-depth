#!/usr/bin/env python3
"""Render the Part 3 figures from figdata-part3.json.

Usage: make_figures_part3.py <figdata-part3.json> <outdir>

Palette and chrome are inherited from Part 1's scripts/make_figures.py so the two parts read
as one document: Bonsai=blue, the drafter-assisted configuration=blue-300 (a slot Part 1
already reserved for exactly this), semantic red for a failure, neutral gray for a
non-series annotation. Aqua is sub-3:1 on this surface, so every mark carries a direct label.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BONSAI = "#2a78d6"
BONSAI_LT = "#6da7ec"
QWEN = "#1baf7a"
WRONG = "#e34948"
NEUTRAL = "#898781"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "DejaVu Sans",
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "svg.fonttype": "none",
})


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)


def save(fig, outdir, name):
    for ext in ("png", "svg"):
        fig.savefig(f"{outdir}/{name}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def f8_drafter_depth(d, outdir):
    """Draft acceptance and decode speed against context depth.

    Design notes, because two earlier drafts were worse.
    A symlog x axis spent half its width on the empty span between 0 and 8k and crushed the
    four interesting points together, so the axis is categorical: six measured depths, evenly
    spaced, each labelled with its real token count. That distorts distance, which is stated
    on the axis, and it buys legibility for every point.
    The drafter's 4,096-token training window was first drawn as a vertical line. On a
    categorical axis that line cannot sit truthfully between two slots, and the line also
    oversold the story: acceptance is still 49.5% at twice the window. It is replaced by a
    ratio row under the axis, which turns the mechanism into a number at every point instead
    of a claim about one position.
    """
    rows = [r for r in d["depth_sweep"] if r["nodraft_tps"]]
    n = len(rows)
    xs = list(range(n))
    ntrain = d["draft_n_ctx_train"]
    depth = [r["depth_actual"] or 0 for r in rows]
    acc = [(r["acceptance"] or {}).get("acceptance") for r in rows]
    mlen = [(r["acceptance"] or {}).get("mean_len") for r in rows]
    nd = [r["nodraft_tps"] for r in rows]
    wd = [r["draft_tps"] for r in rows]

    def human(t):
        return "0" if t == 0 else (f"{t/1000:.0f}k" if t < 1000000 else f"{t/1000:.0f}k")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 6.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.2], "hspace": 0.16})
    for ax in (ax1, ax2):
        style_ax(ax)
        ax.set_xlim(-0.45, n - 0.55)

    # ---- top: acceptance
    ok = [(i, a) for i, a in zip(xs, acc) if a is not None]
    ax1.plot([i for i, _ in ok], [a * 100 for _, a in ok], color=BONSAI_LT,
             linewidth=2, marker="o", markersize=7, zorder=3)
    for i, a, m in zip(xs, acc, mlen):
        if a is None:
            continue
        lab = f"{a*100:.1f}%" if a >= 0.001 else f"{a*100:.2f}%"
        ax1.annotate(lab, (i, a * 100), textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=9.5, color=INK, fontweight="bold")
        ax1.annotate(f"mean run {m:.2f}", (i, a * 100), textcoords="offset points",
                     xytext=(0, 24), ha="center", fontsize=7.5, color=MUTED)
    ax1.set_ylabel("draft acceptance", fontsize=9)
    ax1.set_ylim(-6, 88)
    ax1.set_yticks([0, 20, 40, 60], ["0%", "20%", "40%", "60%"], fontsize=9)
    ax1.set_title("Draft acceptance falls away between 2x and 8x the training window",
                  loc="left", fontsize=12, fontweight="bold", pad=12)

    # ---- bottom: decode speed
    ax2.plot(xs, nd, color=BONSAI, linewidth=2, marker="o", markersize=7,
             label="no drafter", zorder=3)
    xw = [i for i, w in zip(xs, wd) if w]
    yw = [w for w in wd if w]
    ax2.plot(xw, yw, color=BONSAI_LT, linewidth=2, marker="s", markersize=7,
             label="with slim DSpark drafter", zorder=3)
    for i, a, b, r in zip(xs, nd, wd, rows):
        if not b:
            continue
        up = (b / a - 1) * 100
        anomalous = (r["acceptance"] or {}).get("acceptance") == 0.0
        ax2.annotate(f"{up:+.0f}%", (i, b), textcoords="offset points",
                     xytext=(0, 13 if not anomalous else 30), ha="center", fontsize=9.5,
                     color=WRONG if anomalous else INK, fontweight="bold")
        if anomalous:
            ax2.plot([i], [b], marker="s", markersize=14, markerfacecolor="none",
                     markeredgecolor=WRONG, markeredgewidth=1.8, zorder=4)
            ax2.annotate("0% acceptance yet a real\nspeedup: unresolved, ch. 8",
                         (i, b), textcoords="offset points", xytext=(0, 44),
                         ha="center", fontsize=7.5, color=WRONG)
    for i, r in zip(xs, rows):
        if r.get("draft_load_failed"):
            ax2.plot([i], [r["nodraft_tps"]], marker="X", markersize=12, color=WRONG, zorder=4)
            ax2.annotate("drafter will not load\non a 16 GB card", (i, r["nodraft_tps"]),
                         textcoords="offset points", xytext=(0, 16), ha="center",
                         fontsize=7.5, color=WRONG)
    ax2.set_ylabel("decode, tokens/s", fontsize=9)
    ax2.set_ylim(0, 108)
    ax2.legend(loc="upper right", frameon=False, fontsize=9)

    # ---- categorical axis with a ratio row underneath
    ax2.set_xticks(xs, [human(t) for t in depth], fontsize=9.5)
    for i, t in zip(xs, depth):
        ratio = "0" if t == 0 else (f"{t/ntrain:.0f}x" if t / ntrain >= 10
                                    else f"{t/ntrain:.1f}x")
        ax2.annotate(ratio, (i, 0), xycoords=("data", "axes fraction"),
                     textcoords="offset points", xytext=(0, -34), ha="center",
                     fontsize=8.5, color=BONSAI_LT, fontweight="bold")
    ax2.annotate("multiples of the drafter's 4,096-token training window",
                 (0, 0), xycoords=("axes fraction", "axes fraction"),
                 textcoords="offset points", xytext=(0, -50), ha="left",
                 fontsize=8, color=INK2)
    ax2.set_xlabel("context depth when generation starts, evenly spaced, not to scale",
                   fontsize=9, labelpad=42)

    fig.text(0.005, -0.10,
             "Bonsai 27B Q2_g64, prism b10658, one RTX 5060 Ti 16 GB, -fa on -np 1, temp 0, "
             "300 tokens, median of 3 repetitions.\nPrefix is held-out Wikipedia text and "
             "depth is verified through /tokenize. Prefix content matters as much as its "
             "length: see chapter 3.",
             fontsize=7.5, color=MUTED)
    save(fig, outdir, "f8-drafter-acceptance-vs-depth")
    print("wrote f8-drafter-acceptance-vs-depth")


def f9_drafter_vram(d, outdir):
    """Resident VRAM against allocated context, with and without the drafter.

    A direct sequel to Part 1's f4-vram-vs-context, and drawn on the same axes on purpose:
    log-2 context, resident GiB, the card limit as a red dashed line. Part 1's figure said the
    full 262k context fits one 16 GB card with room to spare. This one adds the series that
    spends that room.

    The x axis is ALLOCATED context, the server's -c, and not the depth used in F8. VRAM is a
    function of what the server reserves, and the drafter reserves its own KV cache on top of
    the main model's.
    """
    rows = [r for r in d["depth_sweep"] if r["nodraft_vram"]]
    fileb = d["drafter_file_bytes"] / 2**20        # MiB
    # The runtime reports 15,888 MiB usable at init; NVML reports 16,311 MiB physical.
    # The usable figure is the one a load either fits inside or does not.
    card = d["card_vram_mib_usable"] / 1024       # GiB

    def ctx_of(r):
        return max(8192, r["depth_target"] + 4096)

    xs = [ctx_of(r) / 1024 for r in rows]                       # k tokens
    nd = [r["nodraft_vram"] / 1024 for r in rows]               # GiB
    wd = [(r["draft_vram"] / 1024 if r["draft_vram"] else None) for r in rows]

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.axhline(card, color=WRONG, linestyle="--", linewidth=1.2, zorder=1)
    ax.text(xs[0] * 1.02, card + 0.12,
            f"usable VRAM, {card:.1f} GiB  (16,311 MiB physical)",
            fontsize=8.5, color=WRONG)

    ax.plot(xs, nd, color=BONSAI, linewidth=2, marker="o", markersize=7, zorder=3)
    xw = [x for x, w in zip(xs, wd) if w]
    yw = [w for w in wd if w]
    ax.plot(xw, yw, color=BONSAI_LT, linewidth=2, marker="s", markersize=7, zorder=3)
    for x, y in zip(xs, nd):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, -15),
                    ha="center", fontsize=8.5, color=BONSAI)
    for x, y in zip(xw, yw):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8.5, color=BONSAI_LT)
    # Series labels go at the LEFT end. At the right end they collide with their own value
    # labels and with the load-failure marker; at the left the two series are 2.7 GiB apart.
    ax.annotate("no drafter", (xs[0], nd[0]), textcoords="offset points",
                xytext=(-10, -4), ha="right", fontsize=9.5, color=BONSAI, fontweight="bold")
    ax.annotate("with slim\ndrafter", (xs[0], wd[0]), textcoords="offset points",
                xytext=(-10, -6), ha="right", fontsize=9.5, color=BONSAI_LT,
                fontweight="bold")

    # the load failure, drawn where it was attempted
    fail = [r for r in d["depth_sweep"] if r.get("draft_load_failed")]
    for r in fail:
        x = ctx_of(r) / 1024
        ax.plot([x], [card], marker="X", markersize=13, color=WRONG, zorder=5)
        ax.annotate("drafted server\nfails to load", (x, card),
                    textcoords="offset points", xytext=(-10, 14), ha="right",
                    fontsize=8.5, color=WRONG, fontweight="bold")

    # the point of the chapter, stated on the figure
    i = max(range(len(xw)), key=lambda j: xw[j])
    gap = (yw[i] - nd[xs.index(xw[i])]) * 1024
    # An explicit two-headed arrow across the gap, because a plain leader line ending in
    # empty space did not read as "this distance is the point".
    xg = xw[i]
    lo, hi = nd[xs.index(xg)], yw[i]
    ax.annotate("", xy=(xg, hi), xytext=(xg, lo),
                arrowprops=dict(arrowstyle="<->", color=INK2, linewidth=1.3))
    ax.annotate(f"the drafter costs {gap:,.0f} MiB here,\n"
                f"{gap/fileb:.1f}x its own {fileb:.0f} MiB file",
                (xg, (hi + lo) / 2), textcoords="offset points", xytext=(-12, 0),
                ha="right", va="center", fontsize=9, color=INK)

    ax.set_xscale("log", base=2)
    ticks = sorted({round(x) for x in xs})
    ax.set_xticks(ticks, [f"{t}k" for t in ticks], fontsize=9)
    ax.set_xlim(min(xs) * 0.75, max(xs) * 1.9)
    ax.set_ylim(6.5, card + 1.5)
    ax.set_xlabel("allocated context, the server's -c (tokens)", fontsize=9)
    ax.set_ylabel("resident VRAM (GiB)", fontsize=9)
    ax.set_title("The VRAM the slim drafter returns is re-spent on its own KV cache",
                 loc="left", fontsize=12.5, fontweight="bold", pad=20)
    ax.text(0, 1.035, "Measured resident VRAM (NVML, includes runtime), single slot, "
                      "-ctk q4_0 -ctv q4_0 -fa on. Compare Part 1 figure 4.",
            transform=ax.transAxes, fontsize=8.5, color=INK2)
    style_ax(ax)
    save(fig, outdir, "f9-drafter-vram-vs-context")
    print("wrote f9-drafter-vram-vs-context")


def f10_centering(d, outdir):
    """K-cache mean-centering: absolute damage against context, and the benefit by config.

    Two panels because there are two findings and one panel would hide the larger of them.
    Panel A is a single comparable series (Wikipedia held-out, q4_0 V) so the lines mean what
    they look like they mean. Panel B is a configuration comparison, not a series, so it is
    drawn as bars and each bar names its own configuration.

    Layout notes, both earned by looking at a bad render. Panel B's configuration names were
    first drawn as y-tick labels and were wider than the gap between the panels, so they spilled
    across panel A; they now sit inside panel B, to the right of the zero line, where every bar
    leaves the space empty. Panel A first labelled every point on both series, which collides
    where the two lines converge to within 0.00002; only the uncentered series is labelled now,
    because "the damage grows" is what that panel is for.
    """
    lad = d["kld_ladder"]
    series = sorted([r for r in lad if r["corpus"] == "wiki" and r["v_cache"] == "q4_0"],
                    key=lambda r: r["n_ctx"])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.2, 4.3),
                                   gridspec_kw={"width_ratios": [1, 1.35], "wspace": 0.22})

    # ---- A: absolute divergence against context
    style_ax(axA)
    xs = [r["n_ctx"] for r in series]
    unc = [r["uncentered"] for r in series]
    cen = [r["centered"] for r in series]
    axA.plot(xs, unc, color=BONSAI, linewidth=2, marker="o", markersize=7, zorder=3)
    axA.plot(xs, cen, color=BONSAI_LT, linewidth=2, marker="s", markersize=7, zorder=3)
    for x, y in zip(xs, unc):
        axA.annotate(f"{y:.5f}", (x, y), textcoords="offset points", xytext=(0, 11),
                     ha="center", fontsize=8.5, color=BONSAI)
    # Series labels to the RIGHT of the final markers. Placed inside the plot they landed on
    # the 8k value label, because the two series are 0.00002 apart there.
    axA.annotate("uncentered", (xs[-1], unc[-1]), textcoords="offset points",
                 xytext=(9, 3), ha="left", fontsize=9.5, color=BONSAI, fontweight="bold")
    axA.annotate("centered", (xs[-1], cen[-1]), textcoords="offset points",
                 xytext=(9, -11), ha="left", fontsize=9.5, color=BONSAI_LT,
                 fontweight="bold")
    axA.annotate(f"{(unc[-1]-unc[0])/unc[0]*100:+.0f}% divergence\nfrom 512 to 16k",
                 (xs[0], unc[0]), textcoords="offset points", xytext=(8, -34),
                 ha="left", fontsize=8.5, color=INK2)
    axA.set_xscale("log", base=2)
    axA.set_xticks(xs, [f"{x//1024}k" if x >= 1024 else str(x) for x in xs], fontsize=9)
    axA.set_xlim(xs[0] * 0.6, xs[-1] * 3.4)
    axA.set_ylim(0, max(unc) * 1.32)
    axA.set_xlabel("n_ctx (tokens)", fontsize=9)
    axA.set_ylabel("mean KL divergence against an F16 reference", fontsize=8.5)
    axA.set_title("The damage grows, the fix does not", loc="left",
                  fontsize=10.5, fontweight="bold")

    # ---- B: benefit by configuration
    style_ax(axB)
    order = [("512 tokens, f16 V, original corpus   (the published configuration)",
              "orig", "f16", 512),
             ("512 tokens, q4_0 V, original corpus", "orig", "q4_0", 512),
             ("512 tokens, q4_0 V, Wikipedia", "wiki", "q4_0", 512),
             ("8,192 tokens, q4_0 V, Wikipedia", "wiki", "q4_0", 8192),
             ("16,384 tokens, q4_0 V, Wikipedia", "wiki", "q4_0", 16384)]
    labs, vals = [], []
    for lab, corpus, v, ctx in order:
        m = [r for r in lad if r["corpus"] == corpus and r["v_cache"] == v
             and r["n_ctx"] == ctx]
        if m:
            labs.append(lab); vals.append(m[0]["delta_pct"])
    ys = list(range(len(vals), 0, -1))
    for y, v, lab in zip(ys, vals, labs):
        axB.barh(y, v, height=0.5, color=BONSAI if v < 0 else WRONG,
                 edgecolor=SURFACE, linewidth=1.4, zorder=3)
        # Every value label sits left of the zero line and every configuration label right
        # of it. Putting the positive bar's value on its own side collided with its name.
        axB.text(v - 0.7 if v < 0 else -0.8, y, f"{v:+.1f}%",
                 va="center", ha="right", fontsize=9.5, color=INK, fontweight="bold")
        axB.text(1.6, y, lab, va="center", ha="left", fontsize=8.5, color=INK2)
    axB.axvline(0, color=BASE, linewidth=1.1, zorder=2)
    axB.set_yticks([])
    axB.set_ylim(0.4, len(vals) + 0.7)
    axB.set_xlim(min(vals) * 1.22, 46)
    axB.set_xticks([-20, -10, 0], ["-20%", "-10%", "0"], fontsize=9)
    axB.set_xlabel("change in mean KLD from centering (negative is better)", fontsize=8.5)
    axB.set_title("What the published number becomes in a servable setting",
                  loc="left", fontsize=10.5, fontweight="bold")
    axB.grid(visible=False)

    fig.suptitle("K-cache mean-centering is a short-context result, measured on a cache "
                 "pairing nobody can serve",
                 x=0.006, ha="left", fontsize=12.5, fontweight="bold", y=1.05)
    fig.text(0.006, -0.05,
             "llama-perplexity --kl-divergence against an F16 KV reference, Bonsai 27B Q2_g64, "
             "prism b10658, bias calibrated with -ctk q4_0 as the loader requires.\n"
             "The published row reproduces exactly. -ctk q4_0 with f16 V is an asymmetric cache "
             "pair: on this hardware it runs prefill at 4.53 tok/s at depth 8,192, against "
             "907.71 for the symmetric pair.",
             fontsize=7.5, color=MUTED)
    save(fig, outdir, "f10-centering-benefit")
    print("wrote f10-centering-benefit")


def f11_measurable_range(d, outdir):
    """How far up the context axis each measurement can actually be taken.

    A figure of llama-perplexity's host-RAM reserve against n_ctx was considered and dropped:
    it is one exponential and the seven-row table in the chapter says it better. This says the
    thing the table cannot, which is the paper's structural argument in one image. Of the four
    quantities a 16 GB owner would want at 262k, three are measurable and the fourth, the one
    the feature exists to improve, is not.
    """
    target = 262144
    bars = [
        ("resident VRAM", "NVML with the server up", 266240, True,
         "reaches the target"),
        ("decode speed", "llama-server timings", 266240, True,
         "reaches the target"),
        ("draft acceptance", "llama-server draft counters", 131050, False,
         "drafted server will not\nload on a 16 GB card"),
        ("quality, KL divergence", "llama-perplexity", 16384, False,
         "llama-perplexity reserves\n159 GB of host RAM at 262k"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.1))
    style_ax(ax)
    lo = 512
    ys = list(range(len(bars), 0, -1))
    for y, (name, instr, reach, ok, why) in zip(ys, bars):
        col = BONSAI if ok else WRONG
        ax.plot([lo, reach], [y, y], color=col, linewidth=9, solid_capstyle="butt", zorder=3)
        ax.plot([reach], [y], marker="o" if ok else "X", markersize=11 if ok else 13,
                color=col, zorder=4)
        ax.text(lo * 1.15, y + 0.52, name, fontsize=10, color=INK, fontweight="bold",
                va="bottom")
        # MUTED, not SURFACE. The first render drew these in the background colour and they
        # were invisible; the figure looked finished and was missing a whole column.
        # Both labels ABOVE the bar. Below it, the instrument name sat almost equidistant
        # between its own bar and the next row's title and could be read as either.
        ax.text(lo * 1.15, y + 0.20, instr, fontsize=8, color=MUTED, va="bottom")
        # Every reason sits in one column to the RIGHT of the target line. Anchoring them to
        # the bar ends put two of them underneath the dashed line.
        ax.annotate(why, (target * 1.5, y), ha="left", va="center", fontsize=8.5,
                    color=INK2 if ok else WRONG)
    # vlines, not axvline: a full-height line ran straight through its own caption.
    ax.vlines(target, 0.5, len(bars) + 0.62, color=INK2, linestyle=(0, (4, 3)),
              linewidth=1.3, zorder=2)
    ax.annotate("262,144 tokens:\nthe context these\nfeatures are sold for",
                (target, len(bars) + 0.62), ha="center", va="bottom",
                fontsize=9, color=INK, fontweight="bold")
    ax.set_xscale("log", base=2)
    ticks = [512, 4096, 16384, 65536, 262144]
    ax.set_xticks(ticks, ["512", "4k", "16k", "64k", "262k"], fontsize=9.5)
    ax.set_xlim(lo * 0.8, 9000000)
    ax.set_ylim(0.4, len(bars) + 1.35)
    ax.set_yticks([])
    ax.set_xlabel("context at which the measurement can be taken (tokens)", fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_title("Three of the four things worth knowing at 262k can be measured. "
                 "The fourth cannot.",
                 loc="left", fontsize=12.5, fontweight="bold", pad=26)
    fig.text(0.006, -0.06,
             "Limits measured on this box: one RTX 5060 Ti (16,311 MiB physical, 15,888 usable), 31 GB host RAM.\n"
             "The perplexity ceiling belongs to the tool and the vocabulary, not to the "
             "card: n_ctx x n_vocab x 4 bytes is 159.3 GB at 262,144\n"
             "for a 151,936-token vocabulary. The 32,768 rung was killed by the kernel at "
             "25.5 GB resident.",
             fontsize=7.5, color=MUTED)
    save(fig, outdir, "f11-what-can-be-measured")
    print("wrote f11-what-can-be-measured")


def main():
    d = json.load(open(sys.argv[1]))
    outdir = sys.argv[2]
    f8_drafter_depth(d, outdir)
    f9_drafter_vram(d, outdir)
    f10_centering(d, outdir)
    f11_measurable_range(d, outdir)


if __name__ == "__main__":
    main()
