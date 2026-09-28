#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR11 — Hình cho bản thảo, chỉ đọc JSON đã khóa. Ghi vào Causal_VILRR/paper/figures/.

Main:  fig1_mechanism.png, fig2_phase_diagram.png, fig3_learned_reversal.png, fig4_boundary.png
Supp:  figS1_phase_diagram.png, figS2_finetune.png, figS4_icdl.png (figS3 from vr41)
Bảng màu: khe categorical đã kiểm (dataviz validate_palette); mọi chuỗi có hình dấu và chú giải riêng.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

import vr_common as R

OUT = os.path.join(R.BASE, "paper", "figures")
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
CL = {"color_variegation": "Color variegation", "size": "Size", "lesion_skin_contrast": "Lesion-skin contrast",
      "asymmetry": "Asymmetry", "border_irregularity": "Border irregularity"}
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})


def J(name, base=R.RES):
    return json.load(open(os.path.join(base, name)))


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=300); plt.close(fig); R.log(f"wrote paper/figures/{name}")


def regions(ax, lim=6):
    """Tô các vùng của Theorem 1 trên mặt phẳng (θ, δ)."""
    L = lim
    ax.add_patch(Polygon([[0, 0], [L, L], [0, L]], closed=True, color=ORANGE, alpha=0.12, lw=0))         # θ>0, δ>θ
    ax.add_patch(Polygon([[0, 0], [L, 0], [L, L]], closed=True, color=BLUE, alpha=0.10, lw=0))           # 0<δ<θ
    ax.add_patch(Polygon([[0, 0], [L, 0], [L, -L], [0, -L]], closed=True, color=AQUA, alpha=0.10, lw=0))  # δ<0
    ax.add_patch(Polygon([[0, 0], [-L, -L], [0, -L]], closed=True, color=ORANGE, alpha=0.12, lw=0))       # θ<0, δ<θ
    ax.add_patch(Polygon([[0, 0], [-L, 0], [-L, -L]], closed=True, color=BLUE, alpha=0.10, lw=0))         # θ<δ<0
    ax.add_patch(Polygon([[0, 0], [-L, 0], [-L, L], [0, L]], closed=True, color=AQUA, alpha=0.10, lw=0))  # δ>0
    ax.plot([-L, L], [-L, L], color=INK, lw=1.1); ax.axhline(0, color=INK2, lw=0.8); ax.axvline(0, color=INK2, lw=0.8)


def fig1():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 3.4), gridspec_kw={"width_ratios": [1.3, 1]})
    a.set_axis_off(); a.set_xlim(0, 11); a.set_ylim(-0.6, 7.6)
    W_, H_ = 2.6, 0.8
    nodes = {"U": (1.4, 6.6, "Patient $U$", True), "G": (5.5, 6.6, "Site $G$", False),
             "D": (1.4, 4.4, "Disease $D$", True), "X": (5.5, 4.4, "Appearance $X$", False),
             "C": (9.6, 4.4, "Concept $C$", False), "W": (1.4, 2.2, "Clinical $W$", True),
             "F": (5.5, 2.2, "Flag $F$", False), "S": (5.5, 0.0, "Biopsy $S$", False), "Y": (9.6, 0.0, "Label $Y$", False)}
    for k, (x, y, lab, latent) in nodes.items():
        a.add_patch(FancyBboxPatch((x - W_ / 2, y - H_ / 2), W_, H_, boxstyle="round,pad=0.03",
                                   fc="#ffffff" if latent else "#f3f2ef", ec=INK, lw=0.9, ls="--" if latent else "-"))
        a.text(x, y, lab, ha="center", va="center", fontsize=7.0, color=INK)
    def edge(p, q):
        """Điểm trên cạnh hộp quanh p theo hướng tới q."""
        dx, dy = q[0] - p[0], q[1] - p[1]
        t = min((W_ / 2 + 0.08) / abs(dx) if dx else 1e9, (H_ / 2 + 0.08) / abs(dy) if dy else 1e9)
        return (p[0] + t * dx, p[1] + t * dy)
    def arr(u, v, col=INK, rad=0.0, ls="-"):
        p, q = nodes[u][:2], nodes[v][:2]
        a.add_patch(FancyArrowPatch(edge(p, q), edge(q, p), arrowstyle="-|>", mutation_scale=8, color=col, lw=0.9,
                                    connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=0, shrinkB=0))
    for u, v in (("U", "D"), ("U", "X"), ("G", "X"), ("G", "D"), ("D", "X"), ("D", "W"), ("F", "S"), ("S", "Y")):
        arr(u, v)
    arr("X", "C", INK2, ls="--")
    for u, v, r in (("X", "F", 0.0), ("W", "F", 0.0), ("W", "S", 0.0), ("X", "S", -0.45), ("G", "F", -0.45)):
        arr(u, v, ORANGE, r)
    arr("U", "S", ORANGE, 0.55)
    a.add_patch(FancyArrowPatch(edge(nodes["D"][:2], nodes["Y"][:2]), edge(nodes["Y"][:2], nodes["D"][:2]), arrowstyle="-|>",
                                mutation_scale=8, color=INK, lw=0.9, connectionstyle="arc3,rad=-0.12", shrinkA=0, shrinkB=0))
    a.text(9.6, 0.75, r"$Y = D \cdot S$", ha="center", fontsize=7.5, color=INK2)
    a.text(8.0, 7.05, "orange: inputs to\nverification", fontsize=6.8, color=ORANGE)
    a.text(0.0, 7.4, "a", fontsize=11, fontweight="bold")
    regions(b, 6)
    b.set_xlim(-4, 4); b.set_ylim(-4, 4)
    b.text(1.0, 3.2, "reversal", color=INK, fontsize=8); b.text(2.6, 1.0, "attenuation", color=INK, fontsize=8)
    b.text(1.2, -2.5, "amplification", color=INK, fontsize=8); b.text(-3.6, -3.2, "reversal", color=INK, fontsize=8)
    b.text(-3.8, -1.0, "attenuation", color=INK, fontsize=8); b.text(-3.6, 2.5, "amplification", color=INK, fontsize=8)
    b.text(2.2, 2.9, r"$\delta = \theta$", color=INK, fontsize=8, rotation=45)
    b.set_xlabel(r"$\theta = \log \mathrm{OR}_D$  (true disease contrast)")
    b.set_ylabel(r"$\delta = \log B - \log A$  (differential verification)")
    b.text(-4.9, 4.3, "b", fontsize=11, fontweight="bold")
    b.text(0.1, -3.85, r"$\log \mathrm{OR}_{D|S} = \theta - \delta$", fontsize=8, color=INK)
    fig.tight_layout(); save(fig, "fig1_mechanism.png")


def fig2():
    ph = J("vr2_phase_diagram.json")
    fig, a = plt.subplots(1, 1, figsize=(3.9, 3.4))
    regions(a, 9)
    sty = {"reversal": (ORANGE, "^"), "attenuation": (BLUE, "o"), "amplification": (AQUA, "s")}
    for cat, (col, mk) in sty.items():
        rr = [r for r in ph["rows"] if r["category"] == cat]
        a.scatter([r["theta"] for r in rr], [r["delta"] for r in rr], s=8, color=col, marker=mk, alpha=0.6,
                  linewidths=0, label=f"{cat} ({len(rr)})")
    mis = [r for r in ph["rows"] if r["M2_coef_reversed"] != (r["category"] == "reversal")]
    a.scatter([r["theta"] for r in mis], [r["delta"] for r in mis], s=40, facecolors="none", edgecolors=INK, lw=0.9,
              label=f"learner disagrees ({len(mis)})")
    a.set_xlim(-0.3, 3.0); a.set_ylim(-1.5, 9)
    a.set_xlabel(r"$\theta = \log \mathrm{OR}_D$"); a.set_ylabel(r"$\delta = \log B - \log A$")
    a.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(); save(fig, "figS1_phase_diagram.png")


def fig_external():
    fig, b = plt.subplots(1, 1, figsize=(4.2, 3.6))
    regions(b, 6)
    pad, sv2 = J("vr6_pad_boundary.json"), J("sv2_bracket.json", R.MISS_RES)
    f = pad["features"]
    b.scatter([v["logOR_D"] for v in f.values()], [v["logB"] - v["logA"] for v in f.values()], s=26, marker="D",
              color=VIOLET, edgecolor=SURF, linewidth=0.6, zorder=3, label="PAD-UFES-20, recorded diagnoses")
    for c in ("color_variegation", "size", "lesion_skin_contrast"):
        ad = sv2["proxies"][c]["adjusted"]
        x, y = np.log(ad["OR_Y"]), np.log(ad["B"])
        b.plot([x - np.log(2), x + np.log(2)], [y + np.log(2), y - np.log(2)], color=YELLOW, lw=2.0, zorder=2,
               label="ISIC-2024 sensitivity path, A in [0.5, 2]" if c == "color_variegation" else None)
        b.scatter([x], [y], s=46, marker="o", facecolors=SURF, edgecolors=INK, linewidth=1.1, zorder=4,
                  label="ISIC-2024 reference, A = 1" if c == "color_variegation" else None)
        off = {"color_variegation": (8, 5), "size": (-8, -11), "lesion_skin_contrast": (-8, 6)}[c]
        b.annotate(CL[c], (x, y), xytext=off, textcoords="offset points", fontsize=6.8, ha="right" if off[0] < 0 else "left")
    b.set_xlim(-0.6, 2.8); b.set_ylim(-0.6, 2.8)
    b.set_xlabel(r"$\theta = \log \mathrm{OR}_D$"); b.set_ylabel(r"$\delta = \log B - \log A$")
    b.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout(); save(fig, "fig4_external_boundary.png")


def fig3():
    d = J("vr9_closing.json")["families"]
    fams = [("head_tabular", "Frozen features: tabular"), ("head_image", "Frozen features: image"),
            ("lp", "Linear probe"), ("ft", "Fine-tuned ResNet-50")]
    fig, axes = plt.subplots(1, 4, figsize=(8.4, 3.2), sharey=True)
    concepts = list(CL)[::-1]
    for ax, (fam, title) in zip(axes, fams):
        for j, c in enumerate(concepts):
            t = d[fam]["concepts"][c]["tertile"]
            for key, col, mk, off, lab in (("M0", BLUE, "o", 0.13, "M0: all lesions, Y"),
                                           ("M2", ORANGE, "^", -0.13, "M2: verified lesions only")):
                lo, hi = t[f"{key}_ci"]
                ax.plot([lo, hi], [j + off] * 2, color=col, lw=1.6)
                ax.plot(t[key], j + off, marker=mk, color=col, ms=5.5, mec=SURF, mew=0.7, ls="none",
                        label=lab if j == 0 else None)
        ax.axvline(0, color=INK2, lw=1); ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.set_xlabel(r"Learned contrast $\Delta$ (logit)")
    axes[0].set_yticks(range(len(concepts))); axes[0].set_yticklabels([CL[c] for c in concepts])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); save(fig, "fig2_learned_reversal.png")


def fig4():
    d = J("vr46_dose_fixed_val.json")
    br = J("vr22_pointwise_bridge.json")["families"] if os.path.exists(os.path.join(R.RES, "vr22_pointwise_bridge.json")) else None
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.3))
    etas = np.array(d["etas"])
    for fam, col, mk in (("tabular", BLUE, "o"), ("image", ORANGE, "^")):
        r = d["families"][fam]["color_variegation|320"]
        reps = json.load(open(os.path.join(R.RES, "vr46", f"{fam}_color_variegation_320.json")))["reps"].values()
        M_ = np.array([[o["color_variegation"] for o in rp["raw"]] for rp in reps])      # tương phản logit thô, validation cố định
        m, lo, hi = M_.mean(0), np.percentile(M_, 2.5, 0), np.percentile(M_, 97.5, 0)
        a.fill_between(etas, lo, hi, color=col, alpha=0.15, lw=0)
        a.plot(etas, m, marker=mk, color=col, lw=1.6, ms=5, label=f"{fam}, learned")
        d0 = m[list(etas).index(0.0)]
        sl = r["pred"]["slope"]
        a.plot(etas, d0 + sl * etas, color=col, lw=1.1, ls="--", label=f"{fam}, predicted from input")
    a.axhline(0, color=INK2, lw=0.9); a.grid(color=GRID, lw=0.6); a.set_xlim(etas.min() - 0.2, etas.max() + 0.2)
    a.set_xlabel(r"Selection dose $\eta$ on recorded-negative training lesions"); a.set_ylabel(r"Learned logit $\Delta$, color variegation")
    a.legend(frameon=False, fontsize=6.6)
    a.text(etas.min() - 0.75, a.get_ylim()[1], "a", fontsize=11, fontweight="bold")
    lim = 2.6
    if br is not None:
        for fam, col, mk in (("tabular", BLUE, "o"), ("image", ORANGE, "^")):
            for cc in R.CONCEPTS:
                v = br[fam]["concepts"][cc]
                x, y = v["predicted"], v["observed"]
                b.errorbar(x, y, xerr=[[x - v["predicted_ci"][0]], [v["predicted_ci"][1] - x]],
                           yerr=[[y - v["observed_ci"][0]], [v["observed_ci"][1] - y]], fmt=mk, color=col, ms=5, lw=0.8,
                           mec=SURF, mew=0.6, capsize=0, zorder=3, label=f"{fam}" if cc == "color_variegation" else None)
                if fam == "tabular":
                    off = {"color_variegation": (-6, 6), "lesion_skin_contrast": (6, -8), "size": (-6, 4),
                           "asymmetry": (6, 4), "border_irregularity": (-6, -10)}[cc]
                    b.annotate(CL[cc].split()[0].replace("Lesion-skin", "Contrast"), (x, y), xytext=off, textcoords="offset points",
                               fontsize=6.3, ha="right" if off[0] < 0 else "left")
            lim = max(lim, *(abs(z) for cc in R.CONCEPTS for z in br[fam]["concepts"][cc]["observed_ci"] + br[fam]["concepts"][cc]["predicted_ci"]))
    lim = float(np.ceil(lim * 5) / 5)
    b.plot([-lim, lim], [-lim, lim], color=INK, lw=1); b.axhline(0, color=INK2, lw=0.8); b.axvline(0, color=INK2, lw=0.8)
    b.set_xlim(-lim, lim); b.set_ylim(-lim, lim); b.grid(color=GRID, lw=0.6)
    b.set_xlabel(r"Predicted gap, $E[\log \hat g \mid t=1] - E[\log \hat g \mid t=0]$")
    b.set_ylabel(r"Observed gap $\Delta_{M0} - \Delta_{M2}$")
    b.legend(frameon=False, fontsize=6.8, loc="upper left")
    b.text(-lim * 1.3, lim, "b", fontsize=11, fontweight="bold")
    fig.tight_layout(); save(fig, "fig3_intervention_bridge.png")


def figS1():
    sm = J("vr5_representation.json")["summary"]["ft"]
    keys = [("decision", "Decision contrast"), ("layer3", "Layer-3 intervention"), ("readout_Y", "Common readout, target $Y$"),
            ("readout_S", "Common readout, target $S$")]
    fig, axes = plt.subplots(1, 4, figsize=(8.6, 2.9))
    cs = R.IMAGE_CONCEPTS
    for ax, (k, t) in zip(axes, keys):
        for j, c in enumerate(cs):
            for reg, col, mk, off in (("M0", BLUE, "o", 0.12), ("M2", ORANGE, "^", -0.12)):
                v = sm[k][c][reg]
                ax.plot(v, [j + off] * len(v), ls="none", marker=mk, color=col, ms=5, label=reg if j == 0 else None)
        ax.axvline(0, color=INK2, lw=1); ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_yticks(range(len(cs))); ax.set_yticklabels([CL[c] for c in cs] if ax is axes[0] else [])
        ax.set_title(t, loc="left", fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, ["M0 fine-tuned on $Y$", "M2 fine-tuned on verified lesions"], loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86)); save(fig, "figS2_finetune.png")


def figS2():
    d = J("vr3_icdl.json")
    rows = [r for r in d["sim_rows"] if r["method"] == "icdl_kl"]
    st = sorted({r["s_true"] for r in rows}); sa = sorted({r["s_assumed"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax, key, title in ((axes[0], "true_p_in_oracle_interval", "True $p$ inside oracle interval"),
                           (axes[1], "abs_log_err", r"Mean $|\log \hat p - \log p|$")):
        M = np.full((len(st), len(sa)), np.nan)
        for r in rows:
            M[st.index(r["s_true"]), sa.index(r["s_assumed"])] = r[key]
        im = ax.imshow(M, cmap="Blues", aspect="auto", origin="lower")
        for i in range(len(st)):
            for j in range(len(sa)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.3,
                        color="white" if M[i, j] > np.nanpercentile(M, 70) else INK)
        ax.set_xticks(range(len(sa))); ax.set_xticklabels([f"{s:g}" for s in sa])
        ax.set_yticks(range(len(st))); ax.set_yticklabels([f"{s:g}" for s in st])
        ax.set_xlabel(r"assumed $s_{\min}$"); ax.set_ylabel(r"true $s_{\min}$"); ax.set_title(title, loc="left", fontsize=8.5)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); save(fig, "figS4_icdl.png")


def figS3():
    sc = J("vr3_icdl.json")["isic"]["tabular"]["structured_contrast_s0.5"]
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    cs = list(sc)
    for j, c in enumerate(cs):
        for h, col, off, lab in (("rect", YELLOW, 0.2, r"per-lesion $s$ (Proposition 1)"), ("strata", BLUE, 0.0, r"$s$ shared in 5 strata"),
                                 ("shared", ORANGE, -0.2, r"one shared $s$ ($A = 1$)")):
            lo, hi = sc[c][h]
            if h == "shared":
                ax.plot([lo], [j + off], marker="D", color=col, ms=5, ls="none", label=lab if j == 0 else None)
            else:
                ax.plot([lo, hi], [j + off] * 2, color=col, lw=3, solid_capstyle="butt", label=lab if j == 0 else None)
    ax.set_xscale("log"); ax.axvline(1, color=INK2, lw=1)
    ax.set_xticks([0.5, 1, 2, 4]); ax.set_xticklabels(["0.5", "1", "2", "4"])
    ax.set_yticks(range(len(cs))); ax.set_yticklabels([CL[c] for c in cs])
    ax.set_xlabel(r"Identified set for $\mathrm{RR}_D$ at $s_{\min} = 0.5$"); ax.grid(axis="x", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=7, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3)
    fig.tight_layout(); save(fig, "figS3_identified_sets.png")


if __name__ == "__main__":
    for f in (fig1, fig2, fig_external, fig3, fig4, figS1, figS2):
        f()
