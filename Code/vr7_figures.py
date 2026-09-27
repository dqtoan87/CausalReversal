#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR7 — Hình cho VILRR (chỉ đọc JSON; hình nào thiếu đầu vào thì bỏ qua).

  fig1_phase_diagram.png      Định lý 4: mặt phẳng (θ, δ), lưới mô phỏng + PAD-UFES (A = 1) + ISIC-2024 (giả sử A = 1)
  fig2_theorem5.png           Định lý 5: hệ số M2 dự đoán từ quan sát (β_M0 − β_V) so với hệ số M2 thật
  fig3_misspecification.png   ICDL: sai số log và độ phủ oracle trên lưới s_true × s_assumed
  fig4_representation.png     fine-tune: tương phản quyết định, can thiệp layer3, readout chung, M0 so với M2
  fig5_isic_assumption.png    ISIC-2024: phân phối tipping point s_i* và cận tương phản dưới ba ℋ

Màu: khe categorical đã kiểm (dataviz validate_palette). Mọi chuỗi có hình dấu riêng và chú giải.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import vr_common as R

BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CL = {"color_variegation": "Color", "size": "Size", "lesion_skin_contrast": "Contrast", "asymmetry": "Asymmetry",
      "border_irregularity": "Border"}
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False})


def J(name, base=R.RES):
    p = os.path.join(base, name)
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    fig.savefig(os.path.join(R.RES, name), dpi=200); plt.close(fig); R.log(f"wrote {name}")


def fig1():
    ph, pad, sv2 = J("vr2_phase_diagram.json"), J("vr6_pad_boundary.json"), J("sv2_bracket.json", R.MISS_RES)
    if not ph:
        return
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    sty = {"reversal": (ORANGE, "^", "Simulation: reversal"), "attenuation": (BLUE, "o", "Simulation: attenuation"),
           "amplification": (AQUA, "s", "Simulation: amplification")}
    for cat, (col, mk, lab) in sty.items():
        rr = [r for r in ph["rows"] if r["category"] == cat]
        ax.scatter([r["theta"] for r in rr], [r["delta"] for r in rr], s=9, color=col, marker=mk, alpha=0.45,
                   label=lab, linewidths=0)
    lim = [-1, 8]
    ax.plot(lim, lim, color=INK, lw=1.2); ax.axhline(0, color=INK2, lw=0.8)
    ax.text(5.6, 6.3, "δ = θ (sign boundary)", color=INK, fontsize=7.5, rotation=36)
    if pad:
        f = pad["features"]
        ax.scatter([v["logOR_D"] for v in f.values()], [v["logB"] - v["logA"] for v in f.values()], s=34,
                   marker="D", color=VIOLET, edgecolor=SURF, linewidth=0.8, label="PAD-UFES (A = 1)", zorder=3)
    if sv2:
        for c, v in sv2["proxies"].items():
            if c in ("color_variegation", "size", "lesion_skin_contrast", "asymmetry", "border_irregularity"):
                a = v["adjusted"]
                x, y = np.log(a["OR_Y"]), np.log(a["B"])
                ax.scatter([x], [y], s=46, marker="*", color=YELLOW, edgecolor=INK, linewidth=0.5, zorder=4,
                           label="ISIC-2024, if A = 1" if c == "color_variegation" else None)
                ax.annotate(CL[c], (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK)
    ax.set_xlim(-1.5, 4.5); ax.set_ylim(-1.5, 8)
    ax.set_xlabel("θ = log OR_D (true disease contrast)"); ax.set_ylabel("δ = log B − log A (differential verification)")
    ax.grid(color=GRID, lw=0.6); ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(); save(fig, "fig1_phase_diagram.png")


def fig2():
    ph, th = J("vr2_phase_diagram.json"), J("vr1_reversal_theory.json")
    if not ph:
        return
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    x = [r["M2_pred_observable"] for r in ph["rows"]]; y = [r["beta_M2"] for r in ph["rows"]]
    ax.scatter(x, y, s=8, color=BLUE, alpha=0.5, linewidths=0, label="Phase-diagram grid (population)")
    if th:
        s = th["isic2024_slopes"]["single"]
        ax.scatter([v["slope_M0"] - v["slope_V"] for v in s.values()], [v["slope_M2"] for v in s.values()], s=40,
                   marker="*", color=ORANGE, edgecolor=INK, linewidth=0.5, label="ISIC-2024 concepts", zorder=3)
        for c, v in s.items():
            ax.annotate(CL[c], (v["slope_M0"] - v["slope_V"], v["slope_M2"]), xytext=(4, -8),
                        textcoords="offset points", fontsize=7)
    m = max(np.abs(x).max(), np.abs(y).max()) * 1.05
    ax.plot([-m, m], [-m, m], color=INK, lw=1); ax.axhline(0, color=INK2, lw=0.8); ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Predicted from observables: β_M0 − β_V"); ax.set_ylabel("Verified-only learner: β_M2")
    ax.grid(color=GRID, lw=0.6); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); save(fig, "fig2_theorem5.png")


def fig3():
    d = J("vr3_icdl.json")
    if not d or "sim_rows" not in d:
        return
    rows = [r for r in d["sim_rows"] if r["method"] == "icdl_kl"]
    st = sorted({r["s_true"] for r in rows}); sa = sorted({r["s_assumed"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    for ax, key, title, cmap in ((axes[0], "abs_log_err", "Mean |log p̂ − log p|", "Blues"),
                                 (axes[1], "pred_in_oracle_interval", "Prediction inside oracle 𝓘_D", "Blues")):
        M = np.full((len(st), len(sa)), np.nan)
        for r in rows:
            M[st.index(r["s_true"]), sa.index(r["s_assumed"])] = r[key]
        im = ax.imshow(M, cmap=cmap, aspect="auto", origin="lower")
        for i in range(len(st)):
            for j in range(len(sa)):
                v = M[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > np.nanmean(M) + 0.5 * np.nanstd(M) else INK)
        ax.set_xticks(range(len(sa))); ax.set_xticklabels([f"{s:g}" for s in sa])
        ax.set_yticks(range(len(st))); ax.set_yticklabels([f"{s:g}" for s in st])
        ax.set_xlabel("s_min assumed"); ax.set_ylabel("s_min true"); ax.set_title(title, loc="left", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); save(fig, "fig3_misspecification.png")


def fig4():
    d = J("vr5_representation.json")
    if not d or "ft" not in d.get("summary", {}):
        return
    sm = d["summary"]["ft"]
    keys = [("decision", "Decision contrast"), ("layer3", "Layer-3 intervention Δ"), ("readout_Y", "Common readout (target Y)")]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.0))
    for ax, (k, t) in zip(axes, keys):
        cs = [c for c in R.IMAGE_CONCEPTS if c in sm[k]]
        for j, c in enumerate(cs):
            for reg, col, mk, off in (("M0", BLUE, "o", -0.12), ("M2", ORANGE, "^", 0.12)):
                v = sm[k][c][reg]
                ax.plot(v, [j + off] * len(v), ls="none", marker=mk, color=col, ms=5.5,
                        label=f"{reg}" if j == 0 else None)
        ax.axvline(0, color=INK2, lw=1); ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_yticks(range(len(cs))); ax.set_yticklabels([CL[c] for c in cs] if ax is axes[0] else [])
        ax.set_title(t, loc="left", fontsize=8.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, ["M0 fine-tuned on Y", "M2 fine-tuned on verified"], loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88)); save(fig, "fig4_representation.png")


def fig5():
    d = J("vr3_icdl.json")
    if not d or "isic" not in d or "tabular" not in d["isic"]:
        return
    t = d["isic"]["tabular"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    fr = t["decisions"]["0.01"]["frac_s_star_above"]
    xs = [float(k) for k in fr]; ys = [fr[k] for k in fr]
    axes[0].plot(xs, ys, marker="o", color=BLUE, lw=2)
    axes[0].set_xlabel("Assumed verification floor s"); axes[0].set_ylabel("Fraction of lesions with s* > s")
    axes[0].set_title("Benign decision needs s ≥ s*  (τ = 0.01)", loc="left", fontsize=8.5)
    axes[0].grid(color=GRID, lw=0.6)
    sc = t["structured_contrast_s0.5"]
    cs = list(sc)
    for j, c in enumerate(cs):
        for h, col, off, lab in (("rect", YELLOW, -0.2, "ℋ rectangular"), ("strata", BLUE, 0.0, "ℋ 5 strata"),
                                 ("shared", ORANGE, 0.2, "ℋ shared (A = 1)")):
            lo, hi = sc[c][h]
            axes[1].plot([lo, hi], [j + off] * 2, color=col, lw=3 if h != "shared" else 0, solid_capstyle="butt",
                         label=lab if j == 0 else None)
            if h == "shared":
                axes[1].plot([lo], [j + off], marker="D", color=col, ms=5, ls="none", label=lab if j == 0 else None)
    axes[1].set_xscale("log"); axes[1].axvline(1, color=INK2, lw=1)
    axes[1].set_yticks(range(len(cs))); axes[1].set_yticklabels([CL[c] for c in cs])
    axes[1].set_xlabel("Disease risk ratio, upper vs lower tertile"); axes[1].grid(axis="x", color=GRID, lw=0.6)
    axes[1].set_title("Identified set of RR_D at s = 0.5", loc="left", fontsize=8.5)
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(); save(fig, "fig5_isic_assumption.png")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5):
        try:
            f()
        except Exception as e:                       # một hình hỏng không chặn các hình khác
            R.log(f"{f.__name__} failed: {e!r}")
