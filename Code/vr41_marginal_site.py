#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR41 — Độ nhạy của kết quả liên hệ biên (vr35) theo cơ sở thu nhận và theo thang dùng để chuyển độ phụ thuộc hình thái.

(a) Theo từng cơ sở: số ác tính ở hai tertile ngoài, RR_Y, log OR trong tổn thương đã xác minh, 1/RR_Y.
(b) Chuẩn hóa theo cơ sở: RR_Y,std = Σ_g w_g P(Y=1 | t=1, g) / Σ_g w_g P(Y=1 | t=0, g), w_g = tỉ lệ tổn thương tertile ngoài của
    cơ sở g. Nếu π₀g¹ ≥ s ở mọi cơ sở (sàn riêng từng cơ sở, cùng mức s) và π₁g¹ ≤ 1 thì RR_D,std ≥ s·RR_Y,std, nên dấu dương
    của tỉ số nguy cơ bệnh chuẩn hóa được xác định khi s·RR_Y,std > 1. Liên hệ chỉ-trên-đã-xác-minh chuẩn hóa: log OR
    Mantel–Haenszel theo cơ sở. Biến cố đồng thời trên lưới sàn bước 0,01, bootstrap cụm bệnh nhân 2,000 lần.
(c) Bỏ từng cơ sở: RR_Y toàn cohort và sàn đồng thời.
(d) Mốc so sánh theo gradient logit: với γ = log B_V, A(p₀, γ) = expit(logit p₀ + γ)/p₀; tìm p₀ nhỏ nhất để A ≤ A_max95
    (từ vr35), và vẽ đường biên trên mặt phẳng (p₀, γ). Đây là một mốc độ nhạy, không phải giả định được dữ liệu định danh.
Out -> Result/vr41_marginal_site.json ; paper/figures/figS3_logit_benchmark.png
"""
from __future__ import annotations

import json
import os

import numpy as np

import vr_common as R

B = 2000
FLOORS = np.round(np.arange(0.40, 1.0001, 0.01), 2)


def expit(z):
    return 1 / (1 + np.exp(-z))


def main():
    df, fold = R.load_isic()
    Y, S = df.Y.to_numpy().astype(float), df.S.to_numpy().astype(float)
    site = df["attribution"].astype(str).to_numpy()
    sites = sorted(np.unique(site)); sidx = np.array([sites.index(x) for x in site])
    pid = df[R.SV.GROUP].to_numpy(); pats = np.unique(pid); pidx = {p: i for i, p in enumerate(pats)}
    pc = np.array([pidx[p] for p in pid])
    rng = np.random.default_rng(41)
    W = [np.ones(len(pats))] + [np.bincount(rng.integers(0, len(pats), len(pats)), minlength=len(pats)).astype(float) for _ in range(B)]
    v35 = json.load(open(os.path.join(R.RES, "vr35_marginal_rr.json")))["populations"]["cohort"]
    out = {"B": B, "sites": sites, "concepts": {}}
    for c in ("color_variegation", "size"):
        T = R.SV.tertile(df, R.CONCEPTS[c]).to_numpy()
        m = (T == 0) | (T == 1)
        e = {"by_site": {}, "loso": {}}
        # (a) theo cơ sở, ước lượng điểm
        for g, nm in enumerate(sites):
            k = m & (sidx == g)
            n1, n0 = Y[k & (T == 1)].sum(), Y[k & (T == 0)].sum()
            N1, N0 = (k & (T == 1)).sum(), (k & (T == 0)).sum()
            a, b_ = (Y * S)[k & (T == 1)].sum(), ((1 - Y) * S)[k & (T == 1)].sum()
            cc, dd = (Y * S)[k & (T == 0)].sum(), ((1 - Y) * S)[k & (T == 0)].sum()
            rr = (n1 / N1) / (n0 / N0) if n0 > 0 and N1 > 0 and N0 > 0 else None
            lor = float(np.log((a * dd) / (b_ * cc))) if min(a, b_, cc, dd) > 0 else None
            e["by_site"][nm] = {"n_pos": [int(n0), int(n1)], "n_vb": [int(dd), int(b_)], "rr_y": rr, "log_or_verified": lor,
                                "floor": (1 / rr) if rr and rr > 1 else None}
        # (b) chuẩn hóa theo cơ sở, bootstrap cụm bệnh nhân
        wg = np.array([(m & (sidx == g)).sum() for g in range(len(sites))], float); wg /= wg.sum()

        def stats(wp, excl=None):
            wl = wp[pc] * m
            if excl is not None:
                wl = wl * (sidx != excl)
            num = den = 0.0; mh_n = mh_d = 0.0
            p1s, p0s = [], []
            for g in range(len(sites)):
                if excl is not None and g == excl:
                    continue
                k = sidx == g
                w1, w0 = (wl * k * (T == 1)).sum(), (wl * k * (T == 0)).sum()
                if w1 == 0 or w0 == 0:
                    continue
                num += wg[g] * (wl * k * (T == 1) * Y).sum() / w1; den += wg[g] * (wl * k * (T == 0) * Y).sum() / w0
                a = (wl * k * (T == 1) * Y * S).sum(); b_ = (wl * k * (T == 1) * (1 - Y) * S).sum()
                cc = (wl * k * (T == 0) * Y * S).sum(); dd = (wl * k * (T == 0) * (1 - Y) * S).sum()
                n = a + b_ + cc + dd
                if n > 0:
                    mh_n += a * dd / n; mh_d += b_ * cc / n
            return num / den, float(np.log(mh_n / mh_d)) if mh_n > 0 and mh_d > 0 else np.nan

        bs = np.array([stats(w) for w in W])
        rr0, lor0 = bs[0]; bb = bs[1:]
        jc = {f"{s:.2f}": int(np.sum((bb[:, 0] * s > 1) & (bb[:, 1] < 0))) for s in FLOORS}
        jf = next((float(k) for k, v in jc.items() if v >= 0.95 * B), None)
        e["standardized"] = {"rr_y": float(rr0), "rr_y_ci95": np.percentile(bb[:, 0], [2.5, 97.5]).tolist(),
                             "log_or_mh": float(lor0), "log_or_mh_ci95": np.nanpercentile(bb[:, 1], [2.5, 97.5]).tolist(),
                             "joint_floor": jf}
        # (c) bỏ từng cơ sở
        for g, nm in enumerate(sites):
            bsg = np.array([stats(w, excl=g) for w in W[:501]])
            r0, l0 = bsg[0]; bg = bsg[1:]
            jcg = {s: int(np.sum((bg[:, 0] * s > 1) & (bg[:, 1] < 0))) for s in FLOORS}
            e["loso"][nm] = {"rr_y": float(r0), "log_or_mh": float(l0),
                             "joint_floor": next((float(s) for s, v in jcg.items() if v >= 0.95 * (len(W[:501]) - 1)), None)}
        # (d) mốc gradient logit
        A_max = v35[c]["A_max95"]; gam = float(np.log(v35[c]["B_V"]))
        p0 = np.linspace(0.01, 1.0, 991)
        A = expit(R.logit(np.clip(p0, 1e-9, 1 - 1e-9)) + gam) / p0
        ok = p0[A <= A_max]
        e["logit_benchmark"] = {"gamma": gam, "A_max95": A_max, "p0_threshold": float(ok.min()) if len(ok) else None,
                                "ratio_ceiling_p0": float(1 / v35[c]["B_V"])}
        out["concepts"][c] = e
        R.log(f"{c}: std RR {rr0:.2f} {np.round(e['standardized']['rr_y_ci95'], 2)} MH logOR {lor0:+.2f} joint floor {jf}; "
              f"logit benchmark p0 ≥ {e['logit_benchmark']['p0_threshold']}")
        R.save_json(out, "vr41_marginal_site.json")
    # hình: đường biên A(p₀, γ) = A_max trên mặt phẳng (p₀, γ)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    gg = np.linspace(0, 2.5, 251); pp = np.linspace(0.05, 1, 400)
    Pg, Gg = np.meshgrid(pp, gg)
    Ag = expit(R.logit(np.clip(Pg, 1e-9, 1 - 1e-9)) + Gg) / Pg
    for c, col, lab in (("color_variegation", "C0", "color variegation"), ("size", "C1", "size")):
        lb = out["concepts"][c]["logit_benchmark"]
        ax.contour(Pg, Gg, Ag, levels=[lb["A_max95"]], colors=col)
        ax.plot([lb["p0_threshold"]], [lb["gamma"]], "o", color=col, label=f"{lab}: γ = log B_V, A ≤ {lb['A_max95']:.2f}")
    ax.set_xlabel("malignant verification in the lower tertile, π₀¹")
    ax.set_ylabel("logit gradient of malignant verification, γ")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    figd = os.path.join(os.path.dirname(R.RES), "paper", "figures"); os.makedirs(figd, exist_ok=True)
    fig.savefig(os.path.join(figd, "figS3_logit_benchmark.png"), dpi=200)
    R.log("wrote figS3_logit_benchmark.png")


if __name__ == "__main__":
    main()
