#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR35 — Ước lượng bổ trợ biên: tỉ số nguy cơ bệnh giữa hai tertile ngoài, tính TRỰC TIẾP từ số đếm (không dùng learner).

RR_Y = P(Y=1 | t=1) / P(Y=1 | t=0). Với P(D=1 | t) = P(Y=1 | t) / s̄_t và s̄_t = P(S=1 | D=1, t) ∈ [s, 1] (chỉ cần sàn trên
trung bình của từng tertile, yếu hơn sàn theo từng điểm), RR_D ∈ [RR_Y·s, RR_Y/s]; dấu dương được xác định khi s > 1/RR_Y.
Báo cáo trên quần thể test và toàn cohort, cho năm concept: RR_Y, khoảng 95% bootstrap cụm bệnh nhân (2,000 lần),
sàn tới hạn 1/RR_Y và sàn một phía 95% 1/q₀.₀₅(RR_Y); log OR của Y theo tertile trong tổn thương đã xác minh (liên hệ
chỉ-trên-đã-xác-minh) kèm khoảng 95%.

Out -> Result/vr35_marginal_rr.json
"""
from __future__ import annotations

import numpy as np

import vr_common as R

B = 2000


def main():
    df, fold = R.load_isic()
    Y, S = df.Y.to_numpy(), df.S.to_numpy()
    pid = df[R.SV.GROUP].to_numpy()
    out = {"B": B, "populations": {}}
    rng = np.random.default_rng(35)
    for popn, mask in (("test", fold == "test"), ("cohort", np.ones(len(df), bool))):
        ix = np.flatnonzero(mask); pats = np.unique(pid[ix]); pidx = {p: i for i, p in enumerate(pats)}
        pcode = np.array([pidx[p] for p in pid[ix]])
        W = np.zeros((B, len(pats)))
        for b in range(B):
            np.add.at(W[b], rng.integers(0, len(pats), len(pats)), 1.0)
        res = {}
        for c, col in R.CONCEPTS.items():
            T = R.SV.tertile(df, col).to_numpy()[ix]
            y, s = Y[ix], S[ix]
            def stats(w):
                wl = w[pcode]
                p1 = (wl * y * (T == 1)).sum() / (wl * (T == 1)).sum(); p0 = (wl * y * (T == 0)).sum() / (wl * (T == 0)).sum()
                v1 = (wl * s * (T == 1)); v0 = (wl * s * (T == 0))
                a = (v1 * y).sum(); bb = (v1 * (1 - y)).sum(); cc = (v0 * y).sum(); dd = (v0 * (1 - y)).sum()
                n1 = (wl * (1 - y) * (T == 1)).sum(); n0 = (wl * (1 - y) * (T == 0)).sum()
                return p1 / p0, np.log((a / bb) / (cc / dd)), (bb / n1) / (dd / n0)      # B_V = g₁/g₀ trong nhãn âm
            rr0, lor0, bv0 = stats(np.ones(len(pats)))
            bs = np.array([stats(W[b]) for b in range(B)])
            q05 = float(np.percentile(bs[:, 0], 5))
            res[c] = {"n_pos": [int(y[T == 0].sum()), int(y[T == 1].sum())], "rr_y": float(rr0), "rr_y_ci95": np.percentile(bs[:, 0], [2.5, 97.5]).tolist(),
                      "tipping": float(1 / rr0) if rr0 > 1 else None, "tipping_one_sided": float(1 / q05) if q05 > 1 else None,
                      "log_or_verified": float(lor0), "log_or_verified_ci95": np.percentile(bs[:, 1], [2.5, 97.5]).tolist(),
                      # biến cố đồng thời trong cùng lần lặp: dấu dương của RR_D được xác định tại sàn s (RR_Y·s > 1) và liên hệ
                      # trong tổn thương đã xác minh âm (log OR < 0), tức đảo dấu có điều kiện ở mức liên hệ biên
                      "joint_counts": {f"{s:.2f}": int(np.sum((bs[:, 0] * s > 1) & (bs[:, 1] < 0))) for s in np.round(np.arange(0.5, 1.0001, 0.05), 2)},
                      "verified_negative_count": int(np.sum(bs[:, 1] < 0)),
                      # tham số hóa trực tiếp bằng A = π₁¹/π₀¹: RR_D > 1 ⇔ A < RR_Y. A lớn nhất mà biến cố đồng thời
                      # (RR_Y > A và log OR trong đã xác minh < 0) giữ trong ít nhất 95% lần lặp; sàn một phía tương ứng 1/A.
                      "A_max95": (lambda g: float(g[-1]) if len(g) else None)(
                          [A for A in np.round(np.arange(1.0, 4.0001, 0.01), 2) if np.sum((bs[:, 0] > A) & (bs[:, 1] < 0)) >= 0.95 * B]),
                      "B_V": float(bv0), "B_V_ci95": np.percentile(bs[:, 2], [2.5, 97.5]).tolist()}
            R.log(f"{popn} {c:22s} RR_Y {rr0:.2f} {np.round(res[c]['rr_y_ci95'], 2)} tip {res[c]['tipping']} one-sided {res[c]['tipping_one_sided']} "
                  f"logOR|S {lor0:+.2f} {np.round(res[c]['log_or_verified_ci95'], 2)}")
        out["populations"][popn] = res
    R.save_json(out, "vr35_marginal_rr.json")


if __name__ == "__main__":
    main()
