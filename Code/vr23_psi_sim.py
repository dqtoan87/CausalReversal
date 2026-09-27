#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR23 — Mô phỏng kiểm tập định danh của mục tiêu bệnh ψ_k (Proposition 2) trên cùng functional với Δ.

SCM như vr3 (π¹(x) = s_true + (1 − s_true)·expit(w₁ᵀc) ≥ s_true), thêm biến thể concept tương quan (ρ = 0.5).
Với learner quan sát toàn bộ x: ψ_k thật = E[logit p | t=1] − E[logit p | t=0] trên quần thể đánh giá.
Tập định danh [ψ_L(s), ψ_U(s)] dựng từ q, σ thật (oracle) và từ q̂, σ̂ ước lượng bằng GBM (plug-in). Đo: tỉ lệ ψ thật
nằm trong tập (oracle phải là 1 khi s ≤ s_true), độ rộng, và tỉ lệ dấu được chứng nhận sai (ψ_L > 0 mà ψ < 0 hoặc ngược lại).

Out -> Result/vr23_psi_sim.json
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

import vr_common as R

BETA = np.array([0.5, 0.4, 0.3, 0.4, 0.4])
FW = np.array([1.2, 0.8, 1.5, -1.0, -1.0]); SW = np.array([0.0, 0.3, -0.6, 0.8, 0.9]); W1 = np.array([0.8, 0.5, 0.8, -0.5, -0.5])


def expit(z):
    return 1 / (1 + np.exp(-z))


def simulate(rng, n, s_true, rho, prev=0.005):
    cov = np.full((5, 5), rho); np.fill_diagonal(cov, 1.0)
    c = rng.multivariate_normal(np.zeros(5), cov, n)
    x = np.column_stack([c, rng.standard_normal((n, 3))])
    a = brentq(lambda a: expit(a + c @ BETA).mean() - prev, -20, 5)
    p = expit(a + c @ BETA)
    f0 = brentq(lambda f: expit(f + c @ FW).mean() - 0.05, -30, 10)
    s0 = brentq(lambda s: (expit(f0 + c @ FW) * expit(s + c @ SW)).mean() - 0.003, -30, 10)
    pi0 = expit(f0 + c @ FW) * expit(s0 + c @ SW); pi1 = s_true + (1 - s_true) * expit(c @ W1)
    D = rng.random(n) < p; S = rng.random(n) < np.where(D, pi1, pi0); Y = D & S
    return x, p, p * pi1, p * pi1 + (1 - p) * pi0, Y.astype(int), S.astype(int)


def bounds(q, sig, T, s):
    lq = R.logit(q); U = np.clip(np.minimum(q / s, q + 1 - sig), 1e-7, 1 - 1e-7); lU = R.logit(U)
    return lq[T == 1].mean() - lU[T == 0].mean(), lU[T == 1].mean() - lq[T == 0].mean()


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(R.SEED)
    rows = []
    for rho in (0.0, 0.5):
        for s_true in (0.5, 0.7, 0.9):
            x, p, q, sig, Y, S = simulate(rng, 300_000, s_true, rho)
            idx = rng.permutation(len(Y)); tr, te = idx[:200_000], idx[200_000:]
            qh = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.05, random_state=0).fit(x[tr], Y[tr]).predict_proba(x[te])[:, 1]
            sh = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.05, random_state=0).fit(x[tr], S[tr]).predict_proba(x[te])[:, 1]
            qh, sh = np.clip(qh, 1e-6, 1 - 1e-6), np.clip(sh, 1e-6, 1 - 1e-6)
            for k in range(5):
                T = np.where(x[te, k] <= np.quantile(x[:, k], 1 / 3), 0, np.where(x[te, k] >= np.quantile(x[:, k], 2 / 3), 1, -1))
                lp = R.logit(p[te]); psi = lp[T == 1].mean() - lp[T == 0].mean()
                for s in (0.3, 0.5, 0.7, 0.9):
                    oL, oU = bounds(q[te], sig[te], T, s); eL, eU = bounds(qh, sh, T, s)
                    rows.append({"rho": rho, "s_true": s_true, "s_assumed": s, "concept": k, "psi": float(psi),
                                 "oracle": [float(oL), float(oU)], "plugin": [float(eL), float(eU)],
                                 "oracle_covers": bool(oL - 1e-9 <= psi <= oU + 1e-9), "plugin_covers": bool(eL <= psi <= eU),
                                 "oracle_wrong_sign": bool((oL > 0 and psi < 0) or (oU < 0 and psi > 0)),
                                 "plugin_wrong_sign": bool((eL > 0 and psi < 0) or (eU < 0 and psi > 0)),
                                 "oracle_certifies": bool(oL > 0 or oU < 0), "plugin_certifies": bool(eL > 0 or eU < 0)})
            R.log(f"rho={rho} s_true={s_true} done")
    def agg(sel):
        rr = [r for r in rows if sel(r)]
        return {k: float(np.mean([r[k] for r in rr])) for k in ("oracle_covers", "plugin_covers", "oracle_wrong_sign", "plugin_wrong_sign",
                                                              "oracle_certifies", "plugin_certifies")} | {"n": len(rr)}
    summ = {"valid_floor": agg(lambda r: r["s_assumed"] <= r["s_true"]), "overstated_floor": agg(lambda r: r["s_assumed"] > r["s_true"]),
            "valid_floor_correlated": agg(lambda r: r["s_assumed"] <= r["s_true"] and r["rho"] > 0)}
    R.log(summ)
    R.save_json({"summary": summ, "rows": rows}, "vr23_psi_sim.json")


if __name__ == "__main__":
    main()
