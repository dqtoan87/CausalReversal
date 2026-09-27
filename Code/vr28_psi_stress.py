#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR28 — Mô phỏng căng thẳng cho tập định danh ψ_k (Proposition 2): tìm chế độ lỗi "chứng nhận sai dấu".

Mở rộng vr23:
  - sàn thật thấp: s_true ∈ {0.1, 0.3, 0.5, 0.9}; sàn giả định ∈ {0.1, 0.2, 0.3, 0.5, 0.7, 0.9} (nhiều ô bị phóng đại);
  - kịch bản đối nghịch: xác minh ác tính phụ thuộc mạnh vào concept (W1 lớn, cùng hướng với xác minh lành tính) và
    hiệu ứng bệnh của concept gần 0 hoặc trái dấu, để q̂ có tương phản dương trong khi ψ ≤ 0;
  - ba cách dựng tập: oracle (q, σ thật), plug-in gradient boosting, và quy trình gần với phân tích ISIC (MLP rồi
    hiệu chỉnh Platt trên tập val riêng; 30 epoch cố định vì dừng sớm của sklearn chấm theo độ chính xác, vô nghĩa khi tỉ lệ bệnh 0.5%).
Báo cáo theo (kịch bản, sàn hợp lệ hay phóng đại): độ bao phủ, tỉ lệ chứng nhận dấu, tỉ lệ chứng nhận SAI dấu.

Out -> Result/vr28_psi_stress.json
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

import vr_common as R
from vr23_psi_sim import bounds, expit, FW, SW

SCEN = {
    "base": {"beta": np.array([0.5, 0.4, 0.3, 0.4, 0.4]), "w1": np.array([0.8, 0.5, 0.8, -0.5, -0.5])},
    "adversarial": {"beta": np.array([0.5, 0.0, -0.3, 0.05, -0.2]), "w1": np.array([1.0, 2.5, 2.5, 2.0, 2.0])},
}
S_TRUE = [0.1, 0.3, 0.5, 0.9]
S_ASSUMED = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]


def simulate(rng, n, s_true, beta, w1, prev=0.005):
    c = rng.standard_normal((n, 5))
    x = np.column_stack([c, rng.standard_normal((n, 3))])
    a = brentq(lambda a: expit(a + c @ beta).mean() - prev, -20, 5)
    p = expit(a + c @ beta)
    f0 = brentq(lambda f: expit(f + c @ FW).mean() - 0.05, -30, 10)
    s0 = brentq(lambda s: (expit(f0 + c @ FW) * expit(s + c @ SW)).mean() - 0.003, -30, 10)
    pi0 = expit(f0 + c @ FW) * expit(s0 + c @ SW); pi1 = s_true + (1 - s_true) * expit(-1.0 + c @ w1)
    D = rng.random(n) < p; S = rng.random(n) < np.where(D, pi1, pi0); Y = D & S
    return x, p, p * pi1, p * pi1 + (1 - p) * pi0, Y.astype(int), S.astype(int)


def mlp_platt(x, y, tr, va, te):
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    m = MLPClassifier(hidden_layer_sizes=(128, 128), alpha=1e-4, batch_size=4096, max_iter=30, early_stopping=False,
                      random_state=0).fit(x[tr], y[tr])
    lg = lambda ix: R.logit(np.clip(m.predict_proba(x[ix])[:, 1], 1e-7, 1 - 1e-7))
    cal = LogisticRegression(C=1e6, max_iter=2000).fit(lg(va)[:, None], y[va])
    return np.clip(cal.predict_proba(lg(te)[:, None])[:, 1], 1e-6, 1 - 1e-6)


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(R.SEED + 7)
    rows = []
    for scen, prm in SCEN.items():
        for s_true in S_TRUE:
            x, p, q, sig, Y, S = simulate(rng, 300_000, s_true, prm["beta"], prm["w1"])
            idx = rng.permutation(len(Y)); tr, va, te = idx[:160_000], idx[160_000:200_000], idx[200_000:]
            gb = lambda y: np.clip(HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.05, random_state=0)
                                   .fit(x[np.r_[tr, va]], y[np.r_[tr, va]]).predict_proba(x[te])[:, 1], 1e-6, 1 - 1e-6)
            est = {"gbm": (gb(Y), gb(S)), "mlp_platt": (mlp_platt(x, Y, tr, va, te), mlp_platt(x, S, tr, va, te))}
            for k in range(5):
                T = np.where(x[te, k] <= np.quantile(x[:, k], 1 / 3), 0, np.where(x[te, k] >= np.quantile(x[:, k], 2 / 3), 1, -1))
                lp = R.logit(p[te]); psi = float(lp[T == 1].mean() - lp[T == 0].mean())
                for s in S_ASSUMED:
                    r = {"scenario": scen, "s_true": s_true, "s_assumed": s, "concept": k, "psi": psi}
                    for nm, (qq, ss) in (("oracle", (q[te], sig[te])), ("gbm", est["gbm"]), ("mlp_platt", est["mlp_platt"])):
                        L, U = bounds(qq, ss, T, s)
                        r[nm] = {"L": float(L), "U": float(U), "covers": bool(L - 1e-9 <= psi <= U + 1e-9),
                                 "certifies": bool(L > 0 or U < 0), "wrong": bool((L > 0 and psi < 0) or (U < 0 and psi > 0))}
                    rows.append(r)
            R.log(f"{scen} s_true={s_true} done; ψ = {[round(r['psi'], 2) for r in rows[-5 * len(S_ASSUMED)::len(S_ASSUMED)]]}")
    summ = {}
    for scen in SCEN:
        for lab, sel in (("valid", lambda r: r["s_assumed"] <= r["s_true"]), ("overstated", lambda r: r["s_assumed"] > r["s_true"])):
            rr = [r for r in rows if r["scenario"] == scen and sel(r)]
            summ[f"{scen}|{lab}"] = {"n": len(rr), "n_psi_nonpositive": int(sum(r["psi"] <= 0 for r in rr))} | {
                f"{nm}_{m}": float(np.mean([r[nm][m] for r in rr])) for nm in ("oracle", "gbm", "mlp_platt") for m in ("covers", "certifies", "wrong")}
            R.log(f"{scen}|{lab}: {summ[f'{scen}|{lab}']}")
    R.save_json({"summary": summ, "rows": rows}, "vr28_psi_stress.json")


if __name__ == "__main__":
    main()
