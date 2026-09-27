#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VL1 — Định lý 2 (khoảng của B) và Định lý 3 (khoảng định danh của p(x) = P(D=1 | x)) trên ISIC-2024.

Định lý 2. SV coi B = a₁/a₀ là quan sát được nhờ xấp xỉ bệnh hiếm. Chặt chẽ hơn: Pr(V=1|t) = (1−pₜ)aₜ
và pₜ chỉ bị chặn, pₜ ∈ [qₜ, min(qₜ/s_min, qₜ + 1 − σₜ)] (Định lý 3 áp cho tertile). Khi đó B nằm trong
một khoảng sắc (vl_common.B_interval), và đầu mút Berkson RR_Y/B cũng vậy.

Định lý 3. Với q(x) = P(Y=1|x), σ(x) = P(S=1|x), s(x) ≥ s_min:
    p(x) ∈ [ q(x), min(q(x)/s_min, q(x) + 1 − σ(x)) ].
Hệ quả kiểm được: dự đoán theo MAR (S ⊥ D | x), tức q(x)/σ(x) = P(Y=1 | x, S=1), nằm trong khoảng khi và
chỉ khi σ(x) ≥ s_min. Tỉ lệ tổn thương mà một learner chỉ-đã-xác-minh dự đoán NGOÀI tập định danh là
một con số của dữ liệu, không phải của mô hình.

Nuisance cross-fit không rò rỉ nhãn test: out-of-fold (5 fold theo bệnh nhân) trên train+val, và một mô
hình khớp trên toàn bộ train+val để dự đoán test. x = đặc trưng TBP + Z (vl_common.tabular_X).

Out -> Result/vl1_identification.json, Result/vl_nuisance.npz
Chạy:  python3 vl1_identification.py
"""
from __future__ import annotations

import numpy as np

import sv_common as SV
import vl_common as V

S_GRID = [0.3, 0.5, 0.7, 0.9]


def nuisance(df, X, fold):
    from sklearn.ensemble import HistGradientBoostingClassifier
    g = df[SV.GROUP].to_numpy()
    tv = fold != "test"
    out = {}
    for name in ("Y", "S", "F"):
        y = df[name].to_numpy()
        p = np.zeros(len(y))
        p[tv] = V.crossfit_prob(X[tv], y[tv], g[tv])
        m = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.05,
                                           l2_regularization=1.0, random_state=V.SEED).fit(X[tv], y[tv])
        p[~tv] = m.predict_proba(X[~tv])[:, 1]
        out[name] = np.clip(p, 1e-6, 1 - 1e-6)
        V.log(f"nuisance {name}: AUROC(test)={V.auroc(y[~tv], p[~tv]):.3f} mean={p.mean():.5f} obs={y.mean():.5f}")
    return out


def theorem2(df):
    res = {}
    for name, col in V.CONCEPTS.items():
        T = SV.tertile(df, col).to_numpy()
        r = {}
        q = [df.Y[T == t].mean() for t in (0, 1)]
        v = [df.V[T == t].mean() for t in (0, 1)]
        sig = [df.S[T == t].mean() for t in (0, 1)]
        RR_Y = q[1] / q[0]
        r["RR_Y"] = RR_Y; r["B_rare_approx"] = v[1] / v[0]
        r["grid"] = []
        for s in S_GRID:
            pL = [q[t] for t in (0, 1)]
            pU = [min(q[t] / s, q[t] + 1 - sig[t]) for t in (0, 1)]
            BL, BU = V.B_interval(v[0], v[1], pL, pU)
            r["grid"].append({"s_min": s, "p_bounds": [pL, pU], "B_interval": [BL, BU],
                              "berkson_endpoint_interval": [RR_Y / BU, RR_Y / BL],
                              "B_rel_width": BU / BL - 1})
        # không giả định s_min, chỉ chặn tỉ lệ bệnh: pₜ ≤ p_max
        r["p_max_grid"] = []
        for pm in (0.01, 0.05, 0.2):
            BL, BU = V.B_interval(v[0], v[1], q, [max(pm, q[0]), max(pm, q[1])])
            r["p_max_grid"].append({"p_max": pm, "B_interval": [BL, BU], "B_rel_width": BU / BL - 1})
        res[name] = r
        V.log(f"T2 {name:22s} B≈{r['B_rare_approx']:.3f}  s=0.5: [{r['grid'][1]['B_interval'][0]:.4f}, "
              f"{r['grid'][1]['B_interval'][1]:.4f}]  p_max=0.2: {np.round(r['p_max_grid'][2]['B_interval'], 3)}")
    return res


def theorem3(df, nu, fold):
    q, sig = nu["Y"], nu["S"]
    res = {"s_grid": S_GRID, "by_s": []}
    mar = q / sig                                 # P(Y=1 | x, S=1) = dự đoán MAR
    for s in S_GRID:
        L, U = V.disease_interval(q, sig, s)
        binds = np.mean(q + 1 - sig < q / s)
        outside = np.mean(mar > U * (1 + 1e-9))
        te = fold == "test"
        res["by_s"].append({
            "s_min": s, "median_U_over_L": float(np.median(U / L)),
            "frac_second_upper_binds": float(binds),
            "frac_MAR_prediction_outside_set": float(outside),
            "frac_MAR_outside_among_verified": float(np.mean(mar[df.S == 1] > U[df.S == 1])),
            "mean_L": float(L.mean()), "mean_U": float(U.mean()),
            "test_mean_L": float(L[te].mean()), "test_mean_U": float(U[te].mean()),
        })
        V.log(f"T3 s_min={s}: MAR prediction outside identified set for {outside:.4f} of lesions "
              f"(verified: {res['by_s'][-1]['frac_MAR_outside_among_verified']:.3f}); mean U={U.mean():.5f}")
    res["frac_sigma_ge"] = {str(s): float(np.mean(sig >= s)) for s in S_GRID}
    return res


def main():
    df = SV.load()
    X, names = V.tabular_X(df)
    fold = V.make_split(df)
    V.log(f"X {X.shape}; split " + str({k: int((fold == k).sum()) for k in ("train", "val", "test")}))
    out = {"n_features": X.shape[1], "theorem2": theorem2(df)}
    nu = nuisance(df, X, fold)
    out["theorem3"] = theorem3(df, nu, fold)
    np.savez(f"{V.RES}/vl_nuisance.npz", q=nu["Y"], sigma=nu["S"], phi=nu["F"], fold=fold,
             isic_id=df["isic_id"].to_numpy())
    V.save_json(out, "vl1_identification.json")


if __name__ == "__main__":
    main()
