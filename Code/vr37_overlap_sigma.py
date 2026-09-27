#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR37 — Độ nhạy của phân tích overlap theo cách ước lượng xác suất xác minh chung σ.

Ba ước lượng σ̂(x) = P(S=1 | x), đều không dùng test:
  nuisance   gradient boosting out-of-fold trên đặc trưng bảng (phân tích chính);
  mlp_tab    perceptron học S trên đặc trưng bảng, hiệu chỉnh Platt trên val;
  mlp_img    perceptron học S trên đặc trưng ảnh, hiệu chỉnh Platt trên val.
Với mỗi σ̂: ngưỡng = phân vị 1/5/10% của σ̂ trên tổn thương train đã xác minh; báo tỉ lệ test giữ lại, tỉ lệ tổn thương đổi
nhóm so với σ̂ chính (và chỉ số Jaccard của vùng hỗ trợ), và Δ_M0, Δ_M2 của ba concept chính cho hai họ đóng băng trên vùng
hỗ trợ, dùng các lần chạy gốc (trung bình seed 0–2) với bootstrap bệnh nhân test 2,000 lần, cùng một quy trình cho cả ba σ̂.

Out -> Result/vr37_overlap_sigma.json
"""
from __future__ import annotations

import os

import numpy as np

import vr_common as R
import gpu_setup  # noqa: F401
from vr19_primary_bootstrap import features, arrays, platt_fit, PRIMARY

QS = [0.01, 0.05, 0.10]
B = 2000


def main():
    import vl_models as M
    df, fold = R.load_isic()
    S, Y = df.S.to_numpy(), df.Y.to_numpy()
    tr, va = np.flatnonzero(fold == "train"), np.flatnonzero(fold == "val")
    heads = np.load(os.path.join(R.RES, "closing_heads.npz"))
    te = heads["test_idx"]
    sig = {"nuisance": np.load(os.path.join(R.MISS_RES, "vl_nuisance.npz"))["sigma"]}
    for kind, name in (("tabular", "mlp_tab"), ("image", "mlp_img")):
        X, hid = features(df, kind); A = arrays(df, X); A["Y"] = S.astype(np.float32)
        ps = []
        for seed in range(3):
            m = M.train("erm_y", A, tr, va, hid=hid, seed=seed)
            a, b = platt_fit(R.logit(M.predict(m, X[va])["pD"]), S[va])
            ps.append(1 / (1 + np.exp(-(a * R.logit(M.predict(m, X)["pD"]) + b))))
        sig[name] = np.mean(ps, 0)
        R.log(f"{name}: fitted")
    trS = tr[S[tr] == 1]
    pid = df[R.SV.GROUP].to_numpy()[te]; pats = np.unique(pid); pidx = {p: i for i, p in enumerate(pats)}
    pc = np.array([pidx[p] for p in pid])
    rng = np.random.default_rng(37)
    W = np.zeros((B, len(pats)))
    for b in range(B):
        np.add.at(W[b], rng.integers(0, len(pats), len(pats)), 1.0)
    Tt = {c: R.SV.tertile(df, R.CONCEPTS[c]).fillna(-1).to_numpy()[te] for c in PRIMARY}
    L = {k: (np.mean([heads[f"{k}_M0_s{s}"] for s in range(3)], 0), np.mean([heads[f"{k}_M2_s{s}"] for s in range(3)], 0)) for k in ("tabular", "image")}
    out = {"estimators": list(sig), "thresholds": {}, "membership": {}, "contrasts": {}}
    masks = {}
    for nm, sg in sig.items():
        for q in QS:
            thr = float(np.quantile(sg[trS], q)); masks[(nm, q)] = sg[te] >= thr
            out["thresholds"][f"{nm}|{q}"] = {"threshold": thr, "retained": float(masks[(nm, q)].mean())}
    for nm in sig:
        for q in QS:
            a, b = masks[("nuisance", q)], masks[(nm, q)]
            out["membership"][f"{nm}|{q}"] = {"switched": float(np.mean(a != b)), "jaccard": float((a & b).sum() / max((a | b).sum(), 1))}

    def contrast(v, T, m, w):
        wl = w[pc] * m
        return (wl * v * (T == 1)).sum() / (wl * (T == 1)).sum() - (wl * v * (T == 0)).sum() / (wl * (T == 0)).sum()

    for kind in ("tabular", "image"):
        l0, l2 = L[kind]
        for nm in sig:
            for q in QS:
                m = masks[(nm, q)].astype(float)
                for c in PRIMARY:
                    T = Tt[c]
                    d0 = contrast(l0, T, m, np.ones(len(pats))); d2 = contrast(l2, T, m, np.ones(len(pats)))
                    b0 = np.array([contrast(l0, T, m, W[b]) for b in range(B)]); b2 = np.array([contrast(l2, T, m, W[b]) for b in range(B)])
                    c0, c2 = np.percentile(b0, [2.5, 97.5]), np.percentile(b2, [2.5, 97.5])
                    out["contrasts"][f"{kind}|{nm}|{q}|{c}"] = {"d0": float(d0), "d2": float(d2), "d0_ci": c0.tolist(), "d2_ci": c2.tolist(),
                                                                 "robust_95": bool(c0[0] > 0 and c2[1] < 0), "m2_below": bool(c2[1] < 0), "m0_reaches_zero": bool(c0[0] <= 0)}
                R.log(f"{kind} {nm} q={q}: retained {masks[(nm, q)].mean():.2f} color d0 {out['contrasts'][f'{kind}|{nm}|{q}|color_variegation']['d0']:+.2f} "
                      f"{np.round(out['contrasts'][f'{kind}|{nm}|{q}|color_variegation']['d0_ci'], 2)} d2 {out['contrasts'][f'{kind}|{nm}|{q}|color_variegation']['d2']:+.2f}")
    R.save_json(out, "vr37_overlap_sigma.json")


if __name__ == "__main__":
    main()
