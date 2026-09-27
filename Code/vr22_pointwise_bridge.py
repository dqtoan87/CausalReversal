#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR22 — Kiểm Proposition 1 ở cấp từng tổn thương, với bootstrap đồng thời train + test (200 lần, hai họ chính).

Proposition 1 là đồng nhất thức theo từng điểm: logit f₀(x̃) − logit f₂(x̃) = log g(x̃). Mỗi lần lặp: lấy lại bệnh nhân
train; huấn luyện M0, M2 và ĝ (cùng lớp MLP; ĝ học S trên nhãn âm); hiệu chỉnh Platt cả ba trên val của chính chúng
(M0: val toàn bộ; M2: val đã xác minh; ĝ: val nhãn âm) vì đồng nhất thức nói về xác suất đã hiệu chỉnh; lấy lại bệnh
nhân test; hồi quy khoảng cách logit đã hiệu chỉnh lên log ĝ đã hiệu chỉnh theo từng tổn thương (hệ số kỳ vọng 1,
chặn tự do) và so khoảng cách Δ dự đoán với quan sát cho năm concept.

Out -> Result/vr22_pointwise_bridge.json (+ .partial)
"""
from __future__ import annotations

import json
import os

import numpy as np

import vr_common as R
import gpu_setup  # noqa: F401  (ép GPU, giới hạn luồng)

B = 200


def features(df, kind):
    if kind == "tabular":
        X, _ = R.VL.tabular_X(df)
        return X.astype(np.float32), 128
    z = np.load(os.path.join(R.EMB, "isic2024.npz")); E = z["emb"].astype(np.float32)
    return ((E - E.mean(0)) / (E.std(0) + 1e-6)).astype(np.float32), 256


def platt(lv, yv):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1e6, max_iter=2000).fit(lv[:, None], yv)
    return m.coef_[0, 0], m.intercept_[0]


def main():
    import vl_models as M
    df, fold = R.load_isic()
    Y, S = df.Y.to_numpy(), df.S.to_numpy()
    tr_all, va, te_all = (np.flatnonzero(fold == k) for k in ("train", "val", "test"))
    va2 = va[S[va] == 1]; vneg = va[Y[va] == 0]
    pid = df[R.SV.GROUP].to_numpy()
    trp, tep = np.unique(pid[tr_all]), np.unique(pid[te_all])
    tr_by = {p: tr_all[pid[tr_all] == p] for p in trp}; te_by = {p: te_all[pid[te_all] == p] for p in tep}
    Tall = {c: R.SV.tertile(df, col).fillna(-1).to_numpy() for c, col in R.CONCEPTS.items()}
    part = os.path.join(R.RES, "vr22_pointwise_bridge.json.partial")
    res = json.load(open(part)) if os.path.exists(part) else {"tabular": [], "image": []}
    for kind in ("tabular", "image"):
        X, hid = features(df, kind)
        n = len(df)
        base = dict(sigma=np.ones(n, np.float32), L=np.ones(n, np.float32), U=np.ones(n, np.float32), m=np.zeros(n, np.float32))
        A = dict(x=X, Y=Y.astype(np.float32), F=S.astype(np.float32), S=S.astype(np.float32), **base)
        negm = (Y == 0).astype(np.float32)
        Ag = dict(x=X, Y=S.astype(np.float32), F=negm, S=negm, **base)
        for b in range(len(res[kind]), B):
            rng = np.random.default_rng(70_000 + b)
            tr = np.concatenate([tr_by[p] for p in rng.choice(trp, len(trp))])
            te = np.concatenate([te_by[p] for p in rng.choice(tep, len(tep))])
            m0 = M.train("erm_y", A, tr, va, hid=hid, seed=b); m2 = M.train("erm_verified", A, tr, va, hid=hid, seed=b)
            mg = M.train("erm_verified", Ag, tr, va, hid=hid, seed=b)
            a0, b0 = platt(R.logit(M.predict(m0, X[va])["pD"]), Y[va])
            a2, b2 = platt(R.logit(M.predict(m2, X[va2])["pD"]), Y[va2])
            ag, bg = platt(R.logit(M.predict(mg, X[vneg])["pD"]), S[vneg])
            ute, inv = np.unique(te, return_inverse=True)
            c0 = (a0 * R.logit(M.predict(m0, X[ute])["pD"]) + b0)[inv]
            c2 = (a2 * R.logit(M.predict(m2, X[ute])["pD"]) + b2)[inv]
            lg = np.log(1 / (1 + np.exp(-((ag * R.logit(M.predict(mg, X[ute])["pD"]) + bg)[inv]))))
            gap = c0 - c2
            slope, icpt = np.polyfit(lg, gap, 1)
            r2 = float(np.corrcoef(lg, gap)[0, 1] ** 2)
            conc = {}
            for c, Tf in Tall.items():
                T = Tf[te]
                conc[c] = [float(gap[T == 1].mean() - gap[T == 0].mean()), float(lg[T == 1].mean() - lg[T == 0].mean())]
            res[kind].append({"slope": float(slope), "intercept": float(icpt), "r2": r2, "concepts": conc})
            json.dump(res, open(part, "w"))
            if b % 20 == 0:
                R.log(f"{kind} rep {b}: pointwise slope {slope:.2f} R2 {r2:.2f}; color observed {conc['color_variegation'][0]:+.2f} predicted {conc['color_variegation'][1]:+.2f}")
    out = {"B": B, "families": {}}
    for kind, rr in res.items():
        sl = np.array([r["slope"] for r in rr]); r2 = np.array([r["r2"] for r in rr])
        cal = []
        for r in rr:
            ob = np.array([r["concepts"][c][0] for c in R.CONCEPTS]); pr = np.array([r["concepts"][c][1] for c in R.CONCEPTS])
            cal.append(np.polyfit(pr, ob, 1)[0])
        cal = np.array(cal)
        fam = {"pointwise_slope": float(np.median(sl)), "pointwise_slope_ci": np.percentile(sl, [2.5, 97.5]).tolist(),
               "pointwise_r2": float(np.median(r2)), "concept_calibration_slope": float(np.median(cal)),
               "concept_calibration_slope_ci": np.percentile(cal, [2.5, 97.5]).tolist(), "concepts": {}}
        for c in R.CONCEPTS:
            a = np.array([r["concepts"][c] for r in rr])
            fam["concepts"][c] = {"observed": float(np.median(a[:, 0])), "observed_ci": np.percentile(a[:, 0], [2.5, 97.5]).tolist(),
                                  "predicted": float(np.median(a[:, 1])), "predicted_ci": np.percentile(a[:, 1], [2.5, 97.5]).tolist(),
                                  "sign_agreement_frac": float(np.mean(np.sign(a[:, 0]) == np.sign(a[:, 1])))}
        out["families"][kind] = fam
        R.log(f"{kind}: pointwise slope {fam['pointwise_slope']:.2f} {np.round(fam['pointwise_slope_ci'], 2)}; "
              f"concept calibration {fam['concept_calibration_slope']:.2f} {np.round(fam['concept_calibration_slope_ci'], 2)}")
    R.save_json(out, "vr22_pointwise_bridge.json")


if __name__ == "__main__":
    main()
