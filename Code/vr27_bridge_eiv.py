#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR27 — Hồi quy khoảng cách logit lên log ĝ có sai số trong biến giải thích (errors-in-variables).

log ĝ là biến ước lượng, nên độ dốc OLS của khoảng cách logit f₀ − logit f₂ lên log ĝ bị kéo về 0. Mỗi lần lặp
(100 mỗi họ, bootstrap đồng thời train + test như vr22): huấn luyện M0, M2 trên train đã lấy lại, và hai mô hình ĝ_A,
ĝ_B trên hai nửa RỜI NHAU theo bệnh nhân của train đã lấy lại (cùng lớp MLP, hiệu chỉnh Platt trên nhãn âm của val).
Báo cáo: độ dốc OLS lên log ĝ_A; độ dốc biến công cụ dùng log ĝ_B làm công cụ cho log ĝ_A (nhất quán nếu sai số ước
lượng của hai nửa độc lập), trung bình hai chiều; hệ số tin cậy cov(A, B)/var(A).

Out -> Result/vr27_bridge_eiv.json
"""
from __future__ import annotations

import json
import os

import numpy as np

import vr_common as R
import gpu_setup  # noqa: F401
from vr22_pointwise_bridge import features, platt

B = 100


def main():
    import vl_models as M
    df, fold = R.load_isic()
    Y, S = df.Y.to_numpy(), df.S.to_numpy()
    tr_all, va, te_all = (np.flatnonzero(fold == k) for k in ("train", "val", "test"))
    va2 = va[S[va] == 1]; vneg = va[Y[va] == 0]
    pid = df[R.SV.GROUP].to_numpy()
    trp, tep = np.unique(pid[tr_all]), np.unique(pid[te_all])
    tr_by = {p: tr_all[pid[tr_all] == p] for p in trp}; te_by = {p: te_all[pid[te_all] == p] for p in tep}
    part = os.path.join(R.RES, "vr27_bridge_eiv.json.partial")
    res = json.load(open(part)) if os.path.exists(part) else {"tabular": [], "image": []}
    for kind in ("tabular", "image"):
        X, hid = features(df, kind)
        n = len(df)
        base = dict(sigma=np.ones(n, np.float32), L=np.ones(n, np.float32), U=np.ones(n, np.float32), m=np.zeros(n, np.float32))
        A = dict(x=X, Y=Y.astype(np.float32), F=S.astype(np.float32), S=S.astype(np.float32), **base)
        negm = (Y == 0).astype(np.float32)
        Ag = dict(x=X, Y=S.astype(np.float32), F=negm, S=negm, **base)
        for b in range(len(res[kind]), B):
            rng = np.random.default_rng(90_000 + b)
            bp = rng.choice(trp, len(trp))
            tr = np.concatenate([tr_by[p] for p in bp])
            half = rng.permutation(np.unique(bp)); hA = set(half[: len(half) // 2])
            trA = np.concatenate([tr_by[p] for p in bp if p in hA]); trB = np.concatenate([tr_by[p] for p in bp if p not in hA])
            te = np.concatenate([te_by[p] for p in rng.choice(tep, len(tep))])
            m0 = M.train("erm_y", A, tr, va, hid=hid, seed=b); m2 = M.train("erm_verified", A, tr, va, hid=hid, seed=b)
            gA = M.train("erm_verified", Ag, trA, va, hid=hid, seed=b); gB = M.train("erm_verified", Ag, trB, va, hid=hid, seed=b + 1)
            a0, b0 = platt(R.logit(M.predict(m0, X[va])["pD"]), Y[va])
            a2, b2 = platt(R.logit(M.predict(m2, X[va2])["pD"]), Y[va2])
            ute, inv = np.unique(te, return_inverse=True)
            gap = ((a0 * R.logit(M.predict(m0, X[ute])["pD"]) + b0) - (a2 * R.logit(M.predict(m2, X[ute])["pD"]) + b2))[inv]
            lg = {}
            for nm, mg in (("A", gA), ("B", gB)):
                ag, bg = platt(R.logit(M.predict(mg, X[vneg])["pD"]), S[vneg])
                lg[nm] = np.log(1 / (1 + np.exp(-((ag * R.logit(M.predict(mg, X[ute])["pD"]) + bg)[inv]))))
            cv = lambda u, v: float(np.cov(u, v)[0, 1])
            ols = cv(gap, lg["A"]) / float(np.var(lg["A"], ddof=1))
            ivA = cv(gap, lg["B"]) / cv(lg["A"], lg["B"]); ivB = cv(gap, lg["A"]) / cv(lg["B"], lg["A"])
            rel = cv(lg["A"], lg["B"]) / float(np.sqrt(np.var(lg["A"], ddof=1) * np.var(lg["B"], ddof=1)))
            res[kind].append({"ols": ols, "iv": 0.5 * (ivA + ivB), "reliability": rel})
            json.dump(res, open(part, "w"))
            if b % 10 == 0:
                R.log(f"{kind} rep {b}: OLS {ols:.2f} IV {0.5 * (ivA + ivB):.2f} reliability {rel:.2f}")
    out = {"B": B, "families": {}}
    for kind, rr in res.items():
        g = lambda k: np.array([r[k] for r in rr])
        out["families"][kind] = {k: {"median": float(np.median(g(k))), "ci95": np.percentile(g(k), [2.5, 97.5]).tolist()} for k in ("ols", "iv", "reliability")}
        R.log(f"{kind}: {out['families'][kind]}")
    R.save_json(out, "vr27_bridge_eiv.json")


if __name__ == "__main__":
    main()
