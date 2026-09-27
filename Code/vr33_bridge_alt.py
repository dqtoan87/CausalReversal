#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR33 — Cầu nối Proposition 1 cấp tổn thương với hai lớp mô hình cho ĝ và trong vùng hỗ trợ.

Mỗi lần lặp (100 mỗi họ, bootstrap đồng thời train + test như vr22): M0, M2 (MLP) hiệu chỉnh Platt; ĝ theo hai cách:
MLP (như vr22) và gradient boosting (ảnh: 64 thành phần chính), cả hai học S trên nhãn âm train và hiệu chỉnh Platt trên
nhãn âm val. Báo cáo: độ dốc bình phương tối thiểu của khoảng cách logit lên log ĝ và R², trên toàn test và trên vùng hỗ
trợ (σ̂ ≥ phân vị 1% của tổn thương train đã xác minh); với concept: tỉ lệ dấu trùng và độ dốc hiệu chỉnh.

Out -> Result/vr33_bridge_alt.json
"""
from __future__ import annotations

import json
import os

import numpy as np

import vr_common as R
import gpu_setup  # noqa: F401
from vr22_pointwise_bridge import features, platt
from vr30_q_bootstrap import gbm_input

B = 100


def main():
    import vl_models as M
    from sklearn.ensemble import HistGradientBoostingClassifier
    df, fold = R.load_isic()
    Y, S = df.Y.to_numpy(), df.S.to_numpy()
    tr_all, va, te_all = (np.flatnonzero(fold == k) for k in ("train", "val", "test"))
    va2 = va[S[va] == 1]; vneg = va[Y[va] == 0]
    sig = np.load(os.path.join(R.MISS_RES, "vl_nuisance.npz"))["sigma"]
    sup = sig >= np.quantile(sig[tr_all[S[tr_all] == 1]], 0.01)
    pid = df[R.SV.GROUP].to_numpy()
    trp, tep = np.unique(pid[tr_all]), np.unique(pid[te_all])
    tr_by = {p: tr_all[pid[tr_all] == p] for p in trp}; te_by = {p: te_all[pid[te_all] == p] for p in tep}
    Tall = {c: R.SV.tertile(df, col).fillna(-1).to_numpy() for c, col in R.CONCEPTS.items()}
    part = os.path.join(R.RES, "vr33_bridge_alt.json.partial")
    res = json.load(open(part)) if os.path.exists(part) else {"tabular": [], "image": []}
    for kind in ("tabular", "image"):
        X, hid = features(df, kind)
        Xg = gbm_input(df, X, kind, tr_all)
        n = len(df)
        base = dict(sigma=np.ones(n, np.float32), L=np.ones(n, np.float32), U=np.ones(n, np.float32), m=np.zeros(n, np.float32))
        A = dict(x=X, Y=Y.astype(np.float32), F=S.astype(np.float32), S=S.astype(np.float32), **base)
        negm = (Y == 0).astype(np.float32)
        Ag = dict(x=X, Y=S.astype(np.float32), F=negm, S=negm, **base)
        for b in range(len(res[kind]), B):
            rng = np.random.default_rng(130_000 + b)
            tr = np.concatenate([tr_by[p] for p in rng.choice(trp, len(trp))])
            te = np.concatenate([te_by[p] for p in rng.choice(tep, len(tep))])
            m0 = M.train("erm_y", A, tr, va, hid=hid, seed=b); m2 = M.train("erm_verified", A, tr, va, hid=hid, seed=b)
            mg = M.train("erm_verified", Ag, tr, va, hid=hid, seed=b)
            trn = tr[Y[tr] == 0]
            gb = HistGradientBoostingClassifier(max_depth=4, max_iter=400, learning_rate=0.05, random_state=b).fit(Xg[trn], S[trn])
            a0, b0 = platt(R.logit(M.predict(m0, X[va])["pD"]), Y[va]); a2, b2 = platt(R.logit(M.predict(m2, X[va2])["pD"]), Y[va2])
            ag, bg = platt(R.logit(M.predict(mg, X[vneg])["pD"]), S[vneg])
            ab, bb = platt(R.logit(np.clip(gb.predict_proba(Xg[vneg])[:, 1], 1e-7, 1 - 1e-7)), S[vneg])
            ute, inv = np.unique(te, return_inverse=True)
            gap = ((a0 * R.logit(M.predict(m0, X[ute])["pD"]) + b0) - (a2 * R.logit(M.predict(m2, X[ute])["pD"]) + b2))[inv]
            lg = {"mlp": np.log(1 / (1 + np.exp(-((ag * R.logit(M.predict(mg, X[ute])["pD"]) + bg)[inv])))),
                  "gbm": np.log(1 / (1 + np.exp(-((ab * R.logit(np.clip(gb.predict_proba(Xg[ute])[:, 1], 1e-7, 1 - 1e-7)) + bb)[inv]))))}
            rec = {}
            for nm, l in lg.items():
                for reg, msk in (("all", np.ones(len(te), bool)), ("support", sup[te])):
                    sl = float(np.polyfit(l[msk], gap[msk], 1)[0]); r2 = float(np.corrcoef(l[msk], gap[msk])[0, 1] ** 2)
                    conc = {c: [float(gap[msk & (Tf[te] == 1)].mean() - gap[msk & (Tf[te] == 0)].mean()),
                                float(l[msk & (Tf[te] == 1)].mean() - l[msk & (Tf[te] == 0)].mean())] for c, Tf in Tall.items()}
                    rec[f"{nm}|{reg}"] = {"slope": sl, "r2": r2, "concepts": conc}
            res[kind].append(rec)
            json.dump(res, open(part, "w"))
            if b % 10 == 0:
                R.log(f"{kind} rep {b}: " + " ".join(f"{k} {v['slope']:.2f}" for k, v in rec.items()))
    out = {"B": B, "families": {}}
    for kind, rr in res.items():
        fo = {}
        for key in rr[0]:
            sl = np.array([r[key]["slope"] for r in rr]); r2 = np.array([r[key]["r2"] for r in rr])
            ag = [float(np.mean([np.sign(r[key]["concepts"][c][0]) == np.sign(r[key]["concepts"][c][1]) for r in rr])) for c in R.CONCEPTS]
            cal = np.array([np.polyfit([r[key]["concepts"][c][1] for c in R.CONCEPTS], [r[key]["concepts"][c][0] for c in R.CONCEPTS], 1)[0] for r in rr])
            fo[key] = {"slope": float(np.median(sl)), "slope_ci": np.percentile(sl, [2.5, 97.5]).tolist(), "r2": float(np.median(r2)),
                       "sign_agreement_min": float(min(ag)), "n_concepts_975": int(sum(a >= 0.975 for a in ag)),
                       "concept_calibration": float(np.median(cal)), "concept_calibration_ci": np.percentile(cal, [2.5, 97.5]).tolist()}
            R.log(f"{kind} {key}: slope {fo[key]['slope']:.2f} {np.round(fo[key]['slope_ci'], 2)} R2 {fo[key]['r2']:.2f} "
                  f"signs≥97.5% {fo[key]['n_concepts_975']}/5 calib {fo[key]['concept_calibration']:.2f}")
        out["families"][kind] = fo
    R.save_json(out, "vr33_bridge_alt.json")


if __name__ == "__main__":
    main()
