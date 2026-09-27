#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR21 — Can thiệp có kiểm soát trên biến chọn mẫu huấn luyện R (bản đầy đủ).

Theo Proposition 1 dạng tổng quát: nếu r₁ = P(R=1 | Y=1, x̃) là hằng số và r₀(x) = P(R=1 | Y=0, x) chọn trong các
tổn thương NHÃN ÂM GHI NHẬN, thì với learner quan sát x̃,
    logit P(Y=1 | x̃, R=1) = logit P(Y=1 | x̃) − log E[r₀(X) | x̃, Y=0] + hằng số.
(a) Liều η: r₀ ∝ exp(η z_k). Với learner tabular, z_k nằm trong input nên độ dịch là −η z_k và dự đoán chính xác
    Δ_j(η) = Δ_j(0) − η μ_kj. Với learner ảnh, z_k không đo được từ x̃; dùng xấp xỉ chuẩn z_k | x̃, Y=0 ~ N(m(x̃), τ²)
    (τ² hằng số) cho log E[exp(η z_k) | x̃, Y=0] = η m(x̃) + hằng số, nên độ dốc dự đoán là −(E[m | T_j=1] − E[m | T_j=0]),
    với m(x̃) ước lượng bằng ridge trên embedding, khớp trên nhãn âm train.
(b) Các nhánh cùng cỡ: ngẫu nhiên, gắn cờ chưa xác minh, đã xác minh. Với r₀_arm(x) = P(arm | x, Y=0), dự đoán
    Δ_arm − Δ_random = −(E[log ĝ_arm | T=1] − E[log ĝ_arm | T=0]), ĝ_arm ước lượng bằng cùng lớp MLP trên input learner.
Mọi learner được hiệu chỉnh Platt trên tập val được chọn theo CÙNG quy tắc với tập huấn luyện của nó.
Báo cáo thêm: ESS Kish của trọng số chọn trên tập nhãn âm và tỉ lệ nhãn âm được chọn rơi vào tertile dưới/trên theo η;
độ dốc chỉ khớp trên η ∈ [−1, 1], vùng còn đủ hỗ trợ, với khoảng 2.5–97.5 phân vị qua lần lặp.

Out -> Result/vr21_dose_calibrated.json
"""
from __future__ import annotations

import json
import os

import numpy as np

import vr_common as R
import gpu_setup  # noqa: F401  (ép GPU, giới hạn luồng)

ETAS = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
FIT_ETAS = [-1.0, -0.5, 0.0, 0.5, 1.0]
N_REP, N_MAL, N_BEN = 10, 178, 320
TARGETS = ["color_variegation", "size", "lesion_skin_contrast"]


def features(df, kind):
    if kind == "tabular":
        X, _ = R.VL.tabular_X(df)
        return X.astype(np.float32), 128
    z = np.load(os.path.join(R.EMB, "isic2024.npz")); E = z["emb"].astype(np.float32)
    return ((E - E.mean(0)) / (E.std(0) + 1e-6)).astype(np.float32), 256


def gumbel_topk(rng, pool, w, n):
    key = np.log(w) - np.log(-np.log(rng.random(len(pool))))
    return pool[np.argsort(-key)[:n]]


def platt(lg_val, y_val, lg):
    from sklearn.linear_model import LogisticRegression
    if len(np.unique(y_val)) < 2:
        return lg
    m = LogisticRegression(C=1e6, max_iter=2000).fit(lg_val[:, None], y_val)
    return m.coef_[0, 0] * lg + m.intercept_[0]


def main():
    import vl_models as M
    from sklearn.linear_model import Ridge
    df, fold = R.load_isic()
    Y, S, F = df.Y.to_numpy(), df.S.to_numpy(), df.F.to_numpy()
    Zs = {c: ((df[col] - df[col].mean()) / df[col].std()).fillna(0.0).to_numpy() for c, col in R.CONCEPTS.items()}
    te = np.flatnonzero(fold == "test")
    Tc = {c: R.SV.tertile(df, col).fillna(-1).to_numpy()[te] for c, col in R.CONCEPTS.items()}
    D = lambda v, j: float(v[Tc[j] == 1].mean() - v[Tc[j] == 0].mean())
    pools = {p: {"mal": np.flatnonzero((fold == p) & (Y == 1)), "neg": np.flatnonzero((fold == p) & (Y == 0)),
                 "flag": np.flatnonzero((fold == p) & (Y == 0) & (F == 1) & (S == 0)), "ver": np.flatnonzero((fold == p) & (Y == 0) & (S == 1))}
             for p in ("train", "val")}
    nva_m = int(round(0.8 * len(pools["val"]["mal"]))); nva_b = 111
    tr_all, va_all = np.flatnonzero(fold == "train"), np.flatnonzero(fold == "val")
    Tall = {c: R.SV.tertile(df, col).fillna(-1).to_numpy() for c, col in R.CONCEPTS.items()}
    out = {"etas": ETAS, "fit_etas": FIT_ETAS, "support": {}, "families": {}}
    # (0) hỗ trợ theo liều: ESS Kish và thành phần tertile của tập nhãn âm được chọn (xấp xỉ theo trọng số)
    neg = pools["train"]["neg"]
    for k in TARGETS:
        rows = []
        for eta in ETAS:
            w = np.exp(eta * Zs[k][neg]); p = w / w.sum()
            rows.append({"eta": eta, "ess": float(1 / np.sum(p ** 2)), "ess_frac": float(1 / np.sum(p ** 2) / len(neg)),
                         "share_lower": float(p[Tall[k][neg] == 0].sum()), "share_upper": float(p[Tall[k][neg] == 1].sum())})
        out["support"][k] = rows
    for kind in ("tabular", "image"):
        X, hid = features(df, kind)
        fam = {"dose": {}, "arms": {}}
        # m(x̃) = E[z_k | x̃, Y=0] cho dự đoán theo input của learner (tabular: chính z_k)
        mhat = {}
        for k in R.CONCEPTS:
            if kind == "tabular":
                mhat[k] = Zs[k]
            else:
                ntr = pools["train"]["neg"]; sub = np.random.default_rng(0).choice(ntr, 60000, replace=False)
                mhat[k] = Ridge(alpha=10.0).fit(X[sub], Zs[k][sub]).predict(X)
        base = lambda sel_tr, sel_va, seed: _fit(M, df, X, hid, sel_tr, sel_va, tr_all, va_all, te, seed)
        for k in TARGETS:
            reps = []
            for r in range(N_REP):
                rng = np.random.default_rng(3000 * r + TARGETS.index(k))
                mtr = rng.choice(pools["train"]["mal"], N_MAL, replace=False); mva = rng.choice(pools["val"]["mal"], nva_m, replace=False)
                rep = {}
                for eta in ETAS:
                    btr = gumbel_topk(rng, pools["train"]["neg"], np.exp(eta * Zs[k][pools["train"]["neg"]]), N_BEN)
                    bva = gumbel_topk(rng, pools["val"]["neg"], np.exp(eta * Zs[k][pools["val"]["neg"]]), nva_b)
                    raw, cal = base(np.concatenate([mtr, btr]), np.concatenate([mva, bva]), r)
                    rep[str(eta)] = {"raw": {j: D(raw, j) for j in R.CONCEPTS}, "cal": {j: D(cal, j) for j in R.CONCEPTS}}
                reps.append(rep)
                R.log(f"{kind}|{k} rep {r}: cal Δ by η " + " ".join(f"{e:+.1f}:{rep[str(e)]['cal'][k]:+.2f}" for e in ETAS))
            e = np.array(ETAS); fe = np.array(FIT_ETAS); idx = [ETAS.index(x) for x in FIT_ETAS]
            res = {}
            for scale in ("raw", "cal"):
                Mk = np.array([[rp[str(x)][scale][k] for x in ETAS] for rp in reps])
                slopes = np.array([np.polyfit(fe, row[idx], 1)[0] for row in Mk])
                mean = Mk.mean(0)
                cross = None
                for i in range(len(e) - 1):
                    if mean[i] * mean[i + 1] <= 0 and mean[i] != mean[i + 1]:
                        cross = float(e[i] - mean[i] * (e[i + 1] - e[i]) / (mean[i + 1] - mean[i])); break
                spill = {j: float(np.polyfit(fe, np.array([[rp[str(x)][scale][j] for x in FIT_ETAS] for rp in reps]).mean(0), 1)[0]) for j in R.CONCEPTS}
                res[scale] = {"mean_by_eta": mean.tolist(), "q025": np.percentile(Mk, 2.5, 0).tolist(), "q975": np.percentile(Mk, 97.5, 0).tolist(),
                              "slope_mean": float(slopes.mean()), "slope_range95": np.percentile(slopes, [2.5, 97.5]).tolist(),
                              "crossing": cross, "spillover_slopes": spill}
            pred_z = {j: -D(Zs[k][te], j) for j in R.CONCEPTS}           # −μ_kj (z_k observed)
            pred_in = {j: -D(mhat[k][te], j) for j in R.CONCEPTS}        # input-aware
            d0 = res["cal"]["mean_by_eta"][ETAS.index(0.0)]
            res["predicted_slope_zk"] = pred_z[k]; res["predicted_slope_input"] = pred_in[k]
            res["crossing_predicted_input"] = float(d0 / -pred_in[k]) if pred_in[k] != 0 else None
            res["spillover_sign_agreement"] = int(sum(np.sign(res["cal"]["spillover_slopes"][j]) == np.sign(pred_in[j]) for j in R.CONCEPTS))
            res["predicted_spillover_input"] = pred_in
            fam["dose"][k] = res
            R.log(f"{kind}|{k}: cal slope {res['cal']['slope_mean']:+.2f} {np.round(res['cal']['slope_range95'], 2)} "
                  f"pred(input) {pred_in[k]:+.2f} pred(z) {pred_z[k]:+.2f}; cross {res['cal']['crossing']} vs {res['crossing_predicted_input']}")
        # (b) các nhánh cùng cỡ với dự đoán định lượng
        gh = {}
        for arm, pool in (("flag", "flag"), ("ver", "ver")):
            lab = np.zeros(len(df), np.float32); lab[np.concatenate([pools["train"][pool], pools["val"][pool]])] = 1
            negmask = (Y == 0).astype(np.float32)
            A = dict(x=X, Y=lab, F=negmask, S=negmask, sigma=np.ones(len(df), np.float32), L=np.ones(len(df), np.float32),
                     U=np.ones(len(df), np.float32), m=np.zeros(len(df), np.float32))
            mg = M.train("erm_verified", A, tr_all, va_all, hid=hid, seed=0)
            gh[arm] = np.log(np.clip(M.predict(mg, X[te])["pD"], 1e-7, 1))
        arms = {a: {j: [] for j in R.CONCEPTS} for a in ("random", "flag", "ver")}
        for r in range(N_REP):
            rng = np.random.default_rng(9000 + r)
            mtr = rng.choice(pools["train"]["mal"], N_MAL, replace=False); mva = rng.choice(pools["val"]["mal"], nva_m, replace=False)
            for a, pool in (("random", "neg"), ("flag", "flag"), ("ver", "ver")):
                btr = rng.choice(pools["train"][pool], N_BEN, replace=False); bva = rng.choice(pools["val"][pool], min(nva_b, len(pools["val"][pool])), replace=False)
                _, cal = base(np.concatenate([mtr, btr]), np.concatenate([mva, bva]), r)
                for j in R.CONCEPTS:
                    arms[a][j].append(D(cal, j))
        for a in ("flag", "ver"):
            fam["arms"][a] = {j: {"observed_shift": float(np.mean(arms[a][j]) - np.mean(arms["random"][j])),
                                  "predicted_shift": -D(gh[a], j)} for j in R.CONCEPTS}
            ob = np.array([fam["arms"][a][j]["observed_shift"] for j in R.CONCEPTS]); pr = np.array([fam["arms"][a][j]["predicted_shift"] for j in R.CONCEPTS])
            fam["arms"][a]["summary"] = {"sign_agreement": int(np.sum(np.sign(ob) == np.sign(pr))), "calibration_slope": float(np.polyfit(pr, ob, 1)[0]),
                                         "mean_abs_diff": float(np.mean(np.abs(ob - pr)))}
            R.log(f"{kind} arm {a}: observed {np.round(ob, 2)} predicted {np.round(pr, 2)} {fam['arms'][a]['summary']}")
        fam["arms"]["random_mean"] = {j: float(np.mean(arms["random"][j])) for j in R.CONCEPTS}
        out["families"][kind] = fam
        R.save_json(out, "vr21_dose_calibrated.json")


def _fit(M, df, X, hid, sel_tr, sel_va, tr_all, va_all, te, seed):
    sel = np.zeros(len(df), np.float32); sel[np.concatenate([sel_tr, sel_va])] = 1
    A = dict(x=X, Y=df.Y.to_numpy(np.float32), F=sel, S=sel, sigma=np.ones(len(df), np.float32), L=np.ones(len(df), np.float32),
             U=np.ones(len(df), np.float32), m=np.zeros(len(df), np.float32))
    mdl = M.train("erm_verified", A, tr_all, va_all, hid=hid, seed=seed)
    lg = R.logit(M.predict(mdl, X[te])["pD"])
    lv = R.logit(M.predict(mdl, X[sel_va])["pD"])
    return lg, platt(lv, df.Y.to_numpy()[sel_va], lg)


if __name__ == "__main__":
    main()
