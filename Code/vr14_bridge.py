#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR14 — Cầu nối lý thuyết trực tiếp tới estimand thực nghiệm Δ (Proposition 4).

Mệnh đề (chính xác, phi tham số). Với p(x) = P(D=1|x), π¹(x), π⁰(x) và Y = D·S, predictor Bayes-tối ưu của hai
regime là f*_M0(x) = P(Y=1|x) = p π¹ và f*_M2(x) = P(Y=1|x, S=1) = p π¹ / (p π¹ + (1−p) π⁰). Khi đó
    logit f*_M0(x) − logit f*_M2(x) = log[(1−p)π⁰ / (1 − pπ¹)] = log P(S=1 | x, Y=0) =: log g(x).
g là xác suất được xác minh của một tổn thương có nhãn ghi nhận âm; nó QUAN SÁT ĐƯỢC hoàn toàn (không cần D, không
cần bệnh hiếm, không cần mô hình tuyến tính). Lấy kỳ vọng trên tertile của concept k trong quần thể đánh giá:
    Δ*_M0,k − Δ*_M2,k = E[log g(X) | T_k=1] − E[log g(X) | T_k=0].
Một hằng số cộng vào logit (ví dụ do lấy mẫu lại lớp dương) triệt tiêu trong Δ.

Kiểm tra thực nghiệm: ước lượng g trên các tổn thương nhãn âm của train+val (không dùng test), trên CÙNG loại x
mà learner dùng (tabular: GBM trên 47 đặc trưng; ảnh: GBM trên 128 thành phần chính của embedding), rồi so
khoảng cách dự đoán với khoảng cách học được Δ_M0 − Δ_M2 của vr9 (bốn họ; linear probe và fine-tune dùng ĝ ảnh).

Out -> Result/vr14_bridge.json
"""
from __future__ import annotations

import json
import os

import numpy as np

import vr_common as R

B = 1000


def fit_g(X, df, fold):
    from sklearn.ensemble import HistGradientBoostingClassifier
    neg = df.Y.to_numpy() == 0
    tv = (fold != "test") & neg
    m = HistGradientBoostingClassifier(max_depth=4, max_iter=400, learning_rate=0.05, l2_regularization=1.0,
                                       random_state=R.SEED).fit(X[tv], df.S.to_numpy()[tv])
    g = m.predict_proba(X)[:, 1]
    te = (fold == "test") & neg
    return np.clip(g, 1e-6, 1 - 1e-6), R.auroc(df.S.to_numpy()[te], g[te])


def main():
    df, fold = R.load_isic()
    Xt, _ = R.VL.tabular_X(df)
    z = np.load(os.path.join(R.EMB, "isic2024.npz"))
    E = z["emb"].astype(np.float32); E = (E - E.mean(0)) / (E.std(0) + 1e-6)
    from sklearn.decomposition import PCA
    tr = fold == "train"
    Xi = PCA(128, random_state=R.SEED).fit(E[tr][::4]).transform(E).astype(np.float32)
    g_tab, auc_t = fit_g(Xt, df, fold)
    g_img, auc_i = fit_g(Xi, df, fold)
    R.log(f"g(x) = P(S=1|x,Y=0): AUROC on test negatives tabular {auc_t:.3f}, image {auc_i:.3f}")
    v9 = json.load(open(os.path.join(R.RES, "vr9_closing.json")))["families"]
    heads = np.load(os.path.join(R.RES, "closing_heads.npz"))
    pz = np.load(os.path.join(R.RES, "finetune", "ft_M0_s0.npz"))
    idx = {"head_tabular": (heads["test_idx"], g_tab), "head_image": (heads["test_idx"], g_img),
           "lp": (pz["probe"][pz["is_uniform"]], g_img), "ft": (pz["probe"][pz["is_uniform"]], g_img)}
    pid_all = df[R.SV.GROUP].to_numpy()
    rng = np.random.default_rng(R.SEED)
    out = {"g_auroc_test_negatives": {"tabular": auc_t, "image": auc_i}, "families": {}}
    for fam, (ix, g) in idx.items():
        lg = np.log(g[ix]); pid = pid_all[ix]
        up, inv = np.unique(pid, return_inverse=True)
        pred, pred_ci, obs, obs_ci = [], [], [], []
        W = np.array([np.bincount(rng.integers(0, len(up), len(up)), minlength=len(up)) for _ in range(B)], float)
        for c, col in R.CONCEPTS.items():
            T = R.SV.tertile(df, col).fillna(-1).to_numpy()[ix]
            s1 = np.bincount(inv, weights=lg * (T == 1), minlength=len(up)); n1 = np.bincount(inv, weights=(T == 1).astype(float), minlength=len(up))
            s0 = np.bincount(inv, weights=lg * (T == 0), minlength=len(up)); n0 = np.bincount(inv, weights=(T == 0).astype(float), minlength=len(up))
            p = s1.sum() / n1.sum() - s0.sum() / n0.sum()
            pb = (W @ s1) / (W @ n1) - (W @ s0) / (W @ n0)
            t = v9[fam]["concepts"][c]["tertile"]
            pred.append(p); pred_ci.append(np.percentile(pb, [2.5, 97.5]).tolist())
            obs.append(t["diff"]); obs_ci.append(t["diff_ci"])
        pred, obs = np.array(pred), np.array(obs)
        cal = np.polyfit(pred, obs, 1)
        out["families"][fam] = {"concepts": list(R.CONCEPTS), "predicted_gap": pred.tolist(), "predicted_gap_ci": pred_ci,
                                "observed_gap": obs.tolist(), "observed_gap_ci": obs_ci,
                                "sign_agreement": int(np.sum(np.sign(pred) == np.sign(obs))),
                                "calibration_slope": float(cal[0]), "calibration_intercept": float(cal[1]),
                                "mean_abs_diff": float(np.mean(np.abs(obs - pred))),
                                "observed_within_predicted_ci": int(sum(ci[0] <= o <= ci[1] for o, ci in zip(obs, pred_ci)))}
        R.log(f"{fam:13s} predicted {np.round(pred, 2)} observed {np.round(obs, 2)} slope {cal[0]:.2f} "
              f"int {cal[1]:+.2f} sign {out['families'][fam]['sign_agreement']}/5")
    R.save_json(out, "vr14_bridge.json")


if __name__ == "__main__":
    main()
