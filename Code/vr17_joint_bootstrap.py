#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR17 — Hai phân tích trên hai họ learner đặc trưng đóng băng (tabular, ảnh), dùng GPU nếu có.

(1) Kiểm tra Proposition 4 trên xác suất đã hiệu chỉnh. Đồng nhất thức logit f*_M0 − logit f*_M2 = log g(x),
    g(x) = P(S=1 | x, Y=0), phát biểu cho predictor Bayes-tối ưu, tức xác suất đã hiệu chỉnh. Learner hữu hạn,
    nhất là M2 học từ 622 tổn thương, có thể quá tự tin. Ở đây: huấn luyện M0, M2 (3 seed, như vr9), hiệu chỉnh
    Platt logit của từng learner trên tập val của chính regime (M0: val toàn bộ; M2: val đã xác minh), và ước lượng
    g bằng CÙNG lớp MLP trên tổn thương nhãn âm (train, dừng sớm trên val). So khoảng cách Δ đã hiệu chỉnh với
    khoảng cách dự đoán E[log ĝ | T=1] − E[log ĝ | T=0].
(2) Bootstrap đồng thời. 200 lần: lấy lại bệnh nhân train (có hoàn lại), huấn luyện lại M0 và M2, lấy lại bệnh nhân
    test, tính Δ cho năm concept. Khoảng tin cậy phản ánh cả biến thiên huấn luyện lẫn lấy mẫu test. Quy tắc: một
    đảo dấu chính được gọi là vững nếu khoảng của Δ_M0 và Δ_M2 nằm ở hai phía của 0. Họ chính gồm 3 concept × 2 họ
    learner = 6 so sánh; báo cáo thêm khoảng hiệu chỉnh Bonferroni (mức 1 − 0.05/6) theo xấp xỉ chuẩn.

Out -> Result/vr17_joint_bootstrap.json (+ .partial cho phần bootstrap)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import vr_common as R

PRIMARY = ["color_variegation", "size", "lesion_skin_contrast"]


def features(df, kind):
    if kind == "tabular":
        X, _ = R.VL.tabular_X(df)
        return X.astype(np.float32), 128
    z = np.load(os.path.join(R.EMB, "isic2024.npz"))
    E = z["emb"].astype(np.float32)
    return ((E - E.mean(0)) / (E.std(0) + 1e-6)).astype(np.float32), 256


def arrays(df, X, label=None, mask=None):
    n = len(df)
    y = df.Y.to_numpy(np.float32) if label is None else label.astype(np.float32)
    S = df.S.to_numpy(np.float32) if mask is None else mask.astype(np.float32)
    return dict(x=X, Y=y, F=S, S=S, sigma=np.ones(n, np.float32), L=np.ones(n, np.float32),
                U=np.ones(n, np.float32), m=np.zeros(n, np.float32))


def platt(lg_val, y_val, lg_te):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1e6, max_iter=1000).fit(lg_val[:, None], y_val)
    return m.coef_[0, 0] * lg_te + m.intercept_[0], float(m.coef_[0, 0])


def part1(df, fold):
    import vl_models as M
    tr, va, te = (np.flatnonzero(fold == k) for k in ("train", "val", "test"))
    Y, S = df.Y.to_numpy(), df.S.to_numpy()
    Tc = {c: R.SV.tertile(df, col).fillna(-1).to_numpy()[te] for c, col in R.CONCEPTS.items()}
    D = lambda lg, c: float(lg[Tc[c] == 1].mean() - lg[Tc[c] == 0].mean())
    out = {}
    for kind in ("tabular", "image"):
        X, hid = features(df, kind)
        gaps_raw, gaps_cal, preds, slopes = {c: [] for c in R.CONCEPTS}, {c: [] for c in R.CONCEPTS}, {c: [] for c in R.CONCEPTS}, []
        for seed in range(3):
            m2 = M.train("erm_verified", arrays(df, X), tr, va, hid=hid, seed=seed)
            m0 = M.train("erm_y", arrays(df, X), tr, va, hid=hid, seed=seed)
            l0v = R.logit(M.predict(m0, X[va])["pD"]); l0t = R.logit(M.predict(m0, X[te])["pD"])
            vs = va[S[va] == 1]
            l2v = R.logit(M.predict(m2, X[vs])["pD"]); l2t = R.logit(M.predict(m2, X[te])["pD"])
            c0, a0 = platt(l0v, Y[va], l0t); c2, a2 = platt(l2v, Y[vs], l2t)
            slopes.append({"M0": a0, "M2": a2})
            # ĝ: same MLP class, trained on recorded negatives with label S
            neg = (Y == 0).astype(np.float32)
            mg = M.train("erm_verified", arrays(df, X, label=S, mask=neg), tr, va, hid=hid, seed=seed)
            lgg = np.log(np.clip(M.predict(mg, X[te])["pD"], 1e-7, 1))
            for c in R.CONCEPTS:
                gaps_raw[c].append(D(l0t, c) - D(l2t, c)); gaps_cal[c].append(D(c0, c) - D(c2, c)); preds[c].append(D(lgg, c))
            R.log(f"(1) {kind} seed {seed}: Platt slope M0 {a0:.2f} M2 {a2:.2f}; color gap raw {gaps_raw['color_variegation'][-1]:+.2f} "
                  f"cal {gaps_cal['color_variegation'][-1]:+.2f} pred {preds['color_variegation'][-1]:+.2f}")
        mr = np.array([np.mean(gaps_raw[c]) for c in R.CONCEPTS]); mc = np.array([np.mean(gaps_cal[c]) for c in R.CONCEPTS])
        mp = np.array([np.mean(preds[c]) for c in R.CONCEPTS])
        out[kind] = {"concepts": list(R.CONCEPTS), "gap_raw": mr.tolist(), "gap_calibrated": mc.tolist(), "predicted": mp.tolist(),
                     "platt_slopes": slopes,
                     "sign_agreement_calibrated": int(np.sum(np.sign(mc) == np.sign(mp))),
                     "calibration_slope_raw": float(np.polyfit(mp, mr, 1)[0]),
                     "calibration_slope_calibrated": float(np.polyfit(mp, mc, 1)[0]),
                     "calibration_intercept_calibrated": float(np.polyfit(mp, mc, 1)[1]),
                     "mean_abs_diff_calibrated": float(np.mean(np.abs(mc - mp)))}
        R.log(f"(1) {kind}: predicted {np.round(mp, 2)} calibrated {np.round(mc, 2)} raw {np.round(mr, 2)} "
              f"slope cal {out[kind]['calibration_slope_calibrated']:.2f} raw {out[kind]['calibration_slope_raw']:.2f}")
    return out


def part2(df, fold, B):
    import vl_models as M
    from scipy.stats import norm
    tr_all, va, te_all = (np.flatnonzero(fold == k) for k in ("train", "val", "test"))
    pid = df[R.SV.GROUP].to_numpy()
    trp, tep = np.unique(pid[tr_all]), np.unique(pid[te_all])
    tr_by = {p: tr_all[pid[tr_all] == p] for p in trp}; te_by = {p: te_all[pid[te_all] == p] for p in tep}
    Tfull = {c: R.SV.tertile(df, col).fillna(-1).to_numpy() for c, col in R.CONCEPTS.items()}
    part = os.path.join(R.RES, "vr17_joint_bootstrap.json.partial")
    reps = json.load(open(part)) if os.path.exists(part) else {"tabular": [], "image": []}
    for kind in ("tabular", "image"):
        X, hid = features(df, kind)
        A = arrays(df, X)
        for b in range(len(reps[kind]), B):
            rng = np.random.default_rng(10_000 + b)
            tr = np.concatenate([tr_by[p] for p in rng.choice(trp, len(trp))])
            te = np.concatenate([te_by[p] for p in rng.choice(tep, len(tep))])
            m2 = M.train("erm_verified", A, tr, va, hid=hid, seed=b)
            m0 = M.train("erm_y", A, tr, va, hid=hid, seed=b)
            ute = np.unique(te)
            l0 = dict(zip(ute, R.logit(M.predict(m0, X[ute])["pD"]))); l2 = dict(zip(ute, R.logit(M.predict(m2, X[ute])["pD"])))
            L0 = np.array([l0[i] for i in te]); L2 = np.array([l2[i] for i in te])
            r = {}
            for c in R.CONCEPTS:
                T = Tfull[c][te]
                r[c] = [float(L0[T == 1].mean() - L0[T == 0].mean()), float(L2[T == 1].mean() - L2[T == 0].mean())]
            reps[kind].append(r)
            json.dump(reps, open(part, "w"))
            if b % 10 == 0:
                R.log(f"(2) {kind} rep {b}: color Δ_M0 {r['color_variegation'][0]:+.2f} Δ_M2 {r['color_variegation'][1]:+.2f}")
    z = norm.ppf(1 - 0.05 / 6 / 2)
    out = {"B": B, "families": {}}
    for kind, rr in reps.items():
        fam = {}
        for c in R.CONCEPTS:
            a = np.array([x[c] for x in rr])
            d0, d2, df_ = a[:, 0], a[:, 1], a[:, 0] - a[:, 1]
            q = lambda v: np.percentile(v, [2.5, 97.5]).tolist()
            bon = lambda v: [float(v.mean() - z * v.std(ddof=1)), float(v.mean() + z * v.std(ddof=1))]
            fam[c] = {"M0_mean": float(d0.mean()), "M0_ci": q(d0), "M0_ci_bonferroni": bon(d0),
                      "M2_mean": float(d2.mean()), "M2_ci": q(d2), "M2_ci_bonferroni": bon(d2),
                      "diff_ci": q(df_), "p_sign_differ": float(np.mean(np.sign(d0) != np.sign(d2))),
                      "robust_reversal_95": bool((q(d0)[0] > 0 and q(d2)[1] < 0) or (q(d0)[1] < 0 and q(d2)[0] > 0)),
                      "robust_reversal_bonferroni": bool((bon(d0)[0] > 0 and bon(d2)[1] < 0) or (bon(d0)[1] < 0 and bon(d2)[0] > 0))}
            R.log(f"(2) {kind} {c:22s} Δ_M0 {d0.mean():+.2f} {np.round(q(d0), 2)}  Δ_M2 {d2.mean():+.2f} {np.round(q(d2), 2)}  "
                  f"robust95={fam[c]['robust_reversal_95']} bonf={fam[c]['robust_reversal_bonferroni']}")
        out["families"][kind] = fam
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--parts", default="1,2")
    a = ap.parse_args()
    df, fold = R.load_isic()
    path = os.path.join(R.RES, "vr17_joint_bootstrap.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    if "1" in a.parts.split(","):
        out["calibrated_bridge"] = part1(df, fold); R.save_json(out, "vr17_joint_bootstrap.json")
    if "2" in a.parts.split(","):
        out["joint_bootstrap"] = part2(df, fold, a.B); R.save_json(out, "vr17_joint_bootstrap.json")


if __name__ == "__main__":
    main()
