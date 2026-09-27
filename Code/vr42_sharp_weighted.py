#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR42 — Tập định danh sharp của mục tiêu bệnh khi tertile concept không là hàm của input learner.

ψ_k = E[logit p | t=1] − E[logit p | t=0] = ∫ a_k(x̃) logit p(x̃) dP(x̃),  a_k = P(t=1|x̃)/P(t=1) − P(t=0|x̃)/P(t=0).
Với logit q ≤ logit p ≤ logit U từng điểm, cận sharp chọn đầu mút theo DẤU của a_k(x̃):
  ψ_L^sharp = E[L(X̃) | t=1] − E[L(X̃) | t=0],  L = logit q nếu a_k > 0, logit U nếu a_k < 0  (ψ_U đối xứng).
Công thức theo tertile quan sát (logit q cho t=1, logit U cho t=0) là cận ngoài hợp lệ; hai công thức trùng nhau khi t là
hàm của x̃ (họ bảng: concept nằm trong input).

(A) Mô phỏng lý thuyết (không huấn luyện): x̃ một chiều trên lưới, P(t|x̃) chồng lấn, q, σ, s(x̃) đã biết; so sánh cận theo
    tertile với cận sharp, kiểm tra cả hai chứa ψ thật và cận sharp đạt được; ví dụ dấu â sai làm cận thiếu an toàn.
(B) ISIC-2024: khớp lại M0 (seed 0–2, như phân tích chính, Platt trên val) để có q̂ trên test; σ̂ chung; ước lượng P(t|x̃)
    bằng gradient boosting chéo năm phần theo bệnh nhân trên test (ảnh: 64 thành phần chính); tính ψ_L theo tertile và
    sharp plug-in trên lưới sàn, s∗ của đường trung bình seed, và bootstrap bệnh nhân test 1,000 lần (có điều kiện trên
    learner và â).
Out -> Result/vr42_sharp_weighted.json
"""
from __future__ import annotations

import os

import numpy as np

import vr_common as R

S_FINE = np.round(np.arange(0.01, 1.0001, 0.01), 2)
PRIMARY = ["color_variegation", "size", "lesion_skin_contrast"]
B = 1000


def expit(z):
    return 1 / (1 + np.exp(-z))


def lg(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def tip(L):
    pos = np.flatnonzero(L > 0)
    return float(S_FINE[pos[0]]) if len(pos) else None


# ------------------------------------------------------------------ (A) lý thuyết
def theory():
    x = np.linspace(-3, 3, 601); w = np.exp(-x ** 2 / 2); w /= w.sum()
    out = []
    for overlap in (0.0, 0.5, 1.0, 2.0):          # độ nhiễu của t quanh x̃; 0 = t là hàm của x̃
        z = np.linspace(-4, 4, 161); wz = np.exp(-z ** 2 / 2); wz /= wz.sum()
        c = x[:, None] + overlap * z[None, :]      # concept = x̃ + nhiễu
        W = w[:, None] * wz[None, :]
        cv, cw = c.ravel(), W.ravel(); o = np.argsort(cv); cdf = np.cumsum(cw[o])
        lo_c, hi_c = cv[o][np.searchsorted(cdf, 1 / 3)], cv[o][np.searchsorted(cdf, 2 / 3)]
        p1 = (W * (c > hi_c)).sum(1) / w; p0 = (W * (c < lo_c)).sum(1) / w
        P1, P0 = (w * p1).sum(), (w * p0).sum()
        a = p1 / P1 - p0 / P0
        for smin, beta in ((0.5, 0.3), (0.7, 0.0), (0.9, -0.2)):
            p = expit(-4 + beta * x)                              # bệnh
            s = smin + (1 - smin) * expit(0.8 * x)                # xác minh ác tính, sàn đúng
            q = p * s; sig = q + (1 - p) * 0.01
            U = np.minimum(q / smin, q + 1 - sig)
            psi = (w * a * lg(p)).sum()
            naive_L = (w * p1 * lg(q)).sum() / P1 - (w * p0 * lg(U)).sum() / P0
            naive_U = (w * p1 * lg(U)).sum() / P1 - (w * p0 * lg(q)).sum() / P0
            Lx = np.where(a > 0, lg(q), lg(U)); Ux = np.where(a > 0, lg(U), lg(q))
            sh_L = (w * a * Lx).sum(); sh_U = (w * a * Ux).sum()
            # đạt được: p* = q nơi a>0, U nơi a<0 là một p hợp lệ, ψ(p*) = sh_L
            pstar = np.where(a > 0, q, U); att = abs((w * a * lg(pstar)).sum() - sh_L)
            flip = np.where(-a > 0, lg(q), lg(U)); wrong_L = (w * a * flip).sum()     # dấu â sai hoàn toàn
            out.append({"overlap": overlap, "s_min": smin, "beta": beta, "psi": float(psi), "naive": [float(naive_L), float(naive_U)],
                        "sharp": [float(sh_L), float(sh_U)], "attained_err": float(att),
                        "both_contain": bool(naive_L <= psi <= naive_U and sh_L - 1e-9 <= psi <= sh_U + 1e-9),
                        "sign_flipped_L": float(wrong_L)})
    return out


# ------------------------------------------------------------------ (B) ISIC-2024
def isic():
    import gpu_setup  # noqa: F401
    import vl_models as M
    from sklearn.ensemble import HistGradientBoostingClassifier
    from vr19_primary_bootstrap import features, arrays, platt_fit
    from vr30_q_bootstrap import gbm_input
    df, fold = R.load_isic()
    Y, S = df.Y.to_numpy(), df.S.to_numpy()
    tr, va, te = (np.flatnonzero(fold == k) for k in ("train", "val", "test"))
    sig = np.load(os.path.join(R.MISS_RES, "vl_nuisance.npz"))["sigma"][te]
    Tf = {c: R.SV.tertile(df, R.CONCEPTS[c]).fillna(-1).to_numpy()[te] for c in PRIMARY}
    Tall = {c: R.SV.tertile(df, R.CONCEPTS[c]).to_numpy()[te] for c in PRIMARY}
    pid = df[R.SV.GROUP].to_numpy()[te]; pats = np.unique(pid); pidx = {p: i for i, p in enumerate(pats)}
    pc = np.array([pidx[p] for p in pid])
    rng = np.random.default_rng(42); fold_p = rng.integers(0, 5, len(pats))[pc]
    W = [np.ones(len(pats))] + [np.bincount(rng.integers(0, len(pats), len(pats)), minlength=len(pats)).astype(float) for _ in range(B)]
    res = {}
    for kind in ("tabular", "image"):
        X, hid = features(df, kind); A = arrays(df, X); Xg = gbm_input(df, X, kind, tr)
        qs = []
        for seed in range(3):
            m0 = M.train("erm_y", A, tr, va, hid=hid, seed=seed)
            a0, b0 = platt_fit(R.logit(M.predict(m0, X[va])["pD"]), Y[va])
            qs.append(expit(a0 * R.logit(M.predict(m0, X[te])["pD"]) + b0))
        R.log(f"{kind}: M0 fitted")
        for c in PRIMARY:
            T = Tf[c]; t3 = np.where(np.isnan(Tall[c]), 2, Tall[c]).astype(int)     # 0 dưới, 1 trên, 2 giữa
            pr = np.zeros((len(te), 3))
            for k in range(5):                                       # chéo năm phần theo bệnh nhân
                trk, tek = fold_p != k, fold_p == k
                g = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.05, random_state=k).fit(Xg[te][trk], t3[trk])
                pb = g.predict_proba(Xg[te][tek]); cl = list(g.classes_)
                pr[tek, 0] = pb[:, cl.index(0)]; pr[tek, 1] = pb[:, cl.index(1)]
            P1, P0 = (T == 1).mean(), (T == 0).mean()
            ahat = pr[:, 1] / P1 - pr[:, 0] / P0
            acc = float(np.mean(np.sign(ahat[T != -1]) == np.where(T[T != -1] == 1, 1, -1)))

            def curves(wt):
                Ls, Lo = [], []
                for q in qs:
                    lq = lg(q); lo_, sh_ = [], []
                    for s in S_FINE:
                        lU = lg(np.minimum(q / s, q + 1 - sig))
                        m1, m0 = (T == 1), (T == 0)
                        mean = lambda v, m: np.average(v[m], weights=wt[m])
                        lo_.append(mean(lq, m1) - mean(lU, m0))
                        Lx = np.where(ahat > 0, lq, lU)
                        sh_.append(mean(Lx, m1) - mean(Lx, m0))
                    Lo.append(lo_); Ls.append(sh_)
                return np.mean(Lo, 0), np.mean(Ls, 0)

            o0, s0 = curves(W[0][pc])
            bo, bs = [], []
            for w in W[1:]:
                o, sh = curves(w[pc]); bo.append(tip(o)); bs.append(tip(sh))
            q95 = lambda v: (lambda x: None if x > 1 else float(x))(float(np.quantile([9.0 if z is None else z for z in v], 0.95, method="inverted_cdf")))
            res[f"{kind}|{c}"] = {"tertile_formula": {"s_star": tip(o0), "stability": q95(bo), "psiL": {str(s): float(o0[i]) for i, s in enumerate(S_FINE) if s in (0.5, 0.7, 0.9)}},
                                  "sharp_plugin": {"s_star": tip(s0), "stability": q95(bs), "psiL": {str(s): float(s0[i]) for i, s in enumerate(S_FINE) if s in (0.5, 0.7, 0.9)}},
                                  "sign_agreement": acc, "share_a_negative_in_upper": float(np.mean(ahat[T == 1] < 0)),
                                  "share_a_positive_in_lower": float(np.mean(ahat[T == 0] > 0))}
            R.log(f"{kind} {c}: tertile s* {tip(o0)} (stab {q95(bo)}) sharp s* {tip(s0)} (stab {q95(bs)}) sign agreement {acc:.3f}")
            R.save_json({"theory": TH, "isic": res}, "vr42_sharp_weighted.json")
    return res


if __name__ == "__main__":
    TH = theory()
    for r in TH:
        R.log(f"overlap {r['overlap']} s {r['s_min']}: ψ {r['psi']:+.3f} naive [{r['naive'][0]:+.3f}, {r['naive'][1]:+.3f}] sharp [{r['sharp'][0]:+.3f}, {r['sharp'][1]:+.3f}] "
              f"attained {r['attained_err']:.1e} flipped L {r['sign_flipped_L']:+.3f}")
    R.save_json({"theory": TH}, "vr42_sharp_weighted.json")
    isic()
