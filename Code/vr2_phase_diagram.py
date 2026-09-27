#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR2 — Phase diagram của đảo dấu do xác minh (proposal_v3 §14), với độ nhạy xác minh ác tính thực tế s(x) ≥ 0.5.

SCM trên concept c ~ N(0,1) (lưới [−4,4]) và mức độ nặng ẩn h ~ N(0,1) (Gauss–Hermite 16 nút):
  logit P(D=1 | c, h) = α + β c + γ h                         α chỉnh theo tỉ lệ bệnh
  π⁰(c)   = expit(b₀ + b' c)                                  xác minh tổn thương lành, mù với D, TB ≈ 0.3%
  π¹(c,h) = s_lo + (1 − s_lo)·expit(a c + γ h)                xác minh ác tính, luôn ≥ s_lo
Lưới: β ∈ {0.25, 0.5, 1}, b' ∈ {0, .5, 1, 1.5, 2, 3}, s_lo ∈ {.5, .7, .9}, a ∈ {−1, 0, 1},
      prevalence ∈ {0.002, 0.02}, γ ∈ {0, 1}  → 648 điểm.

Mỗi điểm, bằng kỳ vọng chính xác (không nhiễu mẫu):
  tertile: θ = log OR_D, A, B, δ = log B − log A, log OR_{D|S}, log RR_Y; loại (Định lý 4);
  learner logistic tối ưu tổng thể trên c: hệ số của M0 (Y), M2 (Y | S=1), D (tham chiếu), V (xác minh lành).
Kiểm tra:
  (i)   Định lý 4 ở mức tertile (đồng nhất thức, phải khớp tuyệt đối);
  (ii)  đảo dấu của HỆ SỐ M2 so với đảo dấu tertile theo Định lý 4 (learner liên tục có theo biên pha không?);
  (iii) dự đoán chỉ từ quan sát, β_M2 ≈ β_M0 − β_V (Định lý 5), có đoán đúng dấu của M2 không.
Thêm: trên 24 điểm ngẫu nhiên, học MLP hữu hạn mẫu (n = 300k) dưới M0 và M2 và so dấu của tương phản học được
(tertile trên − dưới của logit, trên cùng tập test) với loại theo Định lý 4.

Out -> Result/vr2_phase_diagram.json
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import brentq

import vr_common as R
from vr1_reversal_theory import classify, pop_logit

CG = np.linspace(-4, 4, 161)
CW = np.exp(-CG ** 2 / 2); CW /= CW.sum()
HN, HW = np.polynomial.hermite_e.hermegauss(16); HW = HW / HW.sum()
Q1, Q3 = -0.4307, 0.4307


def expit(z):
    return 1 / (1 + np.exp(-z))


def population(beta, bprime, s_lo, a, prev, gamma):
    C, Hh = np.meshgrid(CG, HN, indexing="ij")
    Wc = CW[:, None] * HW[None, :]
    alpha = brentq(lambda al: np.sum(Wc * expit(al + beta * C + gamma * Hh)) - prev, -20, 5)
    pD = expit(alpha + beta * C + gamma * Hh)
    b0 = brentq(lambda b: np.sum(CW * expit(b + bprime * CG)) - 0.003, -30, 5)
    pi0 = expit(b0 + bprime * CG)
    pi1 = s_lo + (1 - s_lo) * expit(a * C + gamma * Hh)
    pD_c = np.sum(HW * pD, 1)
    pY_c = np.sum(HW * pD * pi1, 1)
    pV_c = (1 - pD_c) * pi0
    return pD_c, pY_c, pV_c, pi0


def tertile_stats(pD_c, pY_c, pV_c, pi0):
    out = {}
    for t, m in ((0, CG <= Q1), (1, CG >= Q3)):
        w = CW[m] / CW[m].sum()
        pD = np.sum(w * pD_c[m]); pY = np.sum(w * pY_c[m]); pV = np.sum(w * pV_c[m])
        out[t] = dict(pD=pD, pY=pY, pV=pV, s=pY / pD, a=pV / (1 - pD))
    odds = lambda p: p / (1 - p)
    theta = np.log(odds(out[1]["pD"]) / odds(out[0]["pD"]))
    logA = np.log(out[1]["s"] / out[0]["s"]); logB = np.log(out[1]["a"] / out[0]["a"])
    logOR_DS = np.log((out[1]["pY"] / out[1]["pV"]) / (out[0]["pY"] / out[0]["pV"]))
    logRR_Y = np.log(out[1]["pY"] / out[0]["pY"])
    return dict(theta=theta, logA=logA, logB=logB, delta=logB - logA, logOR_DS=logOR_DS, logRR_Y=logRR_Y,
                identity_residual=logOR_DS - (theta - (logB - logA)), category=classify(theta, logB - logA),
                s0=out[0]["s"], s1=out[1]["s"])


def learners(pD_c, pY_c, pV_c):
    X = CG[:, None]
    bD = pop_logit(X, CW, pD_c)[1]
    bM0 = pop_logit(X, CW, pY_c)[1]
    wS = CW * (pY_c + pV_c)
    bM2 = pop_logit(X, wS, pY_c / (pY_c + pV_c))[1]
    bV = pop_logit(X, CW, pV_c)[1]
    return dict(beta_D=bD, beta_M0=bM0, beta_M2=bM2, beta_V=bV, M2_pred_observable=bM0 - bV)


def mlp_check(rng, pts):
    import vl_models as M
    res = []
    for p in pts:
        n = 300_000
        c = rng.standard_normal(n).astype(np.float32); h = rng.standard_normal(n)
        pD, _, _, _ = population(**{k: p[k] for k in ("beta", "bprime", "s_lo", "a", "prev", "gamma")})
        alpha = brentq(lambda al: expit(al + p["beta"] * c[:100000] + p["gamma"] * h[:100000]).mean() - p["prev"], -20, 5)
        D = rng.random(n) < expit(alpha + p["beta"] * c + p["gamma"] * h)
        b0 = brentq(lambda b: expit(b + p["bprime"] * c[:100000]).mean() - 0.003, -30, 5)
        pi0 = expit(b0 + p["bprime"] * c)
        pi1 = p["s_lo"] + (1 - p["s_lo"]) * expit(p["a"] * c + p["gamma"] * h)
        S = rng.random(n) < np.where(D, pi1, pi0)
        Y = (D & S).astype(np.float32)
        x = np.column_stack([c, rng.standard_normal((n, 2))]).astype(np.float32)
        arrs = dict(x=x, Y=Y, F=S.astype(np.float32), S=S.astype(np.float32), sigma=np.ones(n, np.float32),
                    L=np.ones(n, np.float32), U=np.ones(n, np.float32), m=np.zeros(n, np.float32))
        idx = rng.permutation(n); tr, va, te = idx[:180000], idx[180000:240000], idx[240000:]
        T = np.where(c <= Q1, 0, np.where(c >= Q3, 1, -1))
        out = dict(p)
        for meth in ("erm_y", "erm_verified"):
            mdl = M.train(meth, arrs, tr, va, hid=64, epochs=20, seed=0)
            lg = R.logit(M.predict(mdl, x[te])["pD"])
            out[f"learned_contrast_{meth}"] = float(lg[T[te] == 1].mean() - lg[T[te] == 0].mean())
        out["n_verified_train"] = int(S[tr].sum())
        out["mlp_M2_reversed"] = bool(out["learned_contrast_erm_verified"] * p["theta"] < 0)
        out["theory_reversal"] = p["category"] == "reversal"
        res.append(out)
        R.log(f"  MLP β={p['beta']} b'={p['bprime']} s_lo={p['s_lo']} a={p['a']}: θ={p['theta']:+.2f} δ={p['delta']:+.2f} "
              f"{p['category']:13s} | M0 Δ={out['learned_contrast_erm_y']:+.2f} M2 Δ={out['learned_contrast_erm_verified']:+.2f}")
    return res


def main():
    rows = []
    for beta, bp, s_lo, a, prev, gamma in itertools.product([0.25, 0.5, 1.0], [0, 0.5, 1, 1.5, 2, 3],
                                                            [0.5, 0.7, 0.9], [-1, 0, 1], [0.002, 0.02], [0, 1]):
        pD_c, pY_c, pV_c, pi0 = population(beta, bp, s_lo, a, prev, gamma)
        r = {"beta": beta, "bprime": bp, "s_lo": s_lo, "a": a, "prev": prev, "gamma": gamma}
        r.update(tertile_stats(pD_c, pY_c, pV_c, pi0)); r.update(learners(pD_c, pY_c, pV_c))
        r["M2_coef_reversed"] = bool(r["beta_M2"] * r["beta_D"] < 0)
        r["M0_coef_reversed"] = bool(r["beta_M0"] * r["beta_D"] < 0)
        r["observable_pred_reversed"] = bool(r["M2_pred_observable"] * r["beta_D"] < 0)
        rows.append(r)
    cats = {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    summ = {
        "n_points": len(rows), "category_counts": cats,
        "max_identity_residual": max(abs(r["identity_residual"]) for r in rows),
        "min_s_malignant": min(min(r["s0"], r["s1"]) for r in rows),
        "M2_coef_vs_tertile_theorem4_agreement": float(np.mean([r["M2_coef_reversed"] == (r["category"] == "reversal")
                                                                for r in rows])),
        "observable_prediction_agreement": float(np.mean([r["observable_pred_reversed"] == r["M2_coef_reversed"]
                                                          for r in rows])),
        "M0_coef_reversed_count": int(sum(r["M0_coef_reversed"] for r in rows)),
        "reversal_frac_by_bprime": {str(b): float(np.mean([r["category"] == "reversal" for r in rows if r["bprime"] == b]))
                                    for b in [0, 0.5, 1, 1.5, 2, 3]},
        "reversal_frac_by_beta": {str(b): float(np.mean([r["category"] == "reversal" for r in rows if r["beta"] == b]))
                                  for b in [0.25, 0.5, 1.0]},
    }
    R.log(summ)
    rng = np.random.default_rng(R.SEED)
    pick = [rows[i] for i in rng.choice(len(rows), 24, replace=False)]
    mlp = mlp_check(rng, pick)
    summ["mlp_M2_sign_agrees_with_theorem4"] = float(np.mean([m["mlp_M2_reversed"] == m["theory_reversal"] for m in mlp]))
    R.log(f"MLP agreement: {summ['mlp_M2_sign_agrees_with_theorem4']:.3f}")
    R.save_json({"summary": summ, "rows": rows, "mlp": mlp}, "vr2_phase_diagram.json")


if __name__ == "__main__":
    main()
