#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR1 — Định lý 4 (điều kiện đảo dấu) và Định lý 5 (đảo dấu của learner tối ưu tổng thể), proposal_v3 §9–10.

Ký hiệu. θ = log OR_D (tương phản bệnh thật), A = π₁¹/π₀¹ (xác minh ác tính), B = π₁⁰/π₀⁰ (xác minh lành),
δ = log B − log A (tương phản chọn lọc vi phân).

Định lý 4 (chính xác, không xấp xỉ). odds(D | S=1, t) = odds(D | t) · πₜ¹/πₜ⁰, nên
      log OR_{D|S} = θ − δ.
  Với θ > 0:  δ < 0 khuếch đại;  0 < δ < θ suy giảm;  δ = θ triệt tiêu;  δ > θ ĐẢO DẤU (dương → âm).
  Với θ < 0:  δ > 0 khuếch đại;  θ < δ < 0 suy giảm;  δ = θ triệt tiêu;  δ < θ ĐẢO DẤU (âm → dương).
  Gộp lại: đảo dấu ⇔ |δ| > |θ| và sign(δ) = sign(θ). Learner trên nhãn ghi nhận: log RR_Y = log RR_D + log A,
  nên hiệu giữa hai learner, log RR_Y − log OR_{D|S} ≈ log B (bệnh hiếm), QUAN SÁT ĐƯỢC.

Định lý 5 (learner logistic). Nếu logit P(D=1 | x) = α + βᵀx và tỉ số khả năng xác minh
LR(x) = π¹(x)/π⁰(x) = exp(ℓ₀ + λᵀx), thì P(D=1 | x, S=1) là logistic với hệ số β + λ, nên hệ số tối ưu tổng
thể của learner chỉ-đã-xác-minh (M2) là ĐÚNG β + λ, với λ = ∇log π¹ − ∇log π⁰. Nếu thêm π¹(x) = exp(a₀ + aᵀx)
và bệnh hiếm, learner trên Y (M0) có hệ số ≈ β + a. Do đó:
  M2 đảo dấu concept k ⇔ sign(β_k + λ_k) ≠ sign(β_k);   M0 − M2 ≈ ∇log π⁰ = b (độ dốc xác minh lành, quan sát được).

Script này:
  A. kiểm Định lý 4 trên 20,000 mô hình rời rạc ngẫu nhiên (đồng nhất thức phải đúng tới sai số máy);
  B. kiểm Định lý 5 bằng hồi quy logistic tối ưu tổng thể (trọng số chính xác trên lưới x, không nhiễu mẫu),
     cả khi LR log-tuyến tính (định lý phải đúng) và khi π¹ bị chặn dưới kiểu thực tế s(x) ≥ s_min (xấp xỉ);
  C. kiểm hệ quả quan sát được trên ISIC-2024: độ dốc(M0) − độ dốc(M2) ≈ độ dốc(V), từng concept chuẩn hoá,
     hiệu chỉnh Z, và bản đa concept; khoảng tin cậy từ hàm ảnh hưởng theo bệnh nhân.

Out -> Result/vr1_reversal_theory.json
"""
from __future__ import annotations

import numpy as np

import vr_common as R

SV = R.SV


def classify(theta, delta, tol=1e-9):
    ts = theta - delta
    if abs(theta) < tol:
        return "null_true"
    if np.sign(ts) != np.sign(theta) and abs(ts) > tol:
        return "reversal"
    if abs(ts) <= tol:
        return "nullified"
    return "attenuation" if abs(ts) < abs(theta) else "amplification"


def part_a(rng, n=20000):
    err, cats = [], {}
    agree = 0
    for _ in range(n):
        p = rng.uniform(1e-4, 0.3, 2)                       # P(D=1 | t)
        pi1 = rng.uniform(0.05, 1.0, 2)                     # P(S=1 | D=1, t)
        pi0 = rng.uniform(1e-4, 0.2, 2)                     # P(S=1 | D=0, t)
        y = p * pi1; v = (1 - p) * pi0                       # P(Y=1|t), P(S=1, D=0 | t)
        or_ds = (y[1] / v[1]) / (y[0] / v[0])
        theta = np.log(p[1] / (1 - p[1])) - np.log(p[0] / (1 - p[0]))
        delta = np.log(pi0[1] / pi0[0]) - np.log(pi1[1] / pi1[0])
        err.append(abs(np.log(or_ds) - (theta - delta)))
        c_obs = classify(theta, np.log(np.exp(theta) / or_ds))
        c_th = classify(theta, delta)
        agree += c_obs == c_th
        cats[c_th] = cats.get(c_th, 0) + 1
    return {"n_models": n, "max_abs_identity_error": float(max(err)), "category_agreement": agree / n,
            "category_counts": cats}


def pop_logit(Xg, w, y_prob, iters=100):
    """Hồi quy logistic tối ưu tổng thể: cực đại Σ w [y log p + (1−y) log(1−p)] với y là XÁC SUẤT."""
    X = np.column_stack([np.ones(len(Xg)), Xg])
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        g = X.T @ (w * (y_prob - p))
        H = (X * (w * p * (1 - p))[:, None]).T @ X + 1e-10 * np.eye(len(b))
        st = np.linalg.solve(H, g); b += st
        if np.abs(st).max() < 1e-11:
            break
    return b


def part_b():
    """Một concept c và một hiệp biến z tương quan ρ với c, cả hai chuẩn cắt cụt trên [−4, 4] (lưới 161 điểm).
    Cắt cụt để π¹, π⁰ ≤ 1 mà không phải kẹp: kẹp sẽ phá tính log-tuyến tính mà Định lý 5 cần."""
    gh = np.linspace(-4, 4, 161)
    gw = np.exp(-gh ** 2 / 2); gw = gw / gw.sum()
    rows = []
    for rho in (0.0, 0.5):
        C, Zu = np.meshgrid(gh, gh, indexing="ij")
        W = np.outer(gw, gw).ravel()
        c = C.ravel(); z = np.clip(rho * c + np.sqrt(1 - rho ** 2) * Zu.ravel(), -4, 4)
        X = np.column_stack([c, z])
        for beta in (0.3, 0.8):
            for a in (-0.3, 0.0, 0.3):
                for b in (0.0, 0.5, 1.0, 1.6):
                    for kind in ("loglinear", "floored"):
                        alpha = -5.3                                   # prevalence ≈ 0.5–1%
                        pD = 1 / (1 + np.exp(-(alpha + beta * c + 0.3 * z)))
                        if kind == "loglinear":
                            pi1 = 0.25 * np.exp(a * c)                              # ≤ 0.83 trên [−4, 4]
                            lam_th = a - b
                        else:
                            s_min = 0.5
                            pi1 = s_min + (1 - s_min) / (1 + np.exp(-(3 * a * c)))
                            # độ dốc trung bình của log π¹ theo c, dưới phân phối D=1
                            dl = (3 * a) * (1 - s_min) * np.exp(-3 * a * c) / (1 + np.exp(-3 * a * c)) ** 2 / pi1
                            lam_th = float(np.sum(W * pD * dl) / np.sum(W * pD)) - b
                        pi0 = 0.003 * np.exp(b * c - 0.5 * b ** 2)                 # ≤ 0.46 trên [−4, 4]
                        # M2: tổng thể có điều kiện S=1, nhãn D; trọng số ∝ P(S=1 | x)
                        wS = W * (pD * pi1 + (1 - pD) * pi0)
                        yS = pD * pi1 / (pD * pi1 + (1 - pD) * pi0)
                        bM2 = pop_logit(X, wS, yS)
                        # M0: toàn bộ, nhãn Y = D·S
                        bM0 = pop_logit(X, W, pD * pi1)
                        bV = pop_logit(X, W, (1 - pD) * pi0)
                        # β "thật" theo cùng lớp learner (logistic trên x, nhãn D)
                        bD = pop_logit(X, W, pD)
                        rows.append({"rho": rho, "beta": beta, "a": a, "b": b, "selection": kind,
                                     "beta_D_fit": bD[1], "beta_M2": bM2[1], "beta_M0": bM0[1], "slope_V": bV[1],
                                     "theorem5_M2_pred": beta + lam_th,
                                     "abs_err_M2": abs(bM2[1] - (beta + lam_th)),
                                     "M0_minus_M2": bM0[1] - bM2[1],
                                     # |hệ số| < 1e-6 là biên triệt tiêu (β + λ = 0), không tính là đảo dấu
                                     "reversal_M2": bool(bM2[1] * beta < 0 and abs(bM2[1]) > 1e-6),
                                     "reversal_pred": bool((beta + lam_th) * beta < 0 and abs(beta + lam_th) > 1e-6)})
    ll = [r for r in rows if r["selection"] == "loglinear"]
    fl = [r for r in rows if r["selection"] == "floored"]
    summ = {"loglinear_max_abs_err_M2": max(r["abs_err_M2"] for r in ll),
            "loglinear_reversal_agreement": float(np.mean([r["reversal_M2"] == r["reversal_pred"] for r in ll])),
            "floored_max_abs_err_M2": max(r["abs_err_M2"] for r in fl),
            "floored_reversal_agreement": float(np.mean([r["reversal_M2"] == r["reversal_pred"] for r in fl])),
            "M0_minus_M2_vs_slopeV_max_abs_diff": max(abs(r["M0_minus_M2"] - r["slope_V"]) for r in rows),
            "n_reversal_cases": int(sum(r["reversal_M2"] for r in rows)), "n": len(rows)}
    return {"summary": summ, "rows": rows}


def part_c():
    """ISIC-2024: độ dốc logistic của concept chuẩn hoá (hiệu chỉnh Z) cho Y (M0), Y|S=1 (M2) và V."""
    df = SV.load()
    Z = SV.build_Z(df)
    g = df[SV.GROUP].to_numpy()
    out = {"single": {}, "joint": {}}
    cols = list(R.CONCEPTS.values())
    Cz = {}
    for c, col in R.CONCEPTS.items():
        x = df[col].astype(float)
        Cz[c] = ((x - x.mean()) / x.std()).fillna(0.0).to_numpy()
    S = df.S.to_numpy() == 1
    Zs = SV.build_Z(df[S], min_count=20)

    def fit(Xc, Xz, y, grp):
        X = np.column_stack([np.ones(len(y)), Xc, Xz])
        b, IF, _ = SV.logit_if(X, y, grp)
        return b, IF

    for c in R.CONCEPTS:
        b0, I0 = fit(Cz[c][:, None], Z.to_numpy(), df.Y.to_numpy(), g)
        b2, I2 = fit(Cz[c][S][:, None], Zs.to_numpy(), df.Y.to_numpy()[S], g[S])
        bv, Iv = fit(Cz[c][:, None], Z.to_numpy(), df.V.to_numpy(), g)
        cov = SV.joint_cov([I0, I2, Iv], [1, 1, 1])
        diff = b0[1] - b2[1] - bv[1]
        se = np.sqrt(np.array([1, -1, -1]) @ cov @ np.array([1, -1, -1]))
        out["single"][c] = {"slope_M0": b0[1], "slope_M2": b2[1], "slope_V": bv[1],
                            "se": np.sqrt(np.diag(cov)).tolist(),
                            "M0_minus_M2": b0[1] - b2[1], "identity_residual": diff, "residual_se": se,
                            "sign_reversal_M0_vs_M2": bool(np.sign(b0[1]) != np.sign(b2[1]))}
        R.log(f"C {c:22s} M0={b0[1]:+.3f} M2={b2[1]:+.3f} V={bv[1]:+.3f}  M0−M2={b0[1] - b2[1]:+.3f} "
              f"vs V: residual {diff:+.3f} ± {se:.3f}")
    Xc = np.column_stack([Cz[c] for c in R.CONCEPTS])
    b0, I0 = fit(Xc, Z.to_numpy(), df.Y.to_numpy(), g)
    b2, I2 = fit(Xc[S], Zs.to_numpy(), df.Y.to_numpy()[S], g[S])
    bv, Iv = fit(Xc, Z.to_numpy(), df.V.to_numpy(), g)
    for j, c in enumerate(R.CONCEPTS, start=1):
        cov = SV.joint_cov([I0, I2, Iv], [j, j, j])
        se = np.sqrt(np.array([1, -1, -1]) @ cov @ np.array([1, -1, -1]))
        out["joint"][c] = {"slope_M0": b0[j], "slope_M2": b2[j], "slope_V": bv[j],
                           "identity_residual": b0[j] - b2[j] - bv[j], "residual_se": se,
                           "sign_reversal_M0_vs_M2": bool(np.sign(b0[j]) != np.sign(b2[j]))}
    return out


def main():
    rng = np.random.default_rng(R.SEED)
    out = {"theorem4": part_a(rng)}
    R.log(f"T4: {out['theorem4']}")
    out["theorem5_population"] = part_b()
    R.log(f"T5: {out['theorem5_population']['summary']}")
    out["isic2024_slopes"] = part_c()
    R.save_json(out, "vr1_reversal_theory.json")


if __name__ == "__main__":
    main()
