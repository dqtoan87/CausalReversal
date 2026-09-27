#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR10 — Kiểm chứng ký hiệu cuối cùng cho Định lý 4 và 5, bằng sympy.

Miền (positivity): p_t = P(D=1 | t) ∈ (0,1); π_t¹ = P(S=1 | D=1, t) ∈ (0,1]; π_t⁰ = P(S=1 | D=0, t) ∈ (0,1].
Mọi symbol được khai báo dương, nên các phép biến đổi log là hợp lệ trên đúng miền này.

Các mệnh đề được kiểm:
  T4.1  log OR_{D|S} = θ + log A − log B                            (đồng nhất thức, chính xác)
  T4.2  θ > 0: đảo dấu ⇔ δ > θ ⇔ B/A > OR_D;  suy giảm ⇔ 0 < δ < θ;  khuếch đại ⇔ δ < 0
  T4.3  θ < 0: đảo dấu ⇔ δ < θ;  suy giảm ⇔ θ < δ < 0;  khuếch đại ⇔ δ > 0
  T4.4  biên: A = B ⇒ OR_{D|S} = OR_D;  OR_D = 1 ⇒ OR_{D|S} = A/B (chọn lọc một mình tạo ra liên hệ giả)
  T1    RR_Y = RR_D · A                                               (chính xác, vì Y = D·S)
  T5.1  logit P(D | x, S=1) = logit P(D | x) + log π¹(x) − log π⁰(x)  (chính xác)
  T5.2  với logit P(D|x) = α + βx và π¹/π⁰ = exp(ℓ₀ + λx): hệ số của learner M2 là β + λ
  T5.3  log P(Y | x) = log P(D | x) + log π¹(x) chính xác; logit − log = −log(1 − P(Y|x)) ∈ [0, P/(1−P)],
        nên hệ số M0 ≈ β + a chỉ là xấp xỉ bệnh hiếm, sai số bị chặn bởi P(Y|x)/(1 − P(Y|x)).

Out -> Result/vr10_symbolic.json
"""
from __future__ import annotations

import sympy as sp

import vr_common as R


def main():
    p0, p1 = sp.symbols("p0 p1", positive=True)
    a0, a1, b0, b1 = sp.symbols("pi1_0 pi1_1 pi0_0 pi0_1", positive=True)       # π¹_t, π⁰_t
    odds = lambda p: p / (1 - p)
    OR_D = odds(p1) / odds(p0)
    A, B = a1 / a0, b1 / b0
    OR_DS = (p1 * a1 / ((1 - p1) * b1)) / (p0 * a0 / ((1 - p0) * b0))
    res = {}
    res["T4.1_identity"] = sp.simplify(OR_DS - OR_D * A / B) == 0
    lhs = sp.expand_log(sp.log(OR_DS), force=True)
    rhs = sp.expand_log(sp.log(OR_D) + sp.log(A) - sp.log(B), force=True)
    res["T4.1_log_form"] = sp.simplify(lhs - rhs) == 0

    th, d, tau = sp.symbols("theta delta tau", real=True)
    thp = sp.Symbol("theta_p", positive=True)
    # θ > 0
    rev = sp.solve_univariate_inequality(thp - d < 0, d, relational=False)
    att = sp.solve_univariate_inequality(sp.And(thp - d > 0, thp - d < thp).as_set().as_relational(d) if False else
                                         (thp - d > 0), d, relational=False).intersect(
        sp.solve_univariate_inequality(thp - d < thp, d, relational=False))
    amp = sp.solve_univariate_inequality(thp - d > thp, d, relational=False)
    res["T4.2_reversal_set"] = str(rev); res["T4.2_reversal_ok"] = rev == sp.Interval.open(thp, sp.oo)
    res["T4.2_attenuation_set"] = str(att)
    # sympy không tự rút Intersection((−∞, θ), (0, ∞)) khi θ là ký hiệu; kiểm bằng θ dương cụ thể
    res["T4.2_attenuation_ok"] = all(att.subs(thp, v) == sp.Interval.open(0, v)
                                     for v in (sp.Rational(1, 10), 1, sp.Rational(7, 3), 50))
    res["T4.2_amplification_set"] = str(amp); res["T4.2_amplification_ok"] = amp == sp.Interval.open(-sp.oo, 0)
    # δ > θ ⇔ B/A > OR_D (δ = log B − log A, θ = log OR_D, log đơn điệu)
    ORd, Bs, As = sp.symbols("OR_D B A", positive=True)
    res["T4.2_ratio_form_ok"] = sp.simplify(sp.log(Bs / As) - (sp.log(Bs) - sp.log(As))) == 0
    # θ < 0: θ = −τ
    tn = sp.Symbol("tau_p", positive=True)
    thn = -tn
    rev_n = sp.solve_univariate_inequality(thn - d > 0, d, relational=False)
    att_n = sp.solve_univariate_inequality(thn - d < 0, d, relational=False).intersect(
        sp.solve_univariate_inequality(thn - d > thn, d, relational=False))
    amp_n = sp.solve_univariate_inequality(thn - d < thn, d, relational=False)
    res["T4.3_reversal_ok"] = rev_n == sp.Interval.open(-sp.oo, -tn)
    res["T4.3_attenuation_ok"] = all(att_n.subs(tn, v) == sp.Interval.open(-v, 0)
                                     for v in (sp.Rational(1, 10), 1, sp.Rational(7, 3), 50))
    res["T4.3_amplification_ok"] = amp_n == sp.Interval.open(0, sp.oo)
    # biên
    res["T4.4_A_equals_B_no_distortion"] = sp.simplify((OR_D * A / B).subs(b1, b0 * a1 / a0) - OR_D) == 0
    res["T4.4_null_disease_gives_A_over_B"] = sp.simplify((OR_D * A / B).subs(p1, p0) - A / B) == 0
    # T1
    RR_Y = (p1 * a1) / (p0 * a0)
    res["T1_RR_identity"] = sp.simplify(RR_Y - (p1 / p0) * A) == 0

    # T5
    x = sp.Symbol("x", real=True)
    al, be, l0, la = sp.symbols("alpha beta ell0 lam", real=True)
    pD = 1 / (1 + sp.exp(-(al + be * x)))
    # π¹(x), π⁰(x) là giá trị dương tuỳ ý tại x (positivity); khai báo dương để phép log hợp lệ
    P1, P0 = sp.symbols("pi1_x pi0_x", positive=True)
    pDS = pD * P1 / (pD * P1 + (1 - pD) * P0)
    logit = lambda q: sp.log(q / (1 - q))
    # odds(D | x, S=1) / [odds(D | x) · π¹/π⁰] phải bằng 1 đúng
    ratio = sp.simplify((pDS / (1 - pDS)) / ((pD / (1 - pD)) * P1 / P0))
    res["T5.1_exact"] = ratio == 1
    pDS_ll = pDS.subs(P1, P0 * sp.exp(l0 + la * x))
    slope = sp.simplify(sp.diff(sp.expand_log(logit(pDS_ll), force=True), x))
    res["T5.2_M2_slope"] = str(slope); res["T5.2_ok"] = sp.simplify(slope - (be + la)) == 0
    # 0 < P < 1: viết P = 1/(1 + e^{-u}) để sympy biết cả P và 1 − P dương
    u = sp.Symbol("u", real=True)
    P = 1 / (1 + sp.exp(-u))
    gap = sp.simplify(sp.log(P / (1 - P)) - sp.log(P))                       # logit − log
    res["T5.3_logit_minus_log"] = str(gap)
    res["T5.3_ok"] = sp.simplify(sp.exp(gap) - 1 / (1 - P)) == 0             # gap = −log(1 − P)
    # chặn: −log(1 − P) ≤ P/(1 − P) (vì log(1 + z) ≤ z với z = P/(1 − P))
    z = sp.Symbol("z", positive=True)
    res["T5.3_bound_ok"] = bool(sp.limit(sp.log(1 + z) - z, z, 0) == 0 and
                                sp.simplify(sp.diff(sp.log(1 + z) - z, z) + z / (1 + z)) == 0)
    # Proposition 1 (learner gap): logit f0 − logit f2 = log g, g = P(S=1 | x, Y=0) = (1−p)π⁰ / (1 − pπ¹)
    pp, a1x, a0x = sp.symbols("p_x pi1_x2 pi0_x2", positive=True)
    f0 = pp * a1x; f2 = pp * a1x / (pp * a1x + (1 - pp) * a0x); gx = (1 - pp) * a0x / (1 - pp * a1x)
    odds_ratio = sp.simplify((f0 / (1 - f0)) / (f2 / (1 - f2)) - gx)
    res["P1.1_learner_gap_exact"] = odds_ratio == 0
    # hằng số cộng vào logit (lấy mẫu lại lớp) triệt tiêu trong hiệu hai tertile
    c0, l1, l0 = sp.symbols("c0 l1 l0", real=True)
    res["P1.2_constant_cancels"] = sp.simplify(((l1 + c0) - (l0 + c0)) - (l1 - l0)) == 0
    # Hệ quả dose: π⁰ ∝ exp(η z), π¹ hằng số ⇒ logit f2_η − logit f2_0 = −η z
    eta, zz, kk = sp.symbols("eta z k", real=True)
    lg = lambda a0: sp.log(pp * a1x / ((1 - pp) * a0))
    res["P1.3_dose_shift"] = sp.simplify(sp.expand_log(lg(a0x * sp.exp(eta * zz + kk)) - lg(a0x), force=True) + eta * zz + kk) == 0
    # Proposition 1, dạng tổng quát: X nhận hai giá trị x_a, x_b cho cùng x̃ (đầu vào không quan sát hết X).
    # odds(Y=1 | x̃, R=1) = odds(Y=1 | x̃)·r₁ / E[r₀(X) | x̃, Y=0]
    qa, qb, wa, r1, r0a, r0b = sp.symbols("q_a q_b w_a r_1 r0_a r0_b", positive=True)
    wb = 1 - wa
    num = r1 * (wa * qa + wb * qb)                       # P(Y=1, R=1 | x̃)
    den = wa * (1 - qa) * r0a + wb * (1 - qb) * r0b      # P(Y=0, R=1 | x̃)
    qx = wa * qa + wb * qb
    er0 = den / (1 - qx)                                 # E[r₀(X) | x̃, Y=0]
    res["P1.4_general_offset"] = sp.simplify(num / den - (qx / (1 - qx)) * r1 / er0) == 0
    res["all_passed"] = all(v for k, v in res.items() if k.endswith("_ok") or k in
                            ("T4.1_identity", "T4.1_log_form", "T4.4_A_equals_B_no_distortion",
                             "T4.4_null_disease_gives_A_over_B", "T1_RR_identity", "T5.1_exact",
                             "P1.1_learner_gap_exact", "P1.2_constant_cancels", "P1.3_dose_shift", "P1.4_general_offset"))
    res["note"] = ("T5.3_bound: d/dz[log(1+z) − z] = −z/(1+z) < 0 và bằng 0 tại z=0, nên log(1+z) ≤ z; "
                   "với z = P/(1−P): −log(1−P) = log(1+z) ≤ P/(1−P).")
    for k, v in res.items():
        R.log(f"{k}: {v}")
    R.save_json({k: (bool(v) if isinstance(v, (bool, sp.logic.boolalg.BooleanAtom)) else v) for k, v in res.items()},
                "vr10_symbolic.json")


if __name__ == "__main__":
    main()
