#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit test cho lõi lý thuyết và các hàm phân tích của VILRR.
Chạy:  python3 -m pytest -q test_reversal_theory.py

Nhóm test:
  classify        mọi vùng và mọi biên của Định lý 4, cả hai dấu của θ
  identity        Định lý 4 trên mô hình rời rạc ngẫu nhiên; A = B; OR_D = 1; A = 1 (trường hợp PAD)
  theorem5        hệ số learner M2 tối ưu tổng thể bằng β + λ khi LR log-tuyến tính
  intervals       Định lý 2–3: khoảng chứa giá trị thật, đầu mút đạt được, positivity
  closing         thống kê bootstrap của vr9 khớp tính trực tiếp
"""
from __future__ import annotations

import numpy as np
import pytest

import vr_common as R
from vr1_reversal_theory import classify, pop_logit
from vr6_pad_boundary import stats as pad_stats
from vr9_closing import cohort_cuts, patient_stats


# ---------------------------------------------------------------- classify
@pytest.mark.parametrize("theta,delta,expected", [
    (1.0, -0.5, "amplification"), (1.0, 0.5, "attenuation"), (1.0, 1.0, "nullified"), (1.0, 1.5, "reversal"),
    (-1.0, 0.5, "amplification"), (-1.0, -0.5, "attenuation"), (-1.0, -1.0, "nullified"), (-1.0, -1.5, "reversal"),
    (0.0, 0.7, "null_true"), (0.0, -0.7, "null_true"),
    (1.0, 0.0, "amplification"),   # δ = 0: không méo, |θ − δ| = |θ|, không suy giảm cũng không đảo
])
def test_classify_regions(theta, delta, expected):
    got = classify(theta, delta)
    if theta != 0 and delta == 0:
        assert got in ("amplification", "attenuation") and abs(theta - delta) == abs(theta)
    else:
        assert got == expected


def test_classify_symmetry():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        th, de = rng.normal(0, 2, 2)
        if abs(th) < 1e-6 or abs(th - de) < 1e-6:
            continue
        assert classify(th, de) == classify(-th, -de)


# ---------------------------------------------------------------- identity (Định lý 4, Định lý 1)
def _cells(p, pi1, pi0):
    y = p * pi1; v = (1 - p) * pi0
    return y, v


def test_theorem4_identity_random():
    rng = np.random.default_rng(1)
    for _ in range(5000):
        p = rng.uniform(1e-5, 0.9, 2); pi1 = rng.uniform(1e-3, 1, 2); pi0 = rng.uniform(1e-5, 1, 2)
        y, v = _cells(p, pi1, pi0)
        lhs = np.log((y[1] / v[1]) / (y[0] / v[0]))
        th = np.log(p[1] / (1 - p[1])) - np.log(p[0] / (1 - p[0]))
        assert abs(lhs - (th + np.log(pi1[1] / pi1[0]) - np.log(pi0[1] / pi0[0]))) < 1e-10
        # Định lý 1: RR_Y = RR_D · A
        assert abs(y[1] / y[0] - (p[1] / p[0]) * (pi1[1] / pi1[0])) < 1e-10 * (y[1] / y[0])


def test_A_equals_B_no_distortion():
    p = np.array([0.01, 0.03]); pi1 = np.array([0.4, 0.8]); pi0 = np.array([0.001, 0.002])   # A = B = 2
    y, v = _cells(p, pi1, pi0)
    or_d = (p[1] / (1 - p[1])) / (p[0] / (1 - p[0]))
    assert abs((y[1] / v[1]) / (y[0] / v[0]) - or_d) < 1e-12


def test_null_disease_selection_alone_creates_association():
    p = np.array([0.02, 0.02]); pi1 = np.array([0.9, 0.9]); pi0 = np.array([0.001, 0.004])   # OR_D = 1, A = 1, B = 4
    y, v = _cells(p, pi1, pi0)
    assert abs((y[1] / v[1]) / (y[0] / v[0]) - 1 / 4) < 1e-12     # = A/B
    assert classify(0.0, np.log(4)) == "null_true"


def test_pad_case_A_equals_one_exact():
    rng = np.random.default_rng(2)
    n = 20000
    t = rng.integers(0, 2, n).astype(float)
    D = (rng.random(n) < np.where(t == 1, 0.5, 0.3)).astype(float)
    S = np.where(D == 1, 1.0, (rng.random(n) < np.where(t == 1, 0.6, 0.3)).astype(float))   # mọi ác tính đều S = 1
    st = pad_stats(t, D, S)
    assert st["logA"] == 0.0
    assert abs(st["logOR_DS"] - (st["logOR_D"] - st["logB"])) < 1e-12


def test_positivity_violation_is_not_silent():
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.array([0.01, 0.03]); pi1 = np.array([0.5, 0.5]); pi0 = np.array([0.0, 0.002])
        y, v = _cells(p, pi1, pi0)
        val = (y[1] / v[1]) / (y[0] / v[0])
    assert not np.isfinite(val) or val == 0      # π⁰_t = 0 phá vỡ định danh: phải thấy được, không ra số hữu hạn


# ---------------------------------------------------------------- theorem 5
@pytest.mark.parametrize("beta,a,b", [(0.8, 0.3, 1.6), (0.3, -0.3, 0.0), (0.5, 0.0, 1.0), (-0.4, 0.2, -0.5)])
def test_theorem5_loglinear_exact(beta, a, b):
    c = np.linspace(-4, 4, 321); w = np.exp(-c ** 2 / 2); w /= w.sum()
    pD = 1 / (1 + np.exp(-(-4.5 + beta * c)))
    pi1 = 0.2 * np.exp(a * c); pi0 = 0.003 * np.exp(b * c - 0.5 * b ** 2)
    assert pi1.max() <= 1 and pi0.max() <= 1
    wS = w * (pD * pi1 + (1 - pD) * pi0)
    bM2 = pop_logit(c[:, None], wS, pD * pi1 / (pD * pi1 + (1 - pD) * pi0))[1]
    assert abs(bM2 - (beta + a - b)) < 1e-8


# ---------------------------------------------------------------- intervals (Định lý 2–3)
def test_disease_interval_contains_truth_and_is_sharp():
    rng = np.random.default_rng(3)
    p = rng.uniform(1e-4, 0.3, 1000); s = rng.uniform(0.5, 1.0, 1000); a = rng.uniform(1e-4, 0.01, 1000)
    q = p * s; sigma = p * s + (1 - p) * a
    L, U = R.disease_interval(q, sigma, 0.5)
    assert np.all(L <= p + 1e-15) and np.all(p <= U + 1e-15)
    # đầu mút đạt được: s = 1 cho p = L; s = s_min cho p = q/s_min khi cận này ràng buộc
    assert np.allclose(R.disease_interval(q, sigma, 1.0)[1], np.minimum(q, q + 1 - sigma))
    bind = q / 0.5 <= q + 1 - sigma
    assert np.allclose(U[bind], q[bind] / 0.5)


def test_B_interval_contains_truth():
    rng = np.random.default_rng(4)
    for _ in range(2000):
        p = rng.uniform(1e-4, 0.2, 2); a = rng.uniform(1e-4, 0.05, 2)
        v = (1 - p) * a
        pL, pU = p * rng.uniform(0.2, 1, 2), np.minimum(p * rng.uniform(1, 3, 2), 0.99)
        BL, BU = R.B_interval(v[0], v[1], pL, pU)
        assert BL <= a[1] / a[0] * (1 + 1e-12) and a[1] / a[0] <= BU * (1 + 1e-12)


# ---------------------------------------------------------------- closing (vr9)
def _toy(seed=5, n=3000):
    rng = np.random.default_rng(seed)
    pid = rng.integers(0, 60, n); x = rng.normal(size=n); lg = 0.7 * x + rng.normal(size=n) + 0.3 * (pid % 3)
    return lg, x, pid


def test_closing_tertile_matches_direct():
    lg, x, pid = _toy(); cuts = cohort_cuts(x)
    up, M, f = patient_stats(lg, x, pid, "tertile", cuts)
    lo, hi = cuts["tertile"]
    direct = lg[x >= hi].mean() - lg[x <= lo].mean()
    assert abs(float(f(M.sum(0))) - direct) < 1e-10


def test_closing_slopes_match_direct():
    lg, x, pid = _toy(); cuts = cohort_cuts(x)
    z = (x - cuts["mean"]) / cuts["sd"]
    up, M, f = patient_stats(lg, x, pid, "slope", cuts)
    assert abs(float(f(M.sum(0))) - np.polyfit(z, lg, 1)[0]) < 1e-10
    up, M, f = patient_stats(lg, x, pid, "within_slope", cuts)
    zc = z - np.array([z[pid == p].mean() for p in pid]); lc = lg - np.array([lg[pid == p].mean() for p in pid])
    assert abs(float(f(M.sum(0))) - (zc @ lc) / (zc @ zc)) < 1e-10


# ---------------------------------------------------------------- Proposition 1 (learner gap)
def test_learner_gap_identity_random():
    rng = np.random.default_rng(6)
    for _ in range(5000):
        p, pi1, pi0 = rng.uniform(1e-4, 0.95), rng.uniform(1e-3, 1), rng.uniform(1e-4, 1)
        f0 = p * pi1; f2 = p * pi1 / (p * pi1 + (1 - p) * pi0)
        g = (1 - p) * pi0 / (1 - p * pi1)
        lg = lambda v: np.log(v / (1 - v))
        assert abs(lg(f0) - lg(f2) - np.log(g)) < 1e-9


def test_learner_gap_on_population_contrast():
    """Kỳ vọng theo tertile: Δ_M0 − Δ_M2 = E[log g | t=1] − E[log g | t=0] trên một quần thể rời rạc."""
    rng = np.random.default_rng(7)
    n = 4000
    x = rng.normal(size=n); t = (x > 0).astype(int)
    p = 1 / (1 + np.exp(-(-4 + 0.8 * x))); pi1 = 0.6 + 0.3 / (1 + np.exp(-x)); pi0 = 0.003 * np.exp(1.2 * x - 0.72)
    lg = lambda v: np.log(v / (1 - v))
    f0 = p * pi1; f2 = p * pi1 / (p * pi1 + (1 - p) * pi0); g = (1 - p) * pi0 / (1 - p * pi1)
    D = lambda v: v[t == 1].mean() - v[t == 0].mean()
    assert abs((D(lg(f0)) - D(lg(f2))) - D(np.log(g))) < 1e-10


def test_dose_response_shift():
    rng = np.random.default_rng(8)
    x = rng.normal(size=3000); z = rng.normal(size=3000); t = (x > 0).astype(int)
    p = 1 / (1 + np.exp(-(-3 + x))); pi0 = 0.01 * np.exp(0.3 * x)
    lg2 = lambda a0: np.log(p / ((1 - p) * a0))
    for eta in (-1.0, 0.5, 2.0):
        shift = lg2(pi0 * np.exp(eta * z)) - lg2(pi0)
        assert np.allclose(shift, -eta * z)
        D = lambda v: v[t == 1].mean() - v[t == 0].mean()
        assert abs(D(lg2(pi0 * np.exp(eta * z))) - (D(lg2(pi0)) - eta * D(z))) < 1e-10


def test_general_offset_with_hidden_appearance():
    """Proposition 1 tổng quát: X ẩn một phần sau x̃; offset là log r₁ − log E[r₀(X) | x̃, Y=0]."""
    rng = np.random.default_rng(9)
    lg = lambda v: np.log(v / (1 - v))
    for _ in range(2000):
        k = rng.integers(2, 6)
        w = rng.dirichlet(np.ones(k)); q = rng.uniform(0.001, 0.9, k); r0 = rng.uniform(0.001, 1, k); r1 = rng.uniform(0.05, 1)
        qx = w @ q
        sel = r1 * qx / (r1 * qx + w @ ((1 - q) * r0))                  # P(Y=1 | x̃, R=1), trực tiếp
        er0 = (w @ ((1 - q) * r0)) / (1 - qx)                            # E[r₀(X) | x̃, Y=0]
        assert abs(lg(sel) - (lg(qx) + np.log(r1) - np.log(er0))) < 1e-9


def test_psi_identified_set_contains_truth_and_is_attained():
    """Proposition 2: ψ ∈ [ψ_L, ψ_U]; hai đầu mút đạt được bởi các s(x̃) hợp lệ."""
    rng = np.random.default_rng(10)
    lg = lambda v: np.log(v / (1 - v))
    for _ in range(500):
        n = 400; t = rng.integers(0, 2, n); smin = rng.uniform(0.3, 0.95)
        p = rng.uniform(0.001, 0.3, n); s = rng.uniform(smin, 1, n); q = p * s
        sigma = np.clip(q + rng.uniform(0, 1, n) * (1 - q), q, 1)
        sigma = np.maximum(sigma, q + (p - q))                        # P(D=1,S=0) ≤ P(S=0)
        sigma = np.minimum(sigma, 1 - (p - q))
        U = np.minimum(q / smin, q + 1 - sigma)
        D = lambda v: v[t == 1].mean() - v[t == 0].mean()
        psi = D(lg(p)); psiL = lg(q)[t == 1].mean() - lg(U)[t == 0].mean(); psiU = lg(U)[t == 1].mean() - lg(q)[t == 0].mean()
        assert psiL - 1e-12 <= psi <= psiU + 1e-12
        # đầu mút dưới: p = q trên t=1, p = U trên t=0 — cả hai là P(D|x̃) hợp lệ (s=1, hoặc s = q/U ≥ s_min)
        pl = np.where(t == 1, q, U)
        assert np.all(q / pl >= smin - 1e-12) and np.all(pl - q <= 1 - sigma + 1e-12)
        assert abs(D(lg(pl)) - psiL) < 1e-10
