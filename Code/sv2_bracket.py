#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SV2 — Khung chặn (bracket) cho tương phản BỆNH dưới xác minh theo giai đoạn.

Mô hình. Với tertile proxy t (trong một tầng Z hoặc trong một bệnh nhân), đặt
  p_t = Pr(D=1 | t),  π_t^1 = Pr(S=1 | D=1, t),  π_t^0 = Pr(S=1 | D=0, t),
  A = π_1^1/π_0^1  (tỉ số xác minh ÁC TÍNH, chính là π₁/π₀ của v3 Statement 3),
  B = π_1^0/π_0^0  (tỉ số xác minh LÀNH TÍNH).
Hai đồng nhất thức chính xác:
  (I1)  RR_Y     = RR_D · A                (nhãn ghi nhận, toàn cohort)
  (I2)  OR_{D|S} = OR_D · A / B            (chỉ tổn thương đã xác minh)
và B quan sát được, vì Pr(V=1 | t) = (1 − p_t)·π_t^0 với V = S·(1−Y), còn (1−p_t) ≈ 1.
Vậy toàn bộ tập định danh của RR_D là {RR_Y / A}, một tham số A.

Giả định SV-M (tính kẹp giữa). A nằm giữa 1 và B. Khi đó
  RR_D ∈ [ min(RR_Y, RR_Y/B), max(RR_Y, RR_Y/B) ],
đầu mút RR_Y ứng với xác minh ác tính không phân biệt theo proxy (A=1, giả định ngầm của v3), đầu mút
RR_Y/B ≈ OR_{D|S} ứng với việc xác minh không mang thông tin bệnh ngoài ngoại hình (A=B, giả định ngầm
của mọi bộ dữ liệu chỉ-gồm-đã-xác-minh). SV-M được suy ra (Mệnh đề 1 trong proposal) từ mô hình
HAI-ĐƯỜNG: một tổn thương được xác minh nếu đường "ngoại hình" (xác suất a_t, mù với D) HOẶC đường
"bệnh" (xác suất κ, chỉ với D=1, không phụ thuộc t) kích hoạt. Khi đó π_t^0 = a_t,
π_t^1 = κ + (1−κ)a_t, và A = (κ + (1−κ)a_1)/(κ + (1−κ)a_0) đơn điệu từ B (κ=0) về 1 (κ=1).

Hệ quả định lượng. a_t ~ 10⁻³ nên A ≈ 1 + (a_1 − a_0)/s_0 với s_0 = độ nhạy xác minh ác tính ở tertile
dưới. Để A đạt RR_Y (xoá sạch tương phản) cần s_0 ≤ s_0*, tính ở đây. Nới giả định bằng λ = κ_1/κ_0
(đường bệnh phụ thuộc ngoại hình), λ cần thiết để xoá tương phản được báo cáo trên lưới s_0.

Hai thang: hiệu chỉnh Z (logistic, OR có điều kiện, IF theo bệnh nhân cho hiệp phương sai chung) và
trong-bệnh-nhân (conditional logistic, như v3 c41). Khoảng tin cậy của tập định danh là hợp của hai
khoảng đầu mút 95% (bảo thủ).

Out -> Result/sv2_bracket[_noindet].json
Chạy:  python3 sv2_bracket.py [--drop-indeterminate] [--skip-within]
"""
from __future__ import annotations

import argparse

import numpy as np

import sv_common as C

Z95 = 1.959964
S0_GRID = [0.05, 0.1, 0.3, 0.5, 0.8]


def verdict(lo_ci, hi_ci):
    if lo_ci > 1:
        return "positive"
    if hi_ci < 1:
        return "negative"
    return "undetermined"


def bracket_from(thY, thV, cov, thDS=None, seDS=None):
    """thY = log OR_Y, thV = log B; cov = Cov(thY, thV). Trả hai đầu mút và tập định danh."""
    U, L = thY, thY - thV
    seU = np.sqrt(cov[0, 0]); seL = np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    ciU, ciL = C.ci(U, seU), C.ci(L, seL)
    lo, hi = min(U, L), max(U, L)
    set_ci = [min(ciU[0], ciL[0]), max(ciU[1], ciL[1])]
    r = {
        "OR_Y": float(np.exp(thY)), "OR_Y_ci": ciU,
        "B": float(np.exp(thV)), "B_ci": C.ci(thV, np.sqrt(cov[1, 1])),
        "endpoint_A1": float(np.exp(U)), "endpoint_A1_ci": ciU,
        "endpoint_AB": float(np.exp(L)), "endpoint_AB_ci": ciL,
        "identified_set": [float(np.exp(lo)), float(np.exp(hi))],
        "identified_set_ci": set_ci,
        "sign_verdict": verdict(set_ci[0], set_ci[1]),
        "point_sign_verdict": verdict(np.exp(lo), np.exp(hi)),
        "corr_Y_V": float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])),
    }
    if thDS is not None:
        r["OR_D_given_S_direct"] = float(np.exp(thDS))
        r["OR_D_given_S_direct_ci"] = C.ci(thDS, seDS)
    return r


def two_route(R, a0, a1):
    """Mô hình hai đường với RR ghi nhận R và tỉ lệ xác minh lành tính tuyệt đối a0, a1."""
    out = {"a0": a0, "a1": a1, "B_abs": a1 / a0}
    # κ* sao cho A(κ*) = R; tồn tại khi B > R > 1 (hoặc ngược chiều khi B < R < 1)
    den = R - 1 + a1 - R * a0
    k = (a1 - R * a0) / den if den != 0 else np.nan
    if np.isfinite(k) and 0 < k < 1:
        out["kappa_star"] = float(k); out["s0_star"] = float(k + (1 - k) * a0)
    else:
        out["kappa_star"] = None; out["s0_star"] = None   # đường ngoại hình không thể xoá được
    grid = []
    for s0 in S0_GRID:
        k0 = (s0 - a0) / (1 - a0)
        A_app = (k0 + (1 - k0) * a1) / s0                     # λ = 1
        lam = (R * s0 - a1) / (k0 * (1 - a1))                  # λ cần để A = R
        grid.append({"s0": s0, "A_appearance_route": float(A_app),
                     "RR_D_under_two_route": float(R / A_app),
                     "lambda_to_explain_away": float(lam),
                     "lambda_feasible": bool(lam * k0 <= 1)})
    out["grid"] = grid
    return out


def adjusted_scale(df, T, m):
    d = df[m]; Tm = T[m].to_numpy()
    Z = C.build_Z(d)
    X = np.column_stack([np.ones(len(d)), Tm, Z.to_numpy()])
    g = d[C.GROUP].to_numpy()
    bY, IFY, _ = C.logit_if(X, d.Y.to_numpy(), g)
    bV, IFV, _ = C.logit_if(X, d.V.to_numpy(), g)
    cov = C.joint_cov([IFY, IFV], [1, 1])
    # tỉ số trực tiếp trên mẫu đã xác minh (kiểm tra chéo đầu mút A=B)
    s = d[d.S == 1]
    Zs = C.build_Z(s, min_count=20)
    Xs = np.column_stack([np.ones(len(s)), Tm[(d.S == 1).to_numpy()], Zs.to_numpy()])
    bS, IFS, _ = C.logit_if(Xs, s.Y.to_numpy(), s[C.GROUP].to_numpy())
    seS = float(np.sqrt(C.joint_cov([IFS], [1])[0, 0]))
    r = bracket_from(bY[1], bV[1], cov, bS[1], seS)
    # tỉ lệ tuyệt đối chuẩn hoá (g-formula) cho mô hình hai đường
    def std_rate(b, t):
        Xt = X.copy(); Xt[:, 1] = t
        return float(np.mean(1 / (1 + np.exp(-(Xt @ b)))))
    a0, a1 = std_rate(bV, 0), std_rate(bV, 1)
    y0, y1 = std_rate(bY, 0), std_rate(bY, 1)
    r["standardized"] = {"pY0": y0, "pY1": y1, "RR_Y_std": y1 / y0, "pV0": a0, "pV1": a1}
    r["two_route"] = two_route(y1 / y0, a0, a1)
    return r


def within_scale(df, T, m):
    d = df[m]; Tm = T[m].to_numpy()
    res = {}
    fits = {}
    for key, sub, outcome in (("Y", np.ones(len(d), bool), "Y"),
                              ("V", np.ones(len(d), bool), "V"),
                              ("DS", (d.S == 1).to_numpy(), "Y")):
        X, y, strata, pids = C.clogit_prepare(d[sub], Tm[sub], outcome)
        b, IF, ns, nc = C.clogit_if(X, y, strata, pids)
        fits[key] = (b, IF)
        res[f"strata_{key}"] = ns; res[f"cases_{key}"] = nc
        C.log(f"    clogit {key}: OR={np.exp(b[0]):.3f} strata={ns} cases={nc}")
    cov = C.joint_cov([fits["Y"][1], fits["V"][1]], [0, 0])
    seDS = float(np.sqrt(C.joint_cov([fits["DS"][1]], [0])[0, 0]))
    r = bracket_from(fits["Y"][0][0], fits["V"][0][0], cov, fits["DS"][0][0], seDS)
    r.update(res)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-indeterminate", action="store_true")
    ap.add_argument("--skip-within", action="store_true")
    a = ap.parse_args()
    df = C.load()
    if a.drop_indeterminate:
        df = df[df.indet == 0].reset_index(drop=True)
    C.log(f"rows={len(df)} Y={df.Y.sum()} V={df.V.sum()}")
    out = {"drop_indeterminate": a.drop_indeterminate, "s0_grid": S0_GRID, "proxies": {}}
    for p in C.PROXIES:
        T = C.tertile(df, p["col"]); m = T.notna().to_numpy()
        C.log(f"{p['name']}")
        r = {"column": p["col"], "role": p["role"], "adjusted": adjusted_scale(df, T, m)}
        ad = r["adjusted"]
        C.log(f"  adjusted: OR_Y={ad['OR_Y']:.2f} B={ad['B']:.2f} set=[{ad['identified_set'][0]:.2f},"
              f" {ad['identified_set'][1]:.2f}] ci={np.round(ad['identified_set_ci'], 2)} -> {ad['sign_verdict']}"
              f" | direct D|S={ad['OR_D_given_S_direct']:.2f} | s0*={ad['two_route']['s0_star']}")
        if not a.skip_within:
            r["within_patient"] = within_scale(df, T, m)
            w = r["within_patient"]
            C.log(f"  within:   OR_Y={w['OR_Y']:.2f} B={w['B']:.2f} set=[{w['identified_set'][0]:.2f},"
                  f" {w['identified_set'][1]:.2f}] ci={np.round(w['identified_set_ci'], 2)} -> {w['sign_verdict']}")
        out["proxies"][p["name"]] = r
    C.save_json(out, "sv2_bracket_noindet.json" if a.drop_indeterminate else "sv2_bracket.json")


if __name__ == "__main__":
    main()
