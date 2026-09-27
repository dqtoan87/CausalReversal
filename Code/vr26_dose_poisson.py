#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR26 — Liều chọn mẫu với xác suất đưa vào BIẾT CHÍNH XÁC (lấy mẫu Poisson), cho mọi ô concept × họ chính.

Mỗi nhãn âm ghi nhận i trong pool được đưa vào độc lập với xác suất π_i(η) = min(1, κ_η exp(η z_ik)), κ_η chọn để
kỳ vọng số lượng bằng n. Khi đó r₀ của Proposition 1 chính là π(η) và không cần hiệu chỉnh xấp xỉ nào.
  Dự đoán (tabular, z_k trong input): Δ_j(η) − Δ_j(0) = −(E[log π_η | T_j=1] − E[log π_η | T_j=0]) trên test.
  Dự đoán (ảnh): log E[π_η(Z) | x̃, Y=0] ≈ log κ_η + η m(x̃) + η² v(x̃)/2 (xấp xỉ chuẩn, bậc một: chỉ m; bậc hai: thêm
  v(x̃) = Var(z | x̃) ước lượng bằng ridge trên bình phương phần dư). m, v khớp lại trên bệnh nhân train lấy lại ở mỗi
  lần lặp, nên khoảng của dự đoán phản ánh sai số nuisance.
(0) Với thiết kế cũ (Gumbel top-k, không hoàn lại), xác suất đưa vào thực tế xấp xỉ π_i ≈ 1 − exp(−λ w_i) (lấy mẫu kế
    tiếp); báo độ lệch log π_i − log(κ w_i) lớn nhất và độ dốc dự đoán khi dùng π thực, kiểm lại bằng Monte Carlo theo tertile.
(1) Đường học: color variegation với n ∈ {320, 1000, 3000} nhãn âm để xem độ dốc có tiến về dự đoán khi n tăng.
Mọi learner hiệu chỉnh Platt trên val chọn theo cùng quy tắc. Độ dốc khớp trên η ∈ [−1, 1].

Out -> Result/vr26_dose_poisson.json
"""
from __future__ import annotations

import os

import numpy as np
from scipy.optimize import brentq

import vr_common as R
import gpu_setup  # noqa: F401
from vr21_dose_calibrated import features, _fit, TARGETS, N_MAL

ETAS = [-1.0, -0.5, 0.0, 0.5, 1.0]
N_REP = 10
SIZES = [320, 1000, 3000]


def poisson_probs(w, n):
    k = brentq(lambda lk: np.minimum(1.0, np.exp(lk) * w).sum() - n, -60, 60)
    return np.minimum(1.0, np.exp(k) * w), np.exp(k)


def main():
    import vl_models as M
    from sklearn.linear_model import Ridge
    df, fold = R.load_isic()
    Y = df.Y.to_numpy()
    Zs = {c: ((df[col] - df[col].mean()) / df[col].std()).fillna(0.0).to_numpy() for c, col in R.CONCEPTS.items()}
    te = np.flatnonzero(fold == "test")
    Tc = {c: R.SV.tertile(df, col).fillna(-1).to_numpy()[te] for c, col in R.CONCEPTS.items()}
    D = lambda v, j: float(v[Tc[j] == 1].mean() - v[Tc[j] == 0].mean())
    pools = {p: {"mal": np.flatnonzero((fold == p) & (Y == 1)), "neg": np.flatnonzero((fold == p) & (Y == 0))} for p in ("train", "val")}
    nva_m = int(round(0.8 * len(pools["val"]["mal"])))
    tr_all, va_all = np.flatnonzero(fold == "train"), np.flatnonzero(fold == "val")
    pid = df[R.SV.GROUP].to_numpy()
    neg = pools["train"]["neg"]; neg_p = np.unique(pid[neg]); neg_by = {p: neg[pid[neg] == p] for p in neg_p}
    out = {"etas": ETAS, "gumbel_check": {}, "families": {}}
    # (0) thiết kế Gumbel top-k cũ: xác suất đưa vào thực tế
    rng0 = np.random.default_rng(1)
    for k in TARGETS:
        rows = {}
        for eta in ETAS:
            w = np.exp(eta * Zs[k][neg]); n = 320
            lam = brentq(lambda l: (1 - np.exp(-l * w)).sum() - n, 1e-12, 1e3)
            pi_true = 1 - np.exp(-lam * w); pi_nom = n * w / w.sum()
            dev = np.log(pi_true) - np.log(pi_nom)
            # Monte Carlo: tỉ lệ đưa vào theo tertile của concept, so với π thực và π danh nghĩa
            Tn = R.SV.tertile(df, R.CONCEPTS[k]).fillna(-1).to_numpy()[neg]
            cnt = np.zeros(len(neg))
            for _ in range(300):
                key = np.log(w) - np.log(-np.log(rng0.random(len(neg))))
                cnt[np.argpartition(-key, n)[:n]] += 1
            mc = {t: float(cnt[Tn == t].sum() / 300) for t in (0, 1)}
            rows[str(eta)] = {"max_abs_log_dev": float(np.abs(dev).max()), "p99_abs_log_dev": float(np.percentile(np.abs(dev), 99)),
                              "expected_upper_true": float(pi_true[Tn == 1].sum()), "expected_upper_nominal": float(pi_nom[Tn == 1].sum()),
                              "mc_upper": mc[1], "expected_lower_true": float(pi_true[Tn == 0].sum()),
                              "expected_lower_nominal": float(pi_nom[Tn == 0].sum()), "mc_lower": mc[0],
                              "pred_contrast_true": -D(np.log(1 - np.exp(-lam * np.exp(eta * Zs[k][te]))), k),
                              "pred_contrast_nominal": -eta * D(Zs[k][te], k)}
        out["gumbel_check"][k] = rows
        R.log(f"gumbel {k}: η=1 max|log dev| {rows['1.0']['max_abs_log_dev']:.3f}, pred true {rows['1.0']['pred_contrast_true']:+.3f} vs nominal {rows['1.0']['pred_contrast_nominal']:+.3f}")
    R.save_json(out, "vr26_dose_poisson.json")
    for kind in ("tabular", "image"):
        X, hid = features(df, kind)
        fam = {"dose": {}, "learning_curve": {}}

        def run_cell(k, n_ben, seed_base):
            obs, pred1, pred2 = [], [], []
            for r in range(N_REP):
                rng = np.random.default_rng(seed_base + 1000 * r)
                mtr = rng.choice(pools["train"]["mal"], N_MAL, replace=False); mva = rng.choice(pools["val"]["mal"], nva_m, replace=False)
                nva_b = int(round(111 * n_ben / 320))
                if kind == "image":         # nuisance m, v khớp lại trên bệnh nhân train lấy lại
                    bp = rng.choice(neg_p, len(neg_p)); ix = np.concatenate([neg_by[p] for p in bp])
                    ix = rng.choice(ix, min(60000, len(ix)), replace=False); h = len(ix) // 2
                    rm = Ridge(alpha=10.0).fit(X[ix[:h]], Zs[k][ix[:h]])
                    res2 = (Zs[k][ix[h:]] - rm.predict(X[ix[h:]])) ** 2
                    rv = Ridge(alpha=10.0).fit(X[ix[h:]], res2)
                    m_te, v_te = rm.predict(X[te]), np.maximum(rv.predict(X[te]), 0)
                row_o, row_1, row_2 = [], [], []
                for eta in ETAS:
                    ptr, _ = poisson_probs(np.exp(eta * Zs[k][pools["train"]["neg"]]), n_ben)
                    pva, _ = poisson_probs(np.exp(eta * Zs[k][pools["val"]["neg"]]), nva_b)
                    btr = pools["train"]["neg"][rng.random(len(ptr)) < ptr]; bva = pools["val"]["neg"][rng.random(len(pva)) < pva]
                    _, cal = _fit(M, df, X, hid, np.concatenate([mtr, btr]), np.concatenate([mva, bva]), tr_all, va_all, te, r)
                    row_o.append({j: D(cal, j) for j in R.CONCEPTS})
                    if kind == "tabular":
                        _, kap = poisson_probs(np.exp(eta * Zs[k][pools["train"]["neg"]]), n_ben)
                        lpi = np.log(np.minimum(1.0, kap * np.exp(eta * Zs[k][te])))
                        row_1.append({j: -D(lpi, j) for j in R.CONCEPTS}); row_2.append(row_1[-1])
                    else:
                        row_1.append({j: -eta * D(m_te, j) for j in R.CONCEPTS})
                        row_2.append({j: -(eta * D(m_te, j) + eta ** 2 / 2 * D(v_te, j)) for j in R.CONCEPTS})
                obs.append(row_o); pred1.append(row_1); pred2.append(row_2)
            e = np.array(ETAS)
            sl = lambda rows, j: np.array([np.polyfit(e, [x[j] for x in rw], 1)[0] for rw in rows])
            cell = {}
            for j in R.CONCEPTS:
                so, s1, s2 = sl(obs, j), sl(pred1, j), sl(pred2, j)
                cell[j] = {"obs_slope": float(so.mean()), "obs_range95": np.percentile(so, [2.5, 97.5]).tolist(),
                           "pred_slope": float(s1.mean()), "pred_range95": np.percentile(s1, [2.5, 97.5]).tolist(),
                           "pred2_slope": float(s2.mean()), "pred2_range95": np.percentile(s2, [2.5, 97.5]).tolist(),
                           "ratio": float(so.mean() / s1.mean()) if s1.mean() != 0 else None}
            mean = np.mean([[x[k] for x in rw] for rw in obs], 0)
            cross = None
            for i in range(len(e) - 1):
                if mean[i] * mean[i + 1] <= 0 and mean[i] != mean[i + 1]:
                    cross = float(e[i] - mean[i] * (e[i + 1] - e[i]) / (mean[i + 1] - mean[i])); break
            return {"concepts": cell, "mean_by_eta": mean.tolist(), "q025": np.percentile([[x[k] for x in rw] for rw in obs], 2.5, 0).tolist(),
                    "q975": np.percentile([[x[k] for x in rw] for rw in obs], 97.5, 0).tolist(), "crossing": cross,
                    "spillover_sign_agreement": int(sum(np.sign(cell[j]["obs_slope"]) == np.sign(cell[j]["pred_slope"]) for j in R.CONCEPTS))}

        for k in TARGETS:
            fam["dose"][k] = run_cell(k, 320, 11 + TARGETS.index(k))
            c = fam["dose"][k]["concepts"][k]
            R.log(f"{kind}|{k}: obs {c['obs_slope']:+.2f} {np.round(c['obs_range95'], 2)} pred {c['pred_slope']:+.2f} "
                  f"{np.round(c['pred_range95'], 2)} pred2 {c['pred2_slope']:+.2f} ratio {c['ratio']:.2f} cross {fam['dose'][k]['crossing']}")
            out["families"][kind] = fam; R.save_json(out, "vr26_dose_poisson.json")
        for n in SIZES[1:]:
            cc = run_cell("color_variegation", n, 77 + n)
            fam["learning_curve"][str(n)] = cc
            c = cc["concepts"]["color_variegation"]
            R.log(f"{kind} learning curve n={n}: obs {c['obs_slope']:+.2f} pred {c['pred_slope']:+.2f} ratio {c['ratio']:.2f}")
            out["families"][kind] = fam; R.save_json(out, "vr26_dose_poisson.json")
        fam["learning_curve"]["320"] = fam["dose"]["color_variegation"]
        out["families"][kind] = fam
        R.save_json(out, "vr26_dose_poisson.json")


if __name__ == "__main__":
    main()
