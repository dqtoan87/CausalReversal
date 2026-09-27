#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR25 — Độ nhạy của tập định danh ψ_k (Proposition 2) theo hiệu chỉnh, nuisance và vùng hỗ trợ, trên các lần chạy gốc.

Với mỗi họ chính (tabular, ảnh đóng băng) và seed 0–2 (không lấy lại):
  (a) q̂ theo năm cách: M0 thô (không hiệu chỉnh), Platt, isotonic, beta calibration (hồi quy logistic trên
      [log p, −log(1 − p)]), và một ước lượng q khác trên cùng input (gradient boosting; ảnh: 64 thành phần chính),
      tất cả hiệu chỉnh/khớp không dùng test;
  (b) σ̂ theo hai cách: nuisance tabular hiện dùng, và một MLP học S trên chính input của learner;
  (c) cắt đuôi: winsorize logit q̂ và logit U tại phân vị 1 và 99 của test;
  (d) hai sàn: s_low áp cho vùng hỗ trợ thấp (σ̂ dưới ngưỡng phân vị 1% hoặc 5% trên train đã xác minh), s_high cho
      phần còn lại; lưu lưới ψ_L(s_low, s_high) bước 0.05 và biên s_low nhỏ nhất cho ψ_L > 0 theo từng s_high.
s* luôn tính trên đường ψ_L trung bình qua ba seed.

Out -> Result/vr25_psi_sensitivity.json
"""
from __future__ import annotations

import os

import numpy as np

import vr_common as R
import gpu_setup  # noqa: F401
from vr19_primary_bootstrap import features, arrays, platt_fit, S_FINE, PRIMARY
from vr24_full_bootstrap import psiL_curve, iso_fit, tip

GRID2 = np.round(np.arange(0.05, 1.0001, 0.05), 2)


def psiL_two(q, sig, T, low, s_lo, s_hi):
    s = np.where(low, s_lo, s_hi)
    lq = R.logit(np.clip(q, 1e-7, 1 - 1e-7)); U = np.clip(np.minimum(q / s, q + 1 - sig), 1e-7, 1 - 1e-7)
    return lq[T == 1].mean() - R.logit(U)[T == 0].mean()


def psiL_wins(q, sig, T):
    lq = R.logit(np.clip(q, 1e-7, 1 - 1e-7)); lo, hi = np.percentile(lq, [1, 99]); lqw = np.clip(lq, lo, hi)
    out = []
    for s in S_FINE:
        lU = R.logit(np.clip(np.minimum(q / s, q + 1 - sig), 1e-7, 1 - 1e-7)); a, b = np.percentile(lU, [1, 99])
        out.append(lqw[T == 1].mean() - np.clip(lU, a, b)[T == 0].mean())
    return np.array(out)


def main():
    import vl_models as M
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.decomposition import PCA
    df, fold = R.load_isic()
    Y, S = df.Y.to_numpy(), df.S.to_numpy()
    tr, va, te = (np.flatnonzero(fold == k) for k in ("train", "val", "test"))
    sig_tab = np.load(os.path.join(R.MISS_RES, "vl_nuisance.npz"))["sigma"]
    trS = tr[S[tr] == 1]
    low_masks = {q: sig_tab[te] < np.quantile(sig_tab[trS], q) for q in (0.01, 0.05)}
    Tt = {c: R.SV.tertile(df, R.CONCEPTS[c]).fillna(-1).to_numpy()[te] for c in PRIMARY}
    out = {"grid": S_FINE.tolist(), "grid2": GRID2.tolist(), "low_share": {str(k): float(v.mean()) for k, v in low_masks.items()}, "families": {}}
    for fam in ("tabular", "image"):
        X, hid = features(df, fam)
        A = arrays(df, X)
        # σ̂ trên input của learner (MLP học S), hiệu chỉnh Platt trên val
        As = dict(A); As["Y"] = S.astype(np.float32)
        curves = {c: {k: [] for k in ("raw", "platt", "iso", "beta", "altq", "sigma_learner", "winsor")} for c in PRIMARY}
        two = {c: {str(q): [] for q in low_masks} for c in PRIMARY}
        # ước lượng q khác trên cùng input: gradient boosting (ảnh: 64 thành phần chính), hiệu chỉnh Platt trên val
        Xg = X if fam == "tabular" else PCA(64, random_state=0).fit(X[tr[::4]]).transform(X).astype(np.float32)
        gb = HistGradientBoostingClassifier(max_depth=4, max_iter=400, learning_rate=0.05, random_state=0).fit(Xg[tr], Y[tr])
        lgv, lgt = R.logit(np.clip(gb.predict_proba(Xg[va])[:, 1], 1e-7, 1)), R.logit(np.clip(gb.predict_proba(Xg[te])[:, 1], 1e-7, 1))
        a, b = platt_fit(lgv, Y[va]); q_alt = 1 / (1 + np.exp(-(a * lgt + b)))
        for seed in range(3):
            m0 = M.train("erm_y", A, tr, va, hid=hid, seed=seed)
            pv, pt = M.predict(m0, X[va])["pD"], M.predict(m0, X[te])["pD"]
            a, b = platt_fit(R.logit(pv), Y[va]); q_pl = 1 / (1 + np.exp(-(a * R.logit(pt) + b)))
            q_iso = np.clip(iso_fit(pv, Y[va]).predict(pt), 1e-6, 1 - 1e-6)
            fb = lambda p: np.column_stack([np.log(np.clip(p, 1e-7, 1)), -np.log(np.clip(1 - p, 1e-7, 1))])
            lb = LogisticRegression(C=1e6, max_iter=3000).fit(fb(pv), Y[va]); q_beta = lb.predict_proba(fb(pt))[:, 1]
            ms = M.train("erm_y", As, tr, va, hid=hid, seed=seed)
            sv, st = M.predict(ms, X[va])["pD"], M.predict(ms, X[te])["pD"]
            a, b = platt_fit(R.logit(sv), S[va]); sig_l = 1 / (1 + np.exp(-(a * R.logit(st) + b)))
            sg = sig_tab[te]
            for c in PRIMARY:
                T = Tt[c]
                curves[c]["raw"].append(psiL_curve(pt, sg, T)); curves[c]["platt"].append(psiL_curve(q_pl, sg, T))
                curves[c]["iso"].append(psiL_curve(q_iso, sg, T)); curves[c]["beta"].append(psiL_curve(q_beta, sg, T))
                curves[c]["altq"].append(psiL_curve(q_alt, sg, T)); curves[c]["sigma_learner"].append(psiL_curve(q_pl, np.maximum(sig_l, q_pl), T))
                curves[c]["winsor"].append(psiL_wins(q_pl, sg, T))
                for qn, low in low_masks.items():
                    two[c][str(qn)].append(np.array([[psiL_two(q_pl, sg, T, low, sl, sh) for sh in GRID2] for sl in GRID2]))
            R.log(f"{fam} seed {seed}: color s* raw {tip(curves['color_variegation']['raw'][-1])} platt {tip(curves['color_variegation']['platt'][-1])} "
                  f"iso {tip(curves['color_variegation']['iso'][-1])} beta {tip(curves['color_variegation']['beta'][-1])}")
        fo = {}
        for c in PRIMARY:
            e = {"s_star": {k: tip(np.mean(v, 0)) for k, v in curves[c].items()},
                 "psiL_at": {k: {str(s): float(np.mean(v, 0)[S_FINE.tolist().index(s)]) for s in (0.5, 0.7, 0.9)} for k, v in curves[c].items()},
                 "two_floor": {}}
            for qn, arrs in two[c].items():
                g = np.mean(arrs, 0)          # [s_low, s_high]
                bound = {}
                for j, sh in enumerate(GRID2):
                    ok = np.flatnonzero(g[:, j] > 0)
                    bound[str(sh)] = float(GRID2[ok[0]]) if len(ok) else None
                e["two_floor"][qn] = {"boundary_s_low_by_s_high": bound, "grid": np.round(g, 4).tolist()}
            fo[c] = e
            R.log(f"{fam} {c}: s* {e['s_star']}; two-floor(1%) s_low needed at s_high=0.9: {e['two_floor']['0.01']['boundary_s_low_by_s_high']['0.9']}")
        out["families"][fam] = fo
        R.save_json(out, "vr25_psi_sensitivity.json")


if __name__ == "__main__":
    main()
