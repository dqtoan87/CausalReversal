#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR24 — Bootstrap toàn bộ quy trình học: lấy lại bệnh nhân train, val VÀ test độc lập trong từng split (500 lần mỗi họ).

Khác vr19 ở chỗ tập val (dùng cho dừng sớm và hiệu chỉnh Platt) cũng được lấy lại theo bệnh nhân, nên khoảng tin cậy
không còn điều kiện trên tập val đã thực hiện. Mỗi lần lặp lưu:
  (1) Δ_M0, Δ_M2 thô cho năm concept;
  (2) toàn bộ đường ψ_L(s) trên lưới s = 0.01..1 cho ba concept chính, với q̂ = M0 hiệu chỉnh Platt và (độ nhạy)
      hiệu chỉnh isotonic, cả hai trên val đã lấy lại;
  (3) đường ψ_L(s) Platt trên các quần thể test bị giới hạn vào vùng hỗ trợ (σ̂ ≥ phân vị 1/5/10% trên train đã xác minh).
Các lần chạy gốc b = −1, −2, −3 (seed 0–2, không lấy lại) lưu cùng các đại lượng, để ước lượng điểm của s* được tính
từ đường ψ_L TRUNG BÌNH qua seed (cùng quy tắc tổng hợp với bảng ψ).

Chạy:  python3 vr24_full_bootstrap.py --family tabular --start -3 --end 500 ; ... --summarize
Out -> Result/vr24/{family}_{start}_{end}.json ; Result/vr24_full_bootstrap.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

import vr_common as R
import gpu_setup  # noqa: F401  (ép GPU, giới hạn luồng)
from vr19_primary_bootstrap import features, arrays, platt_fit, S_FINE, SUPPORT_Q, PRIMARY

OUTD = os.path.join(R.RES, "vr24")
ALPHA_B = 0.05 / 6


def psiL_curve(q, sig, T):
    lq = R.logit(np.clip(q, 1e-7, 1 - 1e-7)); m1, m0 = T == 1, T == 0
    a = lq[m1].mean(); out = []
    for s in S_FINE:
        U = np.clip(np.minimum(q / s, q + 1 - sig), 1e-7, 1 - 1e-7)
        out.append(a - R.logit(U)[m0].mean())
    return np.array(out)


def iso_fit(p_val, y_val):
    from sklearn.isotonic import IsotonicRegression
    return IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(p_val, y_val)


def one_rep(M, df, X, hid, A, tr, va, te, seed, sig, Tfull, sup_masks):
    Y, S = df.Y.to_numpy(), df.S.to_numpy()
    va2 = va[S[va] == 1]
    m2 = M.train("erm_verified", A, tr, va, hid=hid, seed=seed)
    m0 = M.train("erm_y", A, tr, va, hid=hid, seed=seed)
    pv0 = M.predict(m0, X[va])["pD"]
    a0, b0 = platt_fit(R.logit(pv0), Y[va])
    iso = iso_fit(pv0, Y[va])
    a2, b2 = platt_fit(R.logit(M.predict(m2, X[va2])["pD"]), Y[va2])
    ute, inv = np.unique(te, return_inverse=True)
    p0 = M.predict(m0, X[ute])["pD"][inv]
    l0 = R.logit(p0); l2 = R.logit(M.predict(m2, X[ute])["pD"])[inv]
    q_pl = 1 / (1 + np.exp(-(a0 * l0 + b0))); q_iso = np.clip(iso.predict(p0), 1e-6, 1 - 1e-6)
    sg = sig[te]
    out = {"platt": [a0, a2], "concepts": {}}
    for c, Tf in Tfull.items():
        T = Tf[te]
        r = {"d0": float(l0[T == 1].mean() - l0[T == 0].mean()), "d2": float(l2[T == 1].mean() - l2[T == 0].mean())}
        if c in PRIMARY:
            r["psiL_platt"] = np.round(psiL_curve(q_pl, sg, T), 4).tolist()
            r["psiL_iso"] = np.round(psiL_curve(q_iso, sg, T), 4).tolist()
            r["psiL_support"] = {}
            for qn, msk in sup_masks.items():
                m = msk[te]
                r["psiL_support"][qn] = np.round(psiL_curve(q_pl[m], sg[m], T[m]), 4).tolist()
        out["concepts"][c] = r
    return out


def run(family, start, end):
    import vl_models as M
    df, fold = R.load_isic()
    X, hid = features(df, family)
    A = arrays(df, X)
    S = df.S.to_numpy()
    tr_all, va_all, te_all = (np.flatnonzero(fold == k) for k in ("train", "val", "test"))
    sig = np.load(os.path.join(R.MISS_RES, "vl_nuisance.npz"))["sigma"]
    thr = {str(qq): float(np.quantile(sig[tr_all[S[tr_all] == 1]], qq)) for qq in SUPPORT_Q}
    sup_masks = {k: sig >= v for k, v in thr.items()}
    Tfull = {c: R.SV.tertile(df, col).fillna(-1).to_numpy() for c, col in R.CONCEPTS.items()}
    pid = df[R.SV.GROUP].to_numpy()
    by = {}
    for name, ix in (("tr", tr_all), ("va", va_all), ("te", te_all)):
        ps = np.unique(pid[ix]); by[name] = (ps, {p: ix[pid[ix] == p] for p in ps})
    os.makedirs(OUTD, exist_ok=True)
    path = os.path.join(OUTD, f"{family}_{start}_{end}.json")
    res = json.load(open(path)) if os.path.exists(path) else {"family": family, "support_thresholds": thr, "reps": {}}
    for b in range(start, end):
        if str(b) in res["reps"]:
            continue
        if b < 0:
            tr, va, te = tr_all, va_all, te_all
        else:
            rng = np.random.default_rng(80_000 + b)
            tr, va, te = (np.concatenate([by[k][1][p] for p in rng.choice(by[k][0], len(by[k][0]))]) for k in ("tr", "va", "te"))
        res["reps"][str(b)] = one_rep(M, df, X, hid, A, tr, va, te, (-b - 1) if b < 0 else b, sig, Tfull, sup_masks)
        json.dump(res, open(path + ".tmp", "w")); os.replace(path + ".tmp", path)
        if b % 25 == 0:
            c = res["reps"][str(b)]["concepts"]["color_variegation"]
            R.log(f"{family} rep {b}: color d0 {c['d0']:+.2f} d2 {c['d2']:+.2f}")


def tip(curve):
    pos = np.flatnonzero(np.asarray(curve) > 0)
    return float(S_FINE[pos[0]]) if len(pos) else np.nan


def summarize():
    out = {"B": {}, "grid": S_FINE.tolist(), "families": {}}
    for fam in ("tabular", "image"):
        reps, origs, thr = [], [], None
        for f in glob.glob(os.path.join(OUTD, f"{fam}_*.json")):
            d = json.load(open(f)); thr = d["support_thresholds"]
            for k, v in d["reps"].items():
                (origs if int(k) < 0 else reps).append(v)
        B = len(reps); out["B"][fam] = B
        pct = lambda v, a: [float(np.percentile(v, 100 * a / 2)), float(np.percentile(v, 100 * (1 - a / 2)))]
        fo = {"concepts": {}}
        for c in R.CONCEPTS:
            d0 = np.array([r["concepts"][c]["d0"] for r in reps]); d2 = np.array([r["concepts"][c]["d2"] for r in reps])
            rob = lambda a: bool((pct(d0, a)[0] > 0 and pct(d2, a)[1] < 0) or (pct(d0, a)[1] < 0 and pct(d2, a)[0] > 0))
            e = {"d0_ci95": pct(d0, 0.05), "d2_ci95": pct(d2, 0.05), "d0_ci_bonf": pct(d0, ALPHA_B), "d2_ci_bonf": pct(d2, ALPHA_B),
                 "robust_95": rob(0.05), "robust_bonf": rob(ALPHA_B), "n_same_sign": int(np.sum(np.sign(d0) == np.sign(d2)))}
            if c in PRIMARY and origs:
                for cal in ("platt", "iso"):
                    mean_curve = np.mean([o["concepts"][c][f"psiL_{cal}"] for o in origs], 0)
                    curves = np.array([r["concepts"][c][f"psiL_{cal}"] for r in reps])
                    sp = np.array([tip(cv) for cv in curves])
                    # 95. phân vị của s* = sàn nhỏ nhất mà cận dưới một phía 95% (phân vị 5%) của ψ_L(s) dương
                    lcb = np.percentile(curves, 5, axis=0)
                    joint = {str(s): float(np.mean((curves[:, i] > 0) & (d2 < 0))) for i, s in enumerate(S_FINE) if round(s * 10, 6) % 1 == 0}
                    e[f"s_star_{cal}"] = {"orig_mean_curve": tip(mean_curve),
                                          "orig_max_over_seeds": float(np.nanmax([tip(o["concepts"][c][f"psiL_{cal}"]) for o in origs])),
                                          "q95": float(np.nanpercentile(sp, 95)) if np.isfinite(sp).mean() >= 0.95 else None,
                                          "quantiles": {str(qq): (float(np.nanpercentile(sp, qq)) if np.isfinite(sp).mean() >= qq / 100 else None) for qq in (5, 25, 50, 75, 95)},
                                          "defined_frac": float(np.isfinite(sp).mean()), "lcb_floor": tip(lcb),
                                          "joint_psiL_pos_and_d2_neg": joint,
                                          "psiL_orig": {str(s): float(mean_curve[S_FINE.tolist().index(s)]) for s in (0.5, 0.7, 0.9)}}
                e["s_star_support"] = {}
                for qn in thr:
                    mean_curve = np.mean([o["concepts"][c]["psiL_support"][qn] for o in origs], 0)
                    sp = np.array([tip(r["concepts"][c]["psiL_support"][qn]) for r in reps])
                    e["s_star_support"][qn] = {"orig": tip(mean_curve),
                                               "q95": float(np.nanpercentile(sp, 95)) if np.isfinite(sp).mean() >= 0.95 else None,
                                               "defined_frac": float(np.isfinite(sp).mean())}
            fo["concepts"][c] = e
            R.log(f"{fam} {c:22s} rob95={e['robust_95']} robB={e['robust_bonf']} " +
                  (f"s*={e['s_star_platt']['orig_mean_curve']} q95={e['s_star_platt']['q95']} iso={e['s_star_iso']['orig_mean_curve']}/{e['s_star_iso']['q95']}"
                   if "s_star_platt" in e else ""))
        out["families"][fam] = fo
    R.save_json(out, "vr24_full_bootstrap.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family"); ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=500)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    summarize() if a.summarize else run(a.family, a.start, a.end)


if __name__ == "__main__":
    main()
