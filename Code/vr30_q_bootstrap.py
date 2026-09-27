#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR30 — Bootstrap toàn quy trình (lấy lại bệnh nhân train, val, test độc lập) với BA cách ước lượng q, cùng một tiêu chí.

Mỗi lần lặp (1,000 mỗi họ chính):
  - M0, M2 (MLP như phân tích chính, một seed); Δ_M0, Δ_M2 thô cho năm concept;
  - q̂ theo ba cách, tất cả khớp/hiệu chỉnh trên train/val đã lấy lại:
      mlp_platt  M0 hiệu chỉnh Platt (phân tích plug-in chính);
      mlp_iso    M0 hiệu chỉnh isotonic;
      gbm_platt  gradient boosting của Y trên input (ảnh: 64 thành phần chính của đặc trưng ảnh, PCA khớp một lần trên
                 train gốc), hiệu chỉnh Platt;
  - đường ψ_L(s), s = 0.01..1, cho ba concept chính trên toàn test và trên vùng hỗ trợ (σ̂ ≥ phân vị 1/5/10%).
Lần chạy gốc b = −1..−3 (seed 0–2) còn lưu chẩn đoán của từng q̂ trên test: log-loss, Brier, hiệu chỉnh theo khoảng q̂
(tỉ lệ Y quan sát so với q̂ trung bình), và số Y = 1 theo tertile của concept.

Chạy:  python3 vr30_q_bootstrap.py --family tabular --start -3 --end 330 ; ... --summarize
Out -> Result/vr30/{family}_{start}_{end}.json ; Result/vr30_q_bootstrap.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

import vr_common as R
import gpu_setup  # noqa: F401
from vr19_primary_bootstrap import features, arrays, platt_fit, S_FINE, SUPPORT_Q, PRIMARY
from vr24_full_bootstrap import psiL_curve, iso_fit, tip
from vr19_primary_bootstrap import q95_floor

OUTD = os.path.join(R.RES, "vr30")
EST = ("mlp_platt", "mlp_iso", "gbm_platt")
BINS = [0, 5e-4, 2e-3, 1e-2, 1.0]
ALPHA_B = 0.05 / 6


def gbm_input(df, X, family, tr_all):
    if family == "tabular":
        return X
    from sklearn.decomposition import PCA
    return PCA(64, random_state=0).fit(X[tr_all[::4]]).transform(X).astype(np.float32)


def diagnostics(q, y, Tt):
    q = np.clip(q, 1e-7, 1 - 1e-7)
    out = {"logloss": float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))), "brier": float(np.mean((q - y) ** 2)), "bins": []}
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (q >= lo) & (q < hi)
        out["bins"].append({"range": [lo, hi], "n": int(m.sum()), "mean_q": float(q[m].mean()) if m.any() else None,
                            "obs_rate": float(y[m].mean()) if m.any() else None, "n_pos": int(y[m].sum())})
    out["pos_by_tertile"] = {c: [int(y[T == 0].sum()), int(y[T == 1].sum())] for c, T in Tt.items()}
    return out


def one_rep(M, df, X, Xg, hid, A, tr, va, te, seed, sig, Tfull, sup_masks, diag):
    from sklearn.ensemble import HistGradientBoostingClassifier
    Y, S = df.Y.to_numpy(), df.S.to_numpy()
    va2 = va[S[va] == 1]
    m2 = M.train("erm_verified", A, tr, va, hid=hid, seed=seed)
    m0 = M.train("erm_y", A, tr, va, hid=hid, seed=seed)
    pv0 = M.predict(m0, X[va])["pD"]
    a0, b0 = platt_fit(R.logit(pv0), Y[va]); iso = iso_fit(pv0, Y[va])
    gb = HistGradientBoostingClassifier(max_depth=4, max_iter=400, learning_rate=0.05, random_state=seed % 1000).fit(Xg[tr], Y[tr])
    lgv = R.logit(np.clip(gb.predict_proba(Xg[va])[:, 1], 1e-7, 1 - 1e-7)); ag, bg = platt_fit(lgv, Y[va])
    ute, inv = np.unique(te, return_inverse=True)
    p0 = M.predict(m0, X[ute])["pD"][inv]
    l0 = R.logit(p0); l2 = R.logit(M.predict(m2, X[ute])["pD"])[inv]
    lgt = R.logit(np.clip(gb.predict_proba(Xg[ute])[:, 1], 1e-7, 1 - 1e-7))[inv]
    qs = {"mlp_platt": 1 / (1 + np.exp(-(a0 * l0 + b0))), "mlp_iso": np.clip(iso.predict(p0), 1e-6, 1 - 1e-6),
          "gbm_platt": 1 / (1 + np.exp(-(ag * lgt + bg)))}
    sg = sig[te]
    out = {"concepts": {}}
    for c, Tf in Tfull.items():
        T = Tf[te]
        r = {"d0": float(l0[T == 1].mean() - l0[T == 0].mean()), "d2": float(l2[T == 1].mean() - l2[T == 0].mean())}
        if c in PRIMARY:
            for k, q in qs.items():
                r[f"psiL_{k}"] = np.round(psiL_curve(q, sg, T), 4).tolist()
                r[f"sup_{k}"] = {qn: np.round(psiL_curve(q[msk[te]], sg[msk[te]], T[msk[te]]), 4).tolist() for qn, msk in sup_masks.items()}
        out["concepts"][c] = r
    if diag:
        Tt = {c: Tfull[c][te] for c in PRIMARY}
        out["diagnostics"] = {k: diagnostics(q, Y[te].astype(float), Tt) for k, q in qs.items()}
        out["diagnostics"]["mlp_raw"] = diagnostics(p0, Y[te].astype(float), Tt)
    return out


def run(family, start, end):
    import vl_models as M
    df, fold = R.load_isic()
    X, hid = features(df, family)
    A = arrays(df, X)
    S = df.S.to_numpy()
    tr_all, va_all, te_all = (np.flatnonzero(fold == k) for k in ("train", "val", "test"))
    Xg = gbm_input(df, X, family, tr_all)
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
            rng = np.random.default_rng(110_000 + b)
            tr, va, te = (np.concatenate([by[k][1][p] for p in rng.choice(by[k][0], len(by[k][0]))]) for k in ("tr", "va", "te"))
        res["reps"][str(b)] = one_rep(M, df, X, Xg, hid, A, tr, va, te, (-b - 1) if b < 0 else b, sig, Tfull, sup_masks, b < 0)
        json.dump(res, open(path + ".tmp", "w")); os.replace(path + ".tmp", path)
        if b % 25 == 0:
            c = res["reps"][str(b)]["concepts"]["color_variegation"]
            R.log(f"{family} rep {b}: d0 {c['d0']:+.2f} d2 {c['d2']:+.2f} s* " + " ".join(f"{k} {tip(c['psiL_' + k])}" for k in EST))


def summarize():
    out = {"B": {}, "grid": S_FINE.tolist(), "families": {}}
    rng = np.random.default_rng(3)
    for fam in ("tabular", "image"):
        reps, origs = [], []
        for f in glob.glob(os.path.join(OUTD, f"{fam}_*.json")):
            for k, v in json.load(open(f))["reps"].items():
                (origs if int(k) < 0 else reps).append(v)
        B = len(reps); out["B"][fam] = B
        pct = lambda v, a: [float(np.percentile(v, 100 * a / 2)), float(np.percentile(v, 100 * (1 - a / 2)))]
        fo = {"concepts": {}, "diagnostics": {}}
        if origs:
            for k in origs[0]["diagnostics"]:
                ds = [o["diagnostics"][k] for o in origs]
                fo["diagnostics"][k] = {"logloss": float(np.mean([d["logloss"] for d in ds])), "brier": float(np.mean([d["brier"] for d in ds])),
                                        "bins": [{"range": ds[0]["bins"][i]["range"], "n": float(np.mean([d["bins"][i]["n"] for d in ds])),
                                                  "mean_q": (float(np.mean([d["bins"][i]["mean_q"] for d in ds if d["bins"][i]["mean_q"] is not None]))
                                                             if any(d["bins"][i]["mean_q"] is not None for d in ds) else None),
                                                  "n_pos": float(np.mean([d["bins"][i]["n_pos"] for d in ds]))} for i in range(len(BINS) - 1)],
                                        "pos_by_tertile": ds[0]["pos_by_tertile"]}
        for c in R.CONCEPTS:
            d0 = np.array([r["concepts"][c]["d0"] for r in reps]); d2 = np.array([r["concepts"][c]["d2"] for r in reps])
            rob = lambda a: bool((pct(d0, a)[0] > 0 and pct(d2, a)[1] < 0) or (pct(d0, a)[1] < 0 and pct(d2, a)[0] > 0))
            e = {"robust_95": rob(0.05), "robust_bonf": rob(ALPHA_B), "n_same_sign": int(np.sum(np.sign(d0) == np.sign(d2)))}
            if c in PRIMARY:
                for k in EST:
                    curves = np.array([r["concepts"][c][f"psiL_{k}"] for r in reps])
                    mean_curve = np.mean([o["concepts"][c][f"psiL_{k}"] for o in origs], 0)
                    sp = np.array([tip(cv) for cv in curves])
                    lcb = np.where(np.mean(curves > 0, axis=0) >= 0.95, 1.0, -1.0)     # sàn nhỏ nhất mà ψ_L > 0 ở ≥ 95% lần lặp
                    # sai số Monte Carlo của sàn cận dưới và của biến cố đồng thời tại 0.8 (lấy lại chính các lần lặp)
                    i8 = S_FINE.tolist().index(0.8)
                    mc_l, mc_j = [], []
                    for _ in range(500):
                        ix = rng.integers(0, B, B)
                        mc_l.append(tip(np.where(np.mean(curves[ix] > 0, axis=0) >= 0.95, 1.0, -1.0))); mc_j.append(np.mean((curves[ix, i8] > 0) & (d2[ix] < 0)))
                    joint = {str(s): int(np.sum((curves[:, i] > 0) & (d2 < 0))) for i, s in enumerate(S_FINE) if round(s * 10, 6) % 1 == 0}
                    sup = {}
                    for qn in origs[0]["concepts"][c][f"sup_{k}"]:
                        mc = np.mean([o["concepts"][c][f"sup_{k}"][qn] for o in origs], 0)
                        sc = np.array([r["concepts"][c][f"sup_{k}"][qn] for r in reps])
                        sup[qn] = {"point": tip(mc), "lcb_floor": tip(np.where(np.mean(sc > 0, axis=0) >= 0.95, 1.0, -1.0))}
                    mc_l = np.array(mc_l, float)
                    e[k] = {"point": tip(mean_curve), "q95": q95_floor(sp),
                            "defined_frac": float(np.isfinite(sp).mean()), "lcb_floor": tip(lcb),
                            "lcb_floor_mc_range": [float(np.nanpercentile(mc_l, 2.5)), float(np.nanpercentile(mc_l, 97.5))] if np.isfinite(mc_l).any() else None,
                            "lcb_floor_mc_undefined": float(np.mean(~np.isfinite(mc_l))),
                            "joint_counts": joint, "joint_08_mc_range": [float(np.percentile(mc_j, 2.5)), float(np.percentile(mc_j, 97.5))],
                            "psiL_point": {str(s): float(mean_curve[S_FINE.tolist().index(s)]) for s in (0.5, 0.7, 0.9)}, "support": sup}
            fo["concepts"][c] = e
            R.log(f"{fam} {c:22s} rob95={e['robust_95']} robB={e['robust_bonf']} " +
                  " ".join(f"{k}: pt {e[k]['point']} lcb {e[k]['lcb_floor']}" for k in EST if k in e))
        out["families"][fam] = fo
    R.save_json(out, "vr30_q_bootstrap.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family"); ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=1000)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    summarize() if a.summarize else run(a.family, a.start, a.end)


if __name__ == "__main__":
    main()
