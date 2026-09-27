#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR46 — Liều chọn mẫu chỉ can thiệp lên tập HUẤN LUYỆN: validation cố định qua mọi mức η, số ngẫu nhiên chung.

Khác vr32 ở hai điểm, để cô lập đúng can thiệp mà Proposition 1 mô tả (chọn vào phân phối huấn luyện R):
  1. Validation cố định trong mỗi lần lặp: tập ác tính validation và nhãn âm validation (rút đều, không theo η) được rút một
     lần và dùng chung cho cả năm mức η, nên dừng sớm và hiệu chỉnh Platt dùng cùng quần thể, cùng quy trình.
  2. Số ngẫu nhiên chung (common random numbers): mỗi nhãn âm train nhận một U_i ~ Uniform(0, 1) một lần mỗi lần lặp và vào
     huấn luyện ở mức η khi U_i < π_i(η) (lấy mẫu Poisson, cùng xác suất như vr26/vr32). Tập ác tính train và seed cũng
     dùng chung. Như vậy Δ(η₁) và Δ(η₀) trong cùng lần lặp là hai kết cục tiềm năng của cùng ω.

Kết cục: tương phản thô (logit của learner, đại lượng Proposition 1 nói tới) và tương phản sau Platt trên validation cố định
(hệ số dương nên cùng dấu với tương phản thô). Dự đoán: tabular chính xác cho dịch chuyển Bayes-tối-ưu của phân phối huấn
luyện, −(tương phản tertile của log π_η(z_k)); ảnh bậc một với m_k(x̃) = E[z_k | x̃, Y = 0] ước lượng bằng ridge.

Chạy:  python3 vr46_dose_fixed_val.py --family tabular --cells color_variegation:320 size:320 ...   rồi  --summarize
Out -> Result/vr46/{family}_{concept}_{n}.json ; Result/vr46_dose_fixed_val.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

import vr_common as R

ETAS = [-1.0, -0.5, 0.0, 0.5, 1.0]
N_REP = 20
N_MAL = 178
PRIMARY = ["color_variegation", "size", "lesion_skin_contrast"]
OUTD = os.path.join(R.RES, "vr46")


def run(kind, cells):
    import gpu_setup  # noqa: F401
    import vl_models as M
    from sklearn.linear_model import Ridge
    from vr21_dose_calibrated import features, _fit
    from vr26_dose_poisson import poisson_probs
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
    X, hid = features(df, kind)
    os.makedirs(OUTD, exist_ok=True)
    for cell in cells:
        k, n_ben = cell.split(":"); n_ben = int(n_ben)
        path = os.path.join(OUTD, f"{kind}_{k}_{n_ben}.json")
        res = json.load(open(path)) if os.path.exists(path) else {"family": kind, "concept": k, "n_ben": n_ben, "etas": ETAS, "reps": {}}
        for r in range(N_REP):
            if str(r) in res["reps"]:
                continue
            rng = np.random.default_rng(46_000 + 1000 * r + 7 * PRIMARY.index(k) + n_ben)
            mtr = rng.choice(pools["train"]["mal"], N_MAL, replace=False)
            mva = rng.choice(pools["val"]["mal"], nva_m, replace=False)
            nva_b = int(round(111 * n_ben / 320))
            bva = rng.choice(pools["val"]["neg"], nva_b, replace=False)          # validation cố định, không theo η
            sel_va = np.concatenate([mva, bva])
            U = rng.random(len(neg))                                            # số ngẫu nhiên chung qua các mức η
            rep = {"raw": [], "cal": [], "exact": [], "n_neg": []}
            if kind == "image":
                ix = np.concatenate([neg_by[p] for p in rng.choice(neg_p, len(neg_p))])
                ix = rng.choice(ix, min(60000, len(ix)), replace=False)
                m_r = Ridge(alpha=10.0).fit(X[ix], Zs[k][ix]).predict(X[te])
                rep["ridge"] = {j: -D(m_r, j) for j in R.CONCEPTS}
            for eta in ETAS:
                ptr, kap = poisson_probs(np.exp(eta * Zs[k][neg]), n_ben)
                btr = neg[U < ptr]
                raw, cal = _fit(M, df, X, hid, np.concatenate([mtr, btr]), sel_va, tr_all, va_all, te, r)
                rep["raw"].append({j: D(raw, j) for j in R.CONCEPTS})
                rep["cal"].append({j: D(cal, j) for j in R.CONCEPTS})
                rep["n_neg"].append(int(len(btr)))
                lpi = np.log(np.minimum(1.0, kap * np.exp(eta * Zs[k][te])))
                rep["exact"].append({j: -D(lpi, j) for j in R.CONCEPTS})
            res["reps"][str(r)] = rep
            json.dump(res, open(path + ".tmp", "w")); os.replace(path + ".tmp", path)
            R.log(f"vr46 {kind} {k}|{n_ben} rep {r}: raw {rep['raw'][0][k]:+.2f} -> {rep['raw'][-1][k]:+.2f}")


def summarize():
    rng = np.random.default_rng(46)
    e = np.array(ETAS); i0, i1 = ETAS.index(-1.0), ETAS.index(1.0)
    out = {"etas": ETAS, "n_rep": N_REP, "families": {}}
    for f in sorted(glob.glob(os.path.join(OUTD, "*.json"))):
        d = json.load(open(f)); kind, k, n = d["family"], d["concept"], d["n_ben"]
        reps = [d["reps"][str(r)] for r in range(N_REP) if str(r) in d["reps"]]
        if len(reps) < N_REP:
            R.log(f"incomplete {f}: {len(reps)}"); continue
        c = {"n_rep": len(reps)}
        slope = lambda rp, key, j: float(np.polyfit(e, [o[j] for o in rp[key]], 1)[0])
        for key in ("raw", "cal"):
            so = np.array([slope(rp, key, k) for rp in reps])
            diff = np.array([rp[key][i1][k] - rp[key][i0][k] for rp in reps])
            bt = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10_000)]
            mean_curve = np.mean([[o[k] for o in rp[key]] for rp in reps], 0)
            c[key] = {"obs_slope": float(so.mean()), "obs_range95": np.percentile(so, [2.5, 97.5]).tolist(),
                      "tau_hat": float(diff.mean()), "tau_ci95": np.percentile(bt, [2.5, 97.5]).tolist(),
                      "n_sign_change": int(sum((rp[key][i0][k] > 0) and (rp[key][i1][k] < 0) for rp in reps)),
                      "mean_by_eta": mean_curve.tolist(), "crosses_zero": bool(mean_curve[0] > 0 > mean_curve[-1]),
                      "spill_obs": {j: float(np.mean([slope(rp, key, j) for rp in reps])) for j in R.CONCEPTS}}
        pkey = "exact" if kind == "tabular" else "ridge"
        if kind == "tabular":
            pr = np.array([slope(rp, "exact", k) for rp in reps])
            spill_pred = {j: float(np.mean([slope(rp, "exact", j) for rp in reps])) for j in R.CONCEPTS}
        else:
            pr = np.array([rp["ridge"][k] for rp in reps])
            spill_pred = {j: float(np.mean([rp["ridge"][j] for rp in reps])) for j in R.CONCEPTS}
        so = np.array([slope(rp, "raw", k) for rp in reps])
        bt = []
        for _ in range(2000):
            ii = rng.integers(0, len(reps), len(reps)); bt.append(so[ii].mean() / pr[ii].mean())
        c["pred"] = {"spec": pkey, "slope": float(pr.mean()), "tau_pred": float(pr.mean() * (ETAS[i1] - ETAS[i0])),
                     "ratio_raw": float(so.mean() / pr.mean()), "ratio_raw_ci95": np.percentile(bt, [2.5, 97.5]).tolist(),
                     "spill_pred": spill_pred,
                     "spill_sign_agreement_raw": int(sum(np.sign(c["raw"]["spill_obs"][j]) == np.sign(spill_pred[j]) for j in R.CONCEPTS))}
        c["n_neg_mean"] = np.mean([rp["n_neg"] for rp in reps], 0).tolist()
        out["families"].setdefault(kind, {})[f"{k}|{n}"] = c
        R.log(f"{kind} {k}|{n}: tau raw {c['raw']['tau_hat']:+.2f} {np.round(c['raw']['tau_ci95'], 2)} cal {c['cal']['tau_hat']:+.2f} "
              f"slope raw {c['raw']['obs_slope']:+.2f} pred {c['pred']['slope']:+.2f} ratio {c['pred']['ratio_raw']:.2f} "
              f"cross {c['cal']['crosses_zero']} sign-change {c['cal']['n_sign_change']}/{len(reps)}")
    R.save_json(out, "vr46_dose_fixed_val.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family"); ap.add_argument("--cells", nargs="*", default=[])
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    summarize() if a.summarize else run(a.family, a.cells)
