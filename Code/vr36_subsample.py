#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR36 — Độ nhạy của suy luận chính với cách lấy lại: lấy mẫu con một nửa bệnh nhân KHÔNG hoàn lại (m = n/2) ở train và test.

Mỗi lần lặp (300 mỗi họ): chọn ngẫu nhiên một nửa bệnh nhân train và một nửa bệnh nhân test, không hoàn lại; huấn luyện lại
M0, M2 (một seed, val cố định như phân tích chính); tính Δ_M0, Δ_M2 cho ba concept chính. Khoảng m-out-of-n với tốc độ √n:
[θ̂ − Q_{1−a/2}, θ̂ − Q_{a/2}], Q là phân vị của √(m/n)(θ*_m − θ̂), θ̂ là ước lượng điểm (trung bình ba seed gốc của vr19).
Báo khoảng 95% và Bonferroni (1 − 0.05/6), nhãn robust, và tỉ lệ lần lặp mà Δ_M0 và Δ_M2 cùng dấu.

Chạy:  python3 vr36_subsample.py --family tabular --start 0 --end 150 ; ... --summarize
Out -> Result/vr36/{family}_{start}_{end}.json ; Result/vr36_subsample.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

import vr_common as R
import gpu_setup  # noqa: F401
from vr19_primary_bootstrap import features, arrays, PRIMARY

OUTD = os.path.join(R.RES, "vr36")
ALPHA_B = 0.05 / 6


def run(family, start, end):
    import vl_models as M
    df, fold = R.load_isic()
    X, hid = features(df, family); A = arrays(df, X)
    tr_all, va, te_all = (np.flatnonzero(fold == k) for k in ("train", "val", "test"))
    Tfull = {c: R.SV.tertile(df, R.CONCEPTS[c]).fillna(-1).to_numpy() for c in PRIMARY}
    pid = df[R.SV.GROUP].to_numpy()
    trp, tep = np.unique(pid[tr_all]), np.unique(pid[te_all])
    tr_by = {p: tr_all[pid[tr_all] == p] for p in trp}; te_by = {p: te_all[pid[te_all] == p] for p in tep}
    os.makedirs(OUTD, exist_ok=True)
    path = os.path.join(OUTD, f"{family}_{start}_{end}.json")
    res = json.load(open(path)) if os.path.exists(path) else {"family": family, "reps": {}}
    for b in range(start, end):
        if str(b) in res["reps"]:
            continue
        rng = np.random.default_rng(160_000 + b)
        tr = np.concatenate([tr_by[p] for p in rng.choice(trp, len(trp) // 2, replace=False)])
        te = np.concatenate([te_by[p] for p in rng.choice(tep, len(tep) // 2, replace=False)])
        m2 = M.train("erm_verified", A, tr, va, hid=hid, seed=b); m0 = M.train("erm_y", A, tr, va, hid=hid, seed=b)
        l0 = R.logit(M.predict(m0, X[te])["pD"]); l2 = R.logit(M.predict(m2, X[te])["pD"])
        res["reps"][str(b)] = {c: [float(l0[T[te] == 1].mean() - l0[T[te] == 0].mean()), float(l2[T[te] == 1].mean() - l2[T[te] == 0].mean())]
                               for c, T in Tfull.items()}
        json.dump(res, open(path + ".tmp", "w")); os.replace(path + ".tmp", path)
        if b % 25 == 0:
            R.log(f"{family} rep {b}: color {res['reps'][str(b)]['color_variegation']}")


def summarize():
    v19 = json.load(open(os.path.join(R.RES, "vr19_primary_bootstrap.json")))["families"]
    out = {"families": {}}
    for fam in ("tabular", "image"):
        reps = []
        for f in glob.glob(os.path.join(OUTD, f"{fam}_*.json")):
            reps += list(json.load(open(f))["reps"].values())
        fo = {"B": len(reps)}
        for c in PRIMARY:
            a = np.array([r[c] for r in reps]); th = np.array([v19[fam]["concepts"][c]["d0_orig"], v19[fam]["concepts"][c]["d2_orig"]])
            z = np.sqrt(0.5) * (a - th)
            ci = lambda al, j: [float(th[j] - np.percentile(z[:, j], 100 * (1 - al / 2))), float(th[j] - np.percentile(z[:, j], 100 * al / 2))]
            rob = lambda al: bool((ci(al, 0)[0] > 0 and ci(al, 1)[1] < 0) or (ci(al, 0)[1] < 0 and ci(al, 1)[0] > 0))
            fo[c] = {"d0_ci95": ci(0.05, 0), "d2_ci95": ci(0.05, 1), "d0_ci_bonf": ci(ALPHA_B, 0), "d2_ci_bonf": ci(ALPHA_B, 1),
                     "robust_95": rob(0.05), "robust_bonf": rob(ALPHA_B), "same_sign_frac": float(np.mean(np.sign(a[:, 0]) == np.sign(a[:, 1])))}
            R.log(f"{fam} {c}: d0 {np.round(fo[c]['d0_ci_bonf'], 2)} d2 {np.round(fo[c]['d2_ci_bonf'], 2)} rob95 {fo[c]['robust_95']} robB {fo[c]['robust_bonf']} same-sign {fo[c]['same_sign_frac']:.3f}")
        out["families"][fam] = fo
    R.save_json(out, "vr36_subsample.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family"); ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=300)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    summarize() if a.summarize else run(a.family, a.start, a.end)


if __name__ == "__main__":
    main()
