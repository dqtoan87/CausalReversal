#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR39 — Tách biến thiên do seed khỏi biến thiên do lấy lại bệnh nhân trong bootstrap chính (vr19).

Trong vr19, lần lặp b lấy lại bệnh nhân train và test bằng rng(50_000 + b) và huấn luyện M0, M2 với seed = b, nên
phân phối bootstrap trộn hai nguồn: lấy lại bệnh nhân và tính ngẫu nhiên của huấn luyện. Ước lượng điểm lại là trung bình
ba seed. Script này lấy các lần lặp b = 0..NB−1, giữ nguyên mẫu bệnh nhân của vr19 và huấn luyện thêm hai seed
(b + 100_000, b + 200_000). Với mỗi concept báo cáo:
  - phương sai trong lần lặp (giữa ba seed, cùng mẫu bệnh nhân) và phương sai giữa các lần lặp của một seed;
  - tỉ lệ phương sai do seed;
  - khoảng 95% percentile của Δ một seed và của Δ trung bình ba seed trên cùng NB lần lặp.

Chạy:  python3 vr39_seed_variance.py --family tabular --start 0 --end 100   ...  rồi  --summarize
Out -> Result/vr39/{family}_{start}_{end}.json ; Result/vr39_seed_variance.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

import vr_common as R
import gpu_setup  # noqa: F401
from vr19_primary_bootstrap import features, arrays, one_rep

OUTD = os.path.join(R.RES, "vr39")
V19 = os.path.join(R.RES, "vr19")
EXTRA = (100_000, 200_000)
PRIMARY = ["color_variegation", "size", "lesion_skin_contrast"]


def run(family, start, end):
    import vl_models as M
    df, fold = R.load_isic()
    X, hid = features(df, family)
    A = arrays(df, X)
    S = df.S.to_numpy()
    tr_all, va, te_all = (np.flatnonzero(fold == k) for k in ("train", "val", "test"))
    va2 = va[S[va] == 1]
    nz = np.load(os.path.join(R.MISS_RES, "vl_nuisance.npz"))
    sig = nz["sigma"]
    Tfull = {c: R.SV.tertile(df, col).fillna(-1).to_numpy() for c, col in R.CONCEPTS.items()}
    pid = df[R.SV.GROUP].to_numpy()
    trp, tep = np.unique(pid[tr_all]), np.unique(pid[te_all])
    tr_by = {p: tr_all[pid[tr_all] == p] for p in trp}; te_by = {p: te_all[pid[te_all] == p] for p in tep}
    os.makedirs(OUTD, exist_ok=True)
    path = os.path.join(OUTD, f"{family}_{start}_{end}.json")
    res = json.load(open(path)) if os.path.exists(path) else {"family": family, "reps": {}}
    done = set()                                              # lần lặp đã có ở file khác của cùng họ (khi chia lại luồng)
    for f in glob.glob(os.path.join(OUTD, f"{family}_*.json")):
        if f != path:
            done |= set(json.load(open(f))["reps"])
    for b in range(start, end):
        rng = np.random.default_rng(50_000 + b)                 # cùng mẫu bệnh nhân với vr19
        tr = np.concatenate([tr_by[p] for p in rng.choice(trp, len(trp))])
        te = np.concatenate([te_by[p] for p in rng.choice(tep, len(tep))])
        for off in EXTRA:
            key = f"{b}|{off}"
            if key in res["reps"] or key in done:
                continue
            o = one_rep(M, df, X, hid, A, tr, va, va2, te, b + off, sig, Tfull, {})
            res["reps"][key] = {c: {"d0": v["d0"], "d2": v["d2"]} for c, v in o["concepts"].items()}
            json.dump(res, open(path + ".tmp", "w")); os.replace(path + ".tmp", path)
        if b % 10 == 0:
            R.log(f"vr39 {family} rep {b} done")


def load_v19(family):
    reps = {}
    for f in glob.glob(os.path.join(V19, f"{family}_*.json")):
        for k, v in json.load(open(f))["reps"].items():
            if int(k) >= 0:
                reps[int(k)] = {c: {"d0": x["d0"], "d2": x["d2"]} for c, x in v["concepts"].items()}
    return reps


def summarize():
    out = {"families": {}}
    for fam in ("tabular", "image"):
        extra = {}
        for f in glob.glob(os.path.join(OUTD, f"{fam}_*.json")):
            extra.update(json.load(open(f))["reps"])
        v19 = load_v19(fam)
        bs = sorted(b for b in v19 if all(f"{b}|{o}" in extra for o in EXTRA))
        fo = {"n_reps": len(bs), "concepts": {}}
        for c in R.CONCEPTS:
            e = {}
            for k in ("d0", "d2"):
                M3 = np.array([[v19[b][c][k]] + [extra[f"{b}|{o}"][c][k] for o in EXTRA] for b in bs])   # NB × 3
                within = float(np.mean(np.var(M3, axis=1, ddof=1)))
                total1 = float(np.var(M3[:, 0], ddof=1))
                avg = M3.mean(1)
                e[k] = {"sd_within_seed": within ** 0.5, "sd_single_seed": total1 ** 0.5, "sd_three_seed_avg": float(np.std(avg, ddof=1)),
                        "seed_share": within / total1,
                        "ci95_single": [float(np.percentile(M3[:, 0], 2.5)), float(np.percentile(M3[:, 0], 97.5))],
                        "ci95_avg3": [float(np.percentile(avg, 2.5)), float(np.percentile(avg, 97.5))]}
            # suy luận theo trung bình ba seed trên cùng các lần lặp (đúng thống kê của ước lượng điểm)
            A3 = {k: np.array([[v19[b][c][k]] + [extra[f"{b}|{o}"][c][k] for o in EXTRA] for b in bs]).mean(1) for k in ("d0", "d2")}
            S1 = {k: np.array([v19[b][c][k] for b in bs]) for k in ("d0", "d2")}
            def lab(V, a):
                lo0, hi0 = np.percentile(V["d0"], [100 * a / 2, 100 * (1 - a / 2)]); lo2, hi2 = np.percentile(V["d2"], [100 * a / 2, 100 * (1 - a / 2)])
                return bool((lo0 > 0 and hi2 < 0) or (hi0 < 0 and lo2 > 0)), [float(lo0), float(hi0)], [float(lo2), float(hi2)]
            for nm, V in (("avg3", A3), ("single", S1)):
                r95, _, _ = lab(V, 0.05); rb, c0, c2 = lab(V, 0.05 / 6)
                e[nm] = {"robust_95": r95, "robust_bonf": rb, "d0_ci_bonf": c0, "d2_ci_bonf": c2}
            fo["concepts"][c] = e
        out["families"][fam] = fo
    p = os.path.join(R.RES, "vr39_seed_variance.json")
    json.dump(out, open(p, "w"), indent=1)
    R.log(f"wrote {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family"); ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=200)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    summarize() if a.summarize else run(a.family, a.start, a.end)
