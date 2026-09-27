#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR43 — Suy luận chính của Table 1 bằng bootstrap của đúng thống kê ước lượng điểm: trung bình ba seed.

Ước lượng điểm (vr19, lần chạy gốc) là trung bình Δ của ba seed. vr39 đã huấn luyện thêm hai seed trên cùng mẫu bệnh nhân
cho 1,000 lần lặp đầu của vr19, nên mỗi lần lặp có trung bình ba seed. Script này tính, cho mười ô (ba concept chính, hai
concept phụ × hai họ): khoảng percentile 95% và Bonferroni (1 − 0.05/6) của Δ_M0, Δ_M2, khoảng 95% của Δ_M0 − Δ_M2, số lần
lặp cùng dấu, nhãn tách dấu; sai số Monte Carlo của các đầu mút Bonferroni và tỉ lệ tái lập nhãn qua 2,000 lần lấy lại các
lần lặp; và so sánh nhãn với phân tích một seed 5,000 lần (vr19). Không huấn luyện thêm.

Out -> Result/vr43_three_seed_primary.json
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

import vr_common as R

EXTRA = (100_000, 200_000)
A_B = 0.05 / 6


def load(fam):
    v19, ex = {}, {}
    for f in glob.glob(os.path.join(R.RES, "vr19", f"{fam}_*.json")):
        for k, v in json.load(open(f))["reps"].items():
            if int(k) >= 0:
                v19[int(k)] = {c: (x["d0"], x["d2"]) for c, x in v["concepts"].items()}
    for f in glob.glob(os.path.join(R.RES, "vr39", f"{fam}_*.json")):
        ex.update(json.load(open(f))["reps"])
    bs = sorted(b for b in v19 if all(f"{b}|{o}" in ex for o in EXTRA))
    return v19, ex, bs


def main():
    s19 = json.load(open(os.path.join(R.RES, "vr19_primary_bootstrap.json")))
    out = {"B": {}, "families": {}}
    rng = np.random.default_rng(43)
    for fam in ("tabular", "image"):
        v19, ex, bs = load(fam)
        out["B"][fam] = len(bs)
        fo = {"concepts": {}}
        for c in R.CONCEPTS:
            d0 = np.array([np.mean([v19[b][c][0]] + [ex[f"{b}|{o}"][c]["d0"] for o in EXTRA]) for b in bs])
            d2 = np.array([np.mean([v19[b][c][1]] + [ex[f"{b}|{o}"][c]["d2"] for o in EXTRA]) for b in bs])
            pct = lambda v, a: [float(np.percentile(v, 100 * a / 2)), float(np.percentile(v, 100 * (1 - a / 2)))]
            rob = lambda x0, x2, a: bool((pct(x0, a)[0] > 0 and pct(x2, a)[1] < 0) or (pct(x0, a)[1] < 0 and pct(x2, a)[0] > 0))
            oc = s19["families"][fam]["concepts"][c]
            e = {"d0_orig": oc["d0_orig"], "d2_orig": oc["d2_orig"],
                 "d0_ci95": pct(d0, 0.05), "d2_ci95": pct(d2, 0.05), "d0_ci_bonf": pct(d0, A_B), "d2_ci_bonf": pct(d2, A_B),
                 "diff_ci95": pct(d0 - d2, 0.05), "n_same_sign": int(np.sum(np.sign(d0) == np.sign(d2))),
                 "robust_95": rob(d0, d2, 0.05), "robust_bonf": rob(d0, d2, A_B),
                 "one_seed_robust_95": oc["robust_95"], "one_seed_robust_bonf": oc["robust_bonf"]}
            # Monte Carlo: lấy lại các lần lặp
            ends, labs = [], []
            for _ in range(2000):
                ix = rng.integers(0, len(bs), len(bs))
                ends.append(pct(d0[ix], A_B) + pct(d2[ix], A_B)); labs.append(rob(d0[ix], d2[ix], A_B) == e["robust_bonf"])
            e["mc_se_endpoints"] = np.std(np.array(ends), axis=0, ddof=1).tolist()
            e["label_stability"] = float(np.mean(labs))
            fo["concepts"][c] = e
            R.log(f"{fam} {c}: bonf {e['robust_bonf']} (one-seed {oc['robust_bonf']}) 95 {e['robust_95']} (one-seed {oc['robust_95']}) "
                  f"MC SE max {max(e['mc_se_endpoints']):.3f} label stability {e['label_stability']:.3f}")
        out["families"][fam] = fo
    R.save_json(out, "vr43_three_seed_primary.json")


if __name__ == "__main__":
    main()
