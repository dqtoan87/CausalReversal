#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR29 — Rà soát suy luận chính từ các lần lặp đã lưu của vr19 (không huấn luyện lại).

(a) Sai số Monte Carlo của các đầu mút Bonferroni (phân vị 0.42% và 99.58%) với B = 1,000: lấy lại chính các lần lặp
    (2,000 lần) để ước lượng độ lệch chuẩn của đầu mút và tần suất nhãn "robust" giữ nguyên.
(b) Biến cố đồng thời của đảo dấu so với bệnh: trong CÙNG lần lặp, ψ_L(s) > 0 và Δ_M2 < 0, tại s = 0.3, 0.5, 0.7, 0.9;
    và P(s* ≤ s, Δ_M2 < 0) trên lưới s.
(c) Phân phối của s* qua các lần lặp (phân vị 5, 25, 50, 75, 95), và tính tương đương: phân vị 95 của s* bằng sàn nhỏ
    nhất mà phân vị 5 của ψ_L(s) dương (ψ_L tăng theo s).

Out -> Result/vr29_bootstrap_audit.json
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

import vr_common as R

ALPHA_B = 0.05 / 6
PRIMARY = ["color_variegation", "size", "lesion_skin_contrast"]


def main():
    out = {"families": {}}
    rng = np.random.default_rng(5)
    for fam in ("tabular", "image"):
        reps = []
        for f in glob.glob(os.path.join(R.RES, "vr19", f"{fam}_*.json")):
            reps += [v for k, v in json.load(open(f))["reps"].items() if int(k) >= 0]
        B = len(reps); fo = {"B": B}
        for c in PRIMARY:
            d0 = np.array([r["concepts"][c]["d0"] for r in reps]); d2 = np.array([r["concepts"][c]["d2"] for r in reps])
            lo, hi = 100 * ALPHA_B / 2, 100 * (1 - ALPHA_B / 2)
            ends, labels = [], []
            for _ in range(2000):
                ix = rng.integers(0, B, B)
                e = [np.percentile(d0[ix], lo), np.percentile(d0[ix], hi), np.percentile(d2[ix], lo), np.percentile(d2[ix], hi)]
                ends.append(e); labels.append((e[0] > 0 and e[3] < 0) or (e[1] < 0 and e[2] > 0))
            ends = np.array(ends)
            rob0 = (np.percentile(d0, lo) > 0 and np.percentile(d2, hi) < 0) or (np.percentile(d0, hi) < 0 and np.percentile(d2, lo) > 0)
            sp = np.array([np.nan if r["concepts"][c]["s_pos"] is None else r["concepts"][c]["s_pos"] for r in reps])
            psiL = {s: np.array([r["concepts"][c]["psiL"][s] for r in reps]) for s in ("0.3", "0.5", "0.7", "0.9")}
            grid = np.round(np.arange(0.3, 1.0001, 0.05), 2)
            fo[c] = {"mc_se_endpoints": {"d0_lower": float(ends[:, 0].std()), "d0_upper": float(ends[:, 1].std()),
                                         "d2_lower": float(ends[:, 2].std()), "d2_upper": float(ends[:, 3].std())},
                     "robust_label": bool(rob0), "label_stability": float(np.mean(np.array(labels) == rob0)),
                     "joint_psiL_pos_d2_neg": {s: float(np.mean((v > 0) & (d2 < 0))) for s, v in psiL.items()},
                     "joint_sstar_le_s_d2_neg": {str(s): float(np.mean((np.nan_to_num(sp, nan=9) <= s + 1e-9) & (d2 < 0))) for s in grid},
                     "sstar_quantiles": {str(q): (lambda x: None if x > 1 else x)(float(np.quantile(np.where(np.isfinite(sp), sp, 9.0), q / 100, method="inverted_cdf"))) for q in (5, 25, 50, 75, 95)},
                     "sstar_defined_frac": float(np.isfinite(sp).mean()),
                     "psiL_q05": {s: float(np.percentile(v, 5)) for s, v in psiL.items()}}
            R.log(f"{fam} {c}: robust={rob0} stability {fo[c]['label_stability']:.3f}; MC SE d0 lower {fo[c]['mc_se_endpoints']['d0_lower']:.3f}; "
                  f"joint {fo[c]['joint_psiL_pos_d2_neg']}; s* quantiles {fo[c]['sstar_quantiles']}")
        out["families"][fam] = fo
    R.save_json(out, "vr29_bootstrap_audit.json")


if __name__ == "__main__":
    main()
