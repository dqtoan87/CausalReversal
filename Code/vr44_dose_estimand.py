#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR44 — Estimand can thiệp của thí nghiệm liều chọn mẫu (vr32), không huấn luyện thêm.

Estimand: τ_j(η₁, η₀) = E_ξ[Δ_j(ξ; η₁) − Δ_j(ξ; η₀)], với ξ là tính ngẫu nhiên của quy trình huấn luyện (tập ác tính con,
tập nhãn âm được chọn, seed), điều kiện trên cohort và quần thể test cố định. Trong vr32, mỗi lần lặp r giữ cố định tập ác
tính (train và validation) và seed qua năm mức η; chỉ nhãn âm của train và validation được rút theo từng mức η, nên hiệu Δ(1) − Δ(−1) là ghép cặp.
Báo cáo cho concept được chọn (j = k), 320 nhãn âm, hai họ chính:
  - τ̂(1, −1) = trung bình của hiệu ghép cặp qua 20 lần lặp, khoảng 95% bootstrap theo lần lặp (percentile, 10,000 lần);
  - τ̂ dự đoán từ độ dốc của Proposition 1 (tabular: chính xác; ảnh: ridge), nhân với η₁ − η₀ = 2;
  - số lần lặp có Δ(−1) > 0 > Δ(1) (đổi dấu trong chính lần lặp đó);
  - Δ̂(−1), Δ̂(1) trung bình.
Khoảng này phản ánh tính ngẫu nhiên của huấn luyện, không phản ánh việc lấy lại bệnh nhân.

Out -> Result/vr44_dose_estimand.json
"""
from __future__ import annotations

import json
import os

import numpy as np

import vr_common as R

PRIMARY = ["color_variegation", "size", "lesion_skin_contrast"]
SPEC = {"tabular": "exact", "image": "ridge"}


def main():
    d = json.load(open(os.path.join(R.RES, "vr32_dose_nuisance.json")))
    etas = d["etas"]
    i0, i1 = etas.index(-1.0), etas.index(1.0)
    rng = np.random.default_rng(44)
    out = {"etas": [etas[i0], etas[i1]], "n_rep": d["n_rep"], "families": {}}
    for fam in ("tabular", "image"):
        fo = {}
        for k in PRIMARY:
            reps = d["families"][fam][f"{k}|320"]["per_rep"]
            lo = np.array([r["mean_by_eta"][i0] for r in reps]); hi = np.array([r["mean_by_eta"][i1] for r in reps])
            diff = hi - lo
            boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10_000)]
            pred = np.mean([r["pred"][SPEC[fam]][k] for r in reps]) * (etas[i1] - etas[i0])
            fo[k] = {"tau_hat": float(diff.mean()), "tau_ci95": np.percentile(boot, [2.5, 97.5]).tolist(),
                     "tau_pred": float(pred), "n_sign_change": int(np.sum((lo > 0) & (hi < 0))),
                     "delta_lo_mean": float(lo.mean()), "delta_hi_mean": float(hi.mean()), "pred_spec": SPEC[fam]}
            R.log(f"{fam} {k}: tau {fo[k]['tau_hat']:.2f} {np.round(fo[k]['tau_ci95'], 2)} pred {pred:.2f} "
                  f"sign change {fo[k]['n_sign_change']}/{len(reps)}")
        out["families"][fam] = fo
    R.save_json(out, "vr44_dose_estimand.json")


if __name__ == "__main__":
    main()
