#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR5 — Kiểm tra representation reversal sau fine-tune (proposal_v3 §12, §23), đọc Result/finetune/*.npz.

Mọi phép đo trên CÙNG tập probe (test) cho mọi run. Concept ảnh: color, contrast, asymmetry, border (size bị
loại theo §13; size chỉ xuất hiện ở tương phản đầu ra, vốn dùng giá trị TBP chứ không dùng ảnh).

Cho mỗi run (mode ∈ {ft, lp}, regime ∈ {M0, M2}, seed):
  1. Tương phản đầu ra (decision): logit trung bình tertile trên − dưới của concept TBP, trên phần đồng đều (E0).
  2. Probe concept (§12.1): ridge h → C_k, R² kiểm chứng chéo 5 fold (khả năng giải mã concept);
     alignment A_k = cos(w_fc, w_Ck) giữa hướng quyết định và hướng concept trong không gian h.
  3. Can thiệp layer3 (§12.2): Δ từ vr4 (dịch layer3 ±1 SD theo hướng concept, truyền qua layer4 + head).
  4. PHÉP THỬ READOUT CHUNG — phân định Trường hợp A với B. Trên h của từng encoder, học ridge cho cùng một
     mục tiêu, rồi đo alignment của readout đó với hướng concept:
        target_Y   logit q̂(x), rủi ro nhãn ghi nhận (GBM tabular out-of-fold, vl1_identification.py)
        target_S   logit(q̂/σ̂), rủi ro trên tổn thương đã xác minh (mục tiêu của M2)
     Nếu với CÙNG mục tiêu mà encoder M0 và M2 cho alignment trái dấu, đảo dấu nằm trong representation (A).
     Nếu cùng dấu, representation vẫn biểu diễn được cả hai quan hệ, và đảo dấu nằm ở head/decision (B).
  5. CKA tuyến tính giữa h của M0 và M2 (cùng mode, cùng seed), kèm CKA giữa hai seed của cùng regime làm mốc.
Phán quyết theo §23 được ghi vào "verdict".

Out -> Result/vr5_representation.json
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import vr_common as R

FT = os.path.join(R.RES, "finetune")
N_CKA = 8000


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def ridge_dir(H, y, alpha=100.0):
    ok = np.isfinite(y)
    Hm = H[ok] - H[ok].mean(0)
    yy = (y[ok] - y[ok].mean()) / (y[ok].std() + 1e-9)
    return Ridge(alpha=alpha, fit_intercept=False).fit(Hm, yy).coef_


def cv_r2(H, y, alpha=100.0):
    ok = np.flatnonzero(np.isfinite(y))
    rs = []
    for a, b in KFold(5, shuffle=True, random_state=0).split(ok):
        m = Ridge(alpha=alpha).fit(H[ok[a]], y[ok[a]])
        pr = m.predict(H[ok[b]])
        rs.append(1 - np.mean((y[ok[b]] - pr) ** 2) / np.var(y[ok[b]]))
    return float(np.mean(rs))


def linear_cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    hsic = np.linalg.norm(X.T @ Y) ** 2
    return float(hsic / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def main():
    df, fold = R.load_isic()
    nz = np.load(os.path.join(R.MISS_RES, "vl_nuisance.npz"))
    tY = R.logit(nz["q"]); tS = R.logit(np.clip(nz["q"] / nz["sigma"], 1e-6, 1 - 1e-6))
    runs = {}
    for f in sorted(os.listdir(FT)):
        if f.endswith(".npz"):
            z = np.load(os.path.join(FT, f))
            runs[f[:-4]] = z
    out = {"runs": {}, "cka": {}}
    for tag, z in runs.items():
        meta = json.loads(str(z["meta"]))
        probe, uni = z["probe"], z["is_uniform"]
        H = z["h"].astype(np.float32); lg = z["logit"]; w = z["fc_w"]
        r = {"meta": meta, "interv_layer3": json.loads(str(z["interv"])), "decision_contrast": {},
             "probe_r2": {}, "alignment": {}, "common_readout": {"target_Y": {}, "target_S": {}}}
        for c, col in R.CONCEPTS.items():
            T = R.SV.tertile(df, col).fillna(-1).to_numpy()[probe]
            r["decision_contrast"][c] = float(lg[uni & (T == 1)].mean() - lg[uni & (T == 0)].mean())
        Hu = H[uni]
        dirY = ridge_dir(Hu, tY[probe][uni]); dirS = ridge_dir(Hu, tS[probe][uni])
        for c in R.IMAGE_CONCEPTS:
            conc = df[R.CONCEPTS[c]].to_numpy(float)[probe][uni]
            wc = ridge_dir(Hu, conc)
            r["probe_r2"][c] = cv_r2(Hu, conc)
            r["alignment"][c] = cos(w, wc)
            r["common_readout"]["target_Y"][c] = cos(dirY, wc)
            r["common_readout"]["target_S"][c] = cos(dirS, wc)
        out["runs"][tag] = r
        R.log(f"{tag}: decision {({k[:5]: round(v, 2) for k, v in r['decision_contrast'].items()})} "
              f"A_k {({k[:5]: round(v, 3) for k, v in r['alignment'].items()})} "
              f"readout_Y {({k[:5]: round(v, 3) for k, v in r['common_readout']['target_Y'].items()})}")
    # CKA
    rng = np.random.default_rng(R.SEED)
    for mode in ("ft", "lp"):
        for s in range(3):
            a, b = f"{mode}_M0_s{s}", f"{mode}_M2_s{s}"
            if a in runs and b in runs:
                idx = rng.choice(np.flatnonzero(runs[a]["is_uniform"]), N_CKA, replace=False)
                out["cka"][f"{mode}_M0_vs_M2_s{s}"] = linear_cka(runs[a]["h"][idx].astype(np.float32),
                                                                  runs[b]["h"][idx].astype(np.float32))
        for reg in ("M0", "M2"):
            a, b = f"{mode}_{reg}_s0", f"{mode}_{reg}_s1"
            if a in runs and b in runs:
                idx = rng.choice(np.flatnonzero(runs[a]["is_uniform"]), N_CKA, replace=False)
                out["cka"][f"{mode}_{reg}_s0_vs_s1"] = linear_cka(runs[a]["h"][idx].astype(np.float32),
                                                                   runs[b]["h"][idx].astype(np.float32))
    # tổng hợp và phán quyết (§23)
    summ = {}
    for mode in ("ft", "lp"):
        seeds = sorted({int(t.split("_s")[-1]) for t in runs if t.startswith(mode)})
        if not seeds:
            continue
        sm = {}
        for key, getter in (("decision", lambda r, c: r["decision_contrast"][c]),
                            ("alignment", lambda r, c: r["alignment"][c]),
                            ("layer3", lambda r, c: r["interv_layer3"][c]["delta_mean"]),
                            ("readout_Y", lambda r, c: r["common_readout"]["target_Y"][c]),
                            ("readout_S", lambda r, c: r["common_readout"]["target_S"][c])):
            sm[key] = {}
            concepts = R.CONCEPTS if key == "decision" else R.IMAGE_CONCEPTS
            for c in concepts:
                v0 = [getter(out["runs"][f"{mode}_M0_s{s}"], c) for s in seeds if f"{mode}_M0_s{s}" in out["runs"]]
                v2 = [getter(out["runs"][f"{mode}_M2_s{s}"], c) for s in seeds if f"{mode}_M2_s{s}" in out["runs"]]
                if not v0 or not v2:
                    continue
                opp = [np.sign(a) != np.sign(b) for a, b in zip(v0, v2)]
                sm[key][c] = {"M0": v0, "M2": v2, "opposite_sign_all_seeds": bool(all(opp)),
                              "n_seeds_opposite": int(sum(opp))}
        sm["probe_r2"] = {c: {"M0": [out["runs"][f"{mode}_M0_s{s}"]["probe_r2"][c] for s in seeds if f"{mode}_M0_s{s}" in out["runs"]],
                              "M2": [out["runs"][f"{mode}_M2_s{s}"]["probe_r2"][c] for s in seeds if f"{mode}_M2_s{s}" in out["runs"]]}
                          for c in R.IMAGE_CONCEPTS}
        summ[mode] = sm
    verdict = {}
    if "ft" in summ:
        n_dec = sum(v["opposite_sign_all_seeds"] for v in summ["ft"]["decision"].values())
        n_rep = sum(v["opposite_sign_all_seeds"] for v in summ["ft"]["readout_Y"].values())
        n_rep_S = sum(v["opposite_sign_all_seeds"] for v in summ["ft"]["readout_S"].values())
        n_l3 = sum(v["opposite_sign_all_seeds"] for v in summ["ft"]["layer3"].values())
        verdict = {"ft_decision_reversed_concepts": int(n_dec),
                   "ft_layer3_reversed_concepts": int(n_l3),
                   "ft_common_readout_Y_reversed_concepts": int(n_rep),
                   "ft_common_readout_S_reversed_concepts": int(n_rep_S),
                   "case": ("A: representation reversal" if max(n_rep, n_rep_S) >= 2
                            else ("B: decision/learning reversal" if n_dec >= 2 else "no consistent reversal"))}
    out["summary"] = summ; out["verdict"] = verdict
    R.log(f"verdict: {verdict}")
    R.log(f"CKA: {out['cka']}")
    R.save_json(out, "vr5_representation.json")


if __name__ == "__main__":
    main()
