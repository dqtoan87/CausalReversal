#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR6 — PAD-UFES-20 là trường hợp biên của Định lý 4 (proposal_v3 §17).

Ở PAD-UFES, 100% tổn thương ác tính (BCC, SCC, MEL) được sinh thiết theo thiết kế cohort, nên
A = π₁¹/π₀¹ = 1 CHÍNH XÁC. Định lý 4 rút gọn thành
    log OR_{D|S} = log OR_D − log B,     B = π₁⁰/π₀⁰ = tỉ số sinh thiết của tổn thương lành,
với mọi đại lượng đều quan sát được (D lấy từ chẩn đoán: mô bệnh học nếu S=1, lâm sàng nếu S=0).
Đảo dấu ⇔ |log B| > |log OR_D| và cùng dấu. Đây là phép thử "khoảng cách tới biên pha" trên một cohort có
cơ chế xác minh khác ISIC-2024.

Đặc trưng: sáu triệu chứng nhị phân (itch, grew, hurt, changed, bleed, elevation; UNK bị loại theo từng đặc
trưng), tuổi (tertile), bốn proxy ảnh đã được xác nhận của vl0 (color_var, contrast, asym, border; size bị loại
theo §13). diameter_1 bị loại: thiếu ở 84% tổn thương chưa sinh thiết, tức thiếu phụ thuộc chính S.
Khoảng tin cậy: bootstrap 2,000 lần theo bệnh nhân.

Out -> Result/vr6_pad_boundary.json
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import vr_common as R

MAL = ["MEL", "BCC", "SCC"]
BIN = ["itch", "grew", "hurt", "changed", "bleed", "elevation"]
IMG = {"color_var": "color_variegation", "contrast": "lesion_skin_contrast", "asym": "asymmetry",
       "border": "border_irregularity"}


def exposure_table():
    m = pd.read_csv(os.path.join(R.VL.DATA, "PAD-UFES-20/metadata.csv"))
    z = np.load(os.path.join(R.EMB, "padufes.npz"))
    m = m.set_index("img_id").loc[z["ids"]].reset_index()
    names = list(z["concept_names"])
    E = {}
    for f in BIN:
        v = m[f].astype(str).str.upper()
        E[f] = np.where(v == "TRUE", 1.0, np.where(v == "FALSE", 0.0, np.nan))
    def tert(x):
        q1, q3 = np.nanquantile(x, [1 / 3, 2 / 3])
        return np.where(x <= q1, 0.0, np.where(x >= q3, 1.0, np.nan))
    E["age_tertile"] = tert(m["age"].to_numpy(float))
    for k in IMG:
        E[f"img_{k}"] = tert(z["concepts"][:, names.index(k)].astype(float))
    D = m["diagnostic"].isin(MAL).to_numpy().astype(float)
    S = m["biopsed"].astype(str).str.upper().eq("TRUE").to_numpy().astype(float)
    return E, D, S, m["patient_id"].to_numpy()


def stats(t, D, S):
    ok = np.isfinite(t)
    t, D, S = t[ok], D[ok], S[ok]
    def odds(p):
        return p / (1 - p)
    pD = [D[t == k].mean() for k in (0, 1)]
    pDS = [D[(t == k) & (S == 1)].mean() for k in (0, 1)]
    pi0 = [S[(t == k) & (D == 0)].mean() for k in (0, 1)]
    pi1 = [S[(t == k) & (D == 1)].mean() for k in (0, 1)]
    return {"logOR_D": np.log(odds(pD[1]) / odds(pD[0])), "logOR_DS": np.log(odds(pDS[1]) / odds(pDS[0])),
            "logB": np.log(pi0[1] / pi0[0]), "logA": np.log(pi1[1] / pi1[0]), "n": int(ok.sum())}


def main():
    E, D, S, pid = exposure_table()
    assert S[D == 1].min() == 1.0, "PAD-UFES: every malignant lesion must be biopsied (A = 1)"
    rng = np.random.default_rng(R.SEED)
    pats = np.unique(pid); idx_by = {p: np.flatnonzero(pid == p) for p in pats}
    out = {"n": len(D), "malignant": int(D.sum()), "biopsied": int(S.sum()), "features": {}}
    for f, t in E.items():
        st = stats(t, D, S)
        th, de = st["logOR_D"], st["logB"] - st["logA"]
        boots = []
        for _ in range(2000):
            ii = np.concatenate([idx_by[p] for p in rng.choice(pats, len(pats))])
            try:
                b = stats(t[ii], D[ii], S[ii])
                boots.append([b["logOR_D"], b["logOR_DS"], b["logB"]])
            except ZeroDivisionError:
                continue
        bt = np.array(boots); bt = bt[np.isfinite(bt).all(1)]
        rev_b = np.mean([(x[1] * x[0] < 0) for x in bt])
        ci = lambda j: [float(np.percentile(bt[:, j], 2.5)), float(np.percentile(bt[:, j], 97.5))]
        from vr1_reversal_theory import classify
        out["features"][f] = {**st, "identity_residual": st["logOR_DS"] - (th - de),
                              "category": classify(th, de), "margin_to_boundary": abs(de) - abs(th),
                              "ci_logOR_D": ci(0), "ci_logOR_DS": ci(1), "ci_logB": ci(2),
                              "bootstrap_prob_reversal": float(rev_b)}
        R.log(f"{f:16s} logOR_D={th:+.2f} logB={st['logB']:+.2f} logOR_D|S={st['logOR_DS']:+.2f} "
              f"-> {out['features'][f]['category']:13s} P(rev)={rev_b:.2f} resid={out['features'][f]['identity_residual']:.1e}")
    R.save_json(out, "vr6_pad_boundary.json")


if __name__ == "__main__":
    main()
