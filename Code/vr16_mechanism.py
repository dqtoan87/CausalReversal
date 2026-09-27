#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR16 — Phân tách cơ chế xác minh (bệnh nhân / cơ sở) và kiểm tra overlap.

(a) Trong và giữa bệnh nhân. Trên các tổn thương nhãn âm (Y=0), hồi quy logistic S theo độ lệch của concept so với
    trung bình của bệnh nhân (trong bệnh nhân) và theo trung bình bệnh nhân (giữa bệnh nhân), kèm hiệp biến
    (dạng Mundlak). Đo xem xác minh phụ thuộc ngoại hình tổn thương bên trong một bệnh nhân, hay chủ yếu theo bệnh
    nhân và bối cảnh theo dõi. Sai số chuẩn cụm theo bệnh nhân.
(b) Theo cơ sở thu nhận. Tỉ số xác minh trong nhóm nhãn âm giữa hai tertile, B_V = P(S=1|Y=0,t=1)/P(S=1|Y=0,t=0),
    hiệu chỉnh tuổi, giới, vị trí giải phẫu và tông da, cho từng cơ sở; báo cáo khoảng và tính nhất quán về dấu.
(c) Overlap. M2 chỉ học từ tổn thương đã xác minh nhưng được đánh giá trên toàn bộ test. Ước lượng σ̂(x) = P(S=1|x)
    (nuisance tabular không rò rỉ test); vùng có hỗ trợ là σ̂ ≥ phân vị 1% của σ̂ trên tổn thương train đã xác
    minh. Tính lại Δ_M0, Δ_M2 của hai họ head đóng băng trên (i) toàn bộ test, (ii) vùng có hỗ trợ, (iii) tổn thương
    test được gắn cờ; bootstrap bệnh nhân 2,000 lần.

Out -> Result/vr16_mechanism.json
"""
from __future__ import annotations

import os

import numpy as np

import vr_common as R
from vr9_closing import cohort_cuts, delta_boot, summarise

SV = R.SV


def part_a(df):
    neg = df[df.Y == 0].copy()
    Z = SV.build_Z(neg, cat_cols=["sex", "anatom_site_general", "attribution", "ITA_stratum"])
    g = neg[SV.GROUP].to_numpy()
    out = {}
    for c, col in R.CONCEPTS.items():
        x = neg[col].astype(float)
        zc = ((x - df[col].mean()) / df[col].std()).fillna(0.0)
        pm = zc.groupby(neg[SV.GROUP]).transform("mean")
        X = np.column_stack([np.ones(len(neg)), (zc - pm).to_numpy(), pm.to_numpy(), Z.to_numpy()])
        b, IF, _ = SV.logit_if(X, neg.S.to_numpy(), g)
        cov = SV.joint_cov([IF, IF], [1, 2])
        out[c] = {"within": float(b[1]), "within_se": float(np.sqrt(cov[0, 0])),
                  "between": float(b[2]), "between_se": float(np.sqrt(cov[1, 1]))}
        R.log(f"(a) {c:22s} within {b[1]:+.3f} ± {np.sqrt(cov[0, 0]):.3f}   between {b[2]:+.3f} ± {np.sqrt(cov[1, 1]):.3f}")
    return out


def part_b(df):
    out = {}
    for c, col in R.CONCEPTS.items():
        T = SV.tertile(df, col)
        rows = {}
        for site in sorted(df["attribution"].unique()):
            m = (df["attribution"] == site) & T.notna() & (df.Y == 0)
            d = df[m]; t = T[m].to_numpy()
            if d.S.sum() < 10 or (t == 1).sum() == 0 or (t == 0).sum() == 0:
                continue
            Zs = SV.build_Z(d, cat_cols=["sex", "anatom_site_general", "ITA_stratum"], min_count=20)
            X = np.column_stack([np.ones(len(d)), t, Zs.to_numpy()])
            b, IF, _ = SV.logit_if(X, d.S.to_numpy(), d[SV.GROUP].to_numpy())
            se = float(np.sqrt(SV.joint_cov([IF], [1])[0, 0]))
            rows[site[:40]] = {"log_BV": float(b[1]), "se": se, "n_verified": int(d.S.sum())}
        vals = [v["log_BV"] for v in rows.values()]
        out[c] = {"sites": rows, "min": float(min(vals)), "max": float(max(vals)),
                  "n_sites": len(vals), "n_positive": int(sum(v > 0 for v in vals))}
        R.log(f"(b) {c:22s} log B_V by site from {min(vals):+.2f} to {max(vals):+.2f}; positive in {out[c]['n_positive']}/{len(vals)}")
    return out


def part_c(df, fold):
    nz = np.load(os.path.join(R.MISS_RES, "vl_nuisance.npz"))
    sig = nz["sigma"]
    trS = (fold == "train") & (df.S.to_numpy() == 1)
    thr = float(np.quantile(sig[trS], 0.01))
    heads = np.load(os.path.join(R.RES, "closing_heads.npz"))
    te = heads["test_idx"]
    pid_all = df[SV.GROUP].to_numpy()
    subsets = {"all_test": np.ones(len(te), bool), "supported": sig[te] >= thr, "flagged": df.F.to_numpy()[te] == 1}
    out = {"support_threshold": thr, "fraction_retained": {k: float(v.mean()) for k, v in subsets.items()}, "families": {}}
    rng = np.random.default_rng(R.SEED)
    for kind in ("tabular", "image"):
        l0 = [heads[f"{kind}_M0_s{s}"] for s in range(3)]; l2 = [heads[f"{kind}_M2_s{s}"] for s in range(3)]
        fam = {}
        for name, m in subsets.items():
            ix = te[m]; pid = pid_all[ix]
            pats = np.unique(pid_all); pat_index = {p: i for i, p in enumerate(pats)}
            tp = np.unique(pid)
            cnt = np.zeros((2000, len(pats)))
            for b in range(2000):
                np.add.at(cnt[b], [pat_index[p] for p in rng.choice(tp, len(tp))], 1.0)
            res = {}
            for c in ("color_variegation", "size", "lesion_skin_contrast"):
                col = R.CONCEPTS[c]; xall = df[col].to_numpy(float); cuts = cohort_cuts(xall)
                d0 = delta_boot([l[m] for l in l0], xall[ix], pid, "tertile", cuts, cnt, pat_index)
                d2 = delta_boot([l[m] for l in l2], xall[ix], pid, "tertile", cuts, cnt, pat_index)
                res[c] = summarise(d0, d2)
                R.log(f"(c) {kind} {name:9s} {c:22s} Δ_M0={res[c]['M0']:+.2f} {np.round(res[c]['M0_ci'], 2)} "
                      f"Δ_M2={res[c]['M2']:+.2f} {np.round(res[c]['M2_ci'], 2)} P(sign≠)={res[c]['p_sign_differ']:.3f}")
            fam[name] = res
        out["families"][kind] = fam
    return out


def main():
    df, fold = R.load_isic()
    out = {"within_between": part_a(df), "by_site": part_b(df), "overlap": part_c(df, fold)}
    R.save_json(out, "vr16_mechanism.json")


if __name__ == "__main__":
    main()
