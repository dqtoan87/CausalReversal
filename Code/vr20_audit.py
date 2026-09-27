#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR20 — Rà soát và độ không đồng nhất.

(a) Độ không đồng nhất giữa cơ sở của log B_V (từ vr16, 5 cơ sở ước lượng được): Cochran Q, I², tau² và ước lượng
    gộp DerSimonian–Laird.
(b) Rà soát các giá trị gần trùng nhau giữa concept: tương quan Spearman và độ trùng tertile trên giữa các cặp
    (color, contrast) và (asymmetry, border); in giá trị đủ chữ số để loại trừ việc dùng nhầm cột.
(c) Bỏ lần lượt từng cơ sở (leave-one-site-out): loại bệnh nhân của một cơ sở khỏi train và test, huấn luyện lại M0 và
    M2 (3 seed, hai họ chính), báo Δ của ba concept chính trên phần test còn lại.
(d) Rà soát PAD-UFES-20: số ảnh, tổn thương, bệnh nhân; ánh xạ lớp; số đặc trưng được đánh giá ở hai phân tích.

Out -> Result/vr20_audit.json
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import vr_common as R
import gpu_setup  # noqa: F401  (ép GPU, giới hạn luồng)

PRIMARY = ["color_variegation", "size", "lesion_skin_contrast"]


def part_a():
    import json
    m = json.load(open(os.path.join(R.RES, "vr16_mechanism.json")))["by_site"]
    out = {}
    for c in PRIMARY:
        v = [(x["log_BV"], x["se"]) for x in m[c]["sites"].values() if x["log_BV"] < 8]
        y = np.array([a for a, _ in v]); se = np.array([b for _, b in v]); w = 1 / se ** 2
        mu_fe = np.sum(w * y) / w.sum(); Q = float(np.sum(w * (y - mu_fe) ** 2)); k = len(y)
        tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
        wr = 1 / (se ** 2 + tau2); mu = float(np.sum(wr * y) / wr.sum()); se_mu = float(np.sqrt(1 / wr.sum()))
        I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
        out[c] = {"k": k, "Q": Q, "df": k - 1, "I2": I2, "tau2": tau2, "pooled_DL": mu, "pooled_ci": [mu - 1.96 * se_mu, mu + 1.96 * se_mu],
                  "range": [float(y.min()), float(y.max())]}
        R.log(f"(a) {c}: k={k} Q={Q:.1f} I2={I2:.2f} tau2={tau2:.3f} pooled={mu:.2f} [{mu - 1.96 * se_mu:.2f}, {mu + 1.96 * se_mu:.2f}]")
    return out


def part_b(df):
    out = {}
    for a, b in (("color_variegation", "lesion_skin_contrast"), ("asymmetry", "border_irregularity"), ("color_variegation", "size")):
        xa, xb = df[R.CONCEPTS[a]].astype(float), df[R.CONCEPTS[b]].astype(float)
        ta, tb = R.SV.tertile(df, R.CONCEPTS[a]), R.SV.tertile(df, R.CONCEPTS[b])
        up = ((ta == 1) & (tb == 1)).sum() / max(((ta == 1) | (tb == 1)).sum(), 1)
        out[f"{a}|{b}"] = {"spearman": float(spearmanr(xa, xb, nan_policy="omit").statistic),
                            "jaccard_upper_tertile": float(up), "columns": [R.CONCEPTS[a], R.CONCEPTS[b]]}
        R.log(f"(b) {a} vs {b}: Spearman {out[f'{a}|{b}']['spearman']:.3f}, Jaccard upper {up:.3f}")
    return out


def part_c(df, fold):
    import vl_models as M
    out = {}
    Y = df.Y.to_numpy()
    Tc = {c: R.SV.tertile(df, R.CONCEPTS[c]).fillna(-1).to_numpy() for c in PRIMARY}
    for kind in ("tabular", "image"):
        if kind == "tabular":
            X, _ = R.VL.tabular_X(df); X = X.astype(np.float32); hid = 128
        else:
            z = np.load(os.path.join(R.EMB, "isic2024.npz")); E = z["emb"].astype(np.float32)
            X = ((E - E.mean(0)) / (E.std(0) + 1e-6)).astype(np.float32); hid = 256
        S = df.S.to_numpy(np.float32)
        A = dict(x=X, Y=Y.astype(np.float32), F=S, S=S, sigma=np.ones(len(df), np.float32), L=np.ones(len(df), np.float32),
                 U=np.ones(len(df), np.float32), m=np.zeros(len(df), np.float32))
        res = {}
        for site in sorted(df["attribution"].unique()):
            keep = (df["attribution"] != site).to_numpy()
            tr = np.flatnonzero((fold == "train") & keep); va = np.flatnonzero((fold == "val") & keep)
            te = np.flatnonzero((fold == "test") & keep)
            d = {c: [] for c in PRIMARY}
            for seed in range(3):
                m2 = M.train("erm_verified", A, tr, va, hid=hid, seed=seed)
                m0 = M.train("erm_y", A, tr, va, hid=hid, seed=seed)
                l0 = R.logit(M.predict(m0, X[te])["pD"]); l2 = R.logit(M.predict(m2, X[te])["pD"])
                for c in PRIMARY:
                    T = Tc[c][te]
                    d[c].append([float(l0[T == 1].mean() - l0[T == 0].mean()), float(l2[T == 1].mean() - l2[T == 0].mean())])
            res[site[:40]] = {c: np.mean(v, 0).tolist() for c, v in d.items()}
            R.log(f"(c) {kind} without {site[:30]}: " + " ".join(f"{c[:5]} {res[site[:40]][c][0]:+.2f}/{res[site[:40]][c][1]:+.2f}" for c in PRIMARY))
        out[kind] = res
    return out


def part_d():
    m = pd.read_csv(os.path.join(R.VL.DATA, "PAD-UFES-20/metadata.csv"))
    out = {"images": int(len(m)), "lesions": int(m.lesion_id.nunique()), "patients": int(m.patient_id.nunique()),
           "images_per_lesion": {str(k): int(v) for k, v in m.groupby("lesion_id").size().value_counts().sort_index().items()},
           "diagnosis_counts": m.diagnostic.value_counts().to_dict(),
           "malignant_classes": ["BCC", "SCC", "MEL"], "non_malignant_classes": ["ACK", "NEV", "SEK"],
           "biopsied_images": int(m.biopsed.astype(str).str.upper().eq("TRUE").sum()),
           "association_features": 11, "association_feature_list": "6 symptoms, age tertile, 4 image measures",
           "learner_features_evaluated": 10, "learner_inputs_clinical": "age, body region, 6 symptoms"}
    R.log(f"(d) PAD: {out['images']} images, {out['lesions']} lesions, {out['patients']} patients")
    return out


def main():
    df, fold = R.load_isic()
    out = {"site_heterogeneity": part_a(), "concept_overlap": part_b(df), "pad": part_d()}
    R.save_json(out, "vr20_audit.json")
    out["leave_one_site_out"] = part_c(df, fold)
    R.save_json(out, "vr20_audit.json")


if __name__ == "__main__":
    main()
