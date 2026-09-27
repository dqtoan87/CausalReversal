#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR15 — Can thiệp có kiểm soát vào chính sách chọn mẫu (dose–response trên cơ chế xác minh).

Giữ cố định quần thể và tập ác tính (mỗi lần lặp: 178 trong 222 ca ác tính train, dùng chung cho mọi liều). Tổn thương
lành huấn luyện (320) được rút từ TOÀN BỘ tổn thương lành train với xác suất ∝ exp(η·z_k), trong đó z_k là concept k
chuẩn hoá. η là "liều" chọn lọc theo concept do người thí nghiệm đặt: η = 0 là rút ngẫu nhiên, η > 0 là ưu tiên
tổn thương lành có concept cao, như khi lâm sàng sinh thiết tổn thương trông bất thường.

Dự đoán chính xác cho learner Bayes-tối ưu (Proposition 4 với π⁰(x) ∝ exp(η z_k), π¹ hằng số):
    logit f*_η(x) = logit f*_0(x) − η z_k(x) + hằng số, nên
    Δ_j(η) = Δ_j(0) − η·[E(z_k | T_j=1) − E(z_k | T_j=0)]   cho MỌI concept j (kể cả lan sang concept khác),
và concept k đổi dấu tại η* = Δ_k(0) / [E(z_k | T_k=1) − E(z_k | T_k=0)].
Đo: Δ học được trên quần thể test cố định cho η ∈ {−1, −0.5, 0, 0.5, 1, 1.5, 2, 3}, 10 lần lặp, hai họ learner
đặc trưng đóng băng, ba concept đích (color, size, lesion-skin contrast). So độ dốc và điểm cắt 0 với dự đoán.

Out -> Result/vr15_dose_response.json (+ .partial)
"""
from __future__ import annotations

import json
import os

import numpy as np

import vr_common as R

ETAS = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
N_REP, N_MAL, N_BEN = 10, 178, 320
TARGETS = ["color_variegation", "size", "lesion_skin_contrast"]


def features(df, kind):
    if kind == "tabular":
        X, _ = R.VL.tabular_X(df)
        return X.astype(np.float32), 128
    z = np.load(os.path.join(R.EMB, "isic2024.npz"))
    E = z["emb"].astype(np.float32)
    return ((E - E.mean(0)) / (E.std(0) + 1e-6)).astype(np.float32), 256


def weighted_choice(rng, pool, w, n):
    """Rút n phần tử không hoàn lại với xác suất ∝ w (Gumbel top-k, chính xác cho lấy mẫu có trọng số tuần tự)."""
    key = np.log(w) - np.log(-np.log(rng.random(len(pool))))
    return pool[np.argsort(-key)[:n]]


def main():
    import vl_models as M
    df, fold = R.load_isic()
    Y = df.Y.to_numpy()
    Zs = {c: ((df[col] - df[col].mean()) / df[col].std()).fillna(0.0).to_numpy() for c, col in R.CONCEPTS.items()}
    te = np.flatnonzero(fold == "test")
    Tc = {c: R.SV.tertile(df, col).fillna(-1).to_numpy()[te] for c, col in R.CONCEPTS.items()}
    # dự đoán Bayes: độ dịch trên mỗi đơn vị η của Δ_j khi chọn lọc theo z_k
    shift = {k: {j: float(Zs[k][te][Tc[j] == 1].mean() - Zs[k][te][Tc[j] == 0].mean()) for j in R.CONCEPTS} for k in TARGETS}
    pools = {p: {"mal": np.flatnonzero((fold == p) & (Y == 1)), "ben": np.flatnonzero((fold == p) & (Y == 0))}
             for p in ("train", "val")}
    nva_m = int(round(0.8 * len(pools["val"]["mal"]))); nva_b = 111
    part = os.path.join(R.RES, "vr15_dose_response.json.partial")
    res = json.load(open(part)) if os.path.exists(part) else {}
    for kind in ("tabular", "image"):
        X, hid = features(df, kind)
        tr, va = np.flatnonzero(fold == "train"), np.flatnonzero(fold == "val")
        for k in TARGETS:
            key = f"{kind}|{k}"
            done = res.get(key, {})
            for r in range(N_REP):
                if str(r) in done:
                    continue
                rng = np.random.default_rng(1000 * r + TARGETS.index(k))
                mal_tr = rng.choice(pools["train"]["mal"], N_MAL, replace=False)
                mal_va = rng.choice(pools["val"]["mal"], nva_m, replace=False)
                rep = {}
                for eta in ETAS:
                    btr = weighted_choice(rng, pools["train"]["ben"], np.exp(eta * Zs[k][pools["train"]["ben"]]), N_BEN)
                    bva = weighted_choice(rng, pools["val"]["ben"], np.exp(eta * Zs[k][pools["val"]["ben"]]), nva_b)
                    sel = np.zeros(len(df), np.float32); sel[np.concatenate([mal_tr, btr, mal_va, bva])] = 1
                    arrs = dict(x=X, Y=df.Y.to_numpy(np.float32), F=sel, S=sel, sigma=np.ones(len(df), np.float32),
                                L=np.ones(len(df), np.float32), U=np.ones(len(df), np.float32), m=np.zeros(len(df), np.float32))
                    mdl = M.train("erm_verified", arrs, tr, va, hid=hid, seed=r)
                    lg = R.logit(M.predict(mdl, X[te])["pD"])
                    rep[str(eta)] = {j: float(lg[Tc[j] == 1].mean() - lg[Tc[j] == 0].mean()) for j in R.CONCEPTS}
                done[str(r)] = rep
                res[key] = done
                json.dump(res, open(part, "w"))
                R.log(f"{key} rep {r}: Δ_{k[:5]} by η " + " ".join(f"{e:+.1f}:{rep[str(e)][k]:+.2f}" for e in ETAS))
    # tóm tắt
    out = {"etas": ETAS, "n_rep": N_REP, "targets": TARGETS, "predicted_shift_per_eta": shift, "results": {}}
    for key, done in res.items():
        kind, k = key.split("|")
        M_ = {j: np.array([[done[r][str(e)][j] for e in ETAS] for r in done]) for j in R.CONCEPTS}
        mean = {j: M_[j].mean(0) for j in R.CONCEPTS}
        e = np.array(ETAS)
        slopes = {j: float(np.polyfit(e, mean[j], 1)[0]) for j in R.CONCEPTS}
        rep_slopes = np.array([np.polyfit(e, M_[k][i], 1)[0] for i in range(len(M_[k]))])
        d0 = float(mean[k][ETAS.index(0.0)])
        eta_star_pred = d0 / shift[k][k]
        # điểm cắt 0 quan sát: nội suy tuyến tính trên đường trung bình
        m = mean[k]; cross = None
        for i in range(len(e) - 1):
            if m[i] * m[i + 1] <= 0 and m[i] != m[i + 1]:
                cross = float(e[i] - m[i] * (e[i + 1] - e[i]) / (m[i + 1] - m[i])); break
        out["results"][key] = {
            "mean_delta_by_eta": {j: mean[j].tolist() for j in R.CONCEPTS},
            "q025": {j: np.percentile(M_[j], 2.5, axis=0).tolist() for j in R.CONCEPTS},
            "q975": {j: np.percentile(M_[j], 97.5, axis=0).tolist() for j in R.CONCEPTS},
            "observed_slope": slopes, "predicted_slope": {j: -shift[k][j] for j in R.CONCEPTS},
            "target_slope_replicate_range": [float(rep_slopes.min()), float(rep_slopes.max())],
            "delta_at_eta0": d0, "eta_star_predicted": float(eta_star_pred), "eta_star_observed": cross,
            "spillover_sign_agreement": int(sum(np.sign(slopes[j]) == np.sign(-shift[k][j]) for j in R.CONCEPTS))}
        R.log(f"{key}: slope obs {slopes[k]:+.2f} pred {-shift[k][k]:+.2f}; η* obs {cross} pred {eta_star_pred:.2f}; "
              f"spillover signs {out['results'][key]['spillover_sign_agreement']}/5")
    R.save_json(out, "vr15_dose_response.json")


if __name__ == "__main__":
    main()
