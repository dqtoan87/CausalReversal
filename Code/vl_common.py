#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vl_common — module dùng chung cho Staged Verification Causal Learning (SVCL, proposal_v2.md).

Nội dung:
  * đường dẫn, bảng đặc trưng TBP, chia tách theo bệnh nhân (đóng băng, ghi ra Result/vl_split.json)
  * khoảng định danh P(D=1 | x) theo Định lý 3 (dùng lại được ở vl1, vl2, vl3)
  * các độ đo: AUROC, ECE, độ phủ khoảng, rủi ro xấu nhất trên tập hình chữ nhật ℋ
  * mô hình SVCL và các baseline (torch), cùng một vòng huấn luyện

Quy ước: D là bệnh thật, F gắn cờ, S mô bệnh học, Y = D·S. Mọi khoảng định danh ở đây là khoảng cho
xác suất có điều kiện p(x) = P(D=1 | x), không phải cho nhãn của một tổn thương riêng lẻ.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

import sv_common as SV

HERE = SV.HERE
RES = SV.RES
DATA = os.environ.get("CAUSALREVERSAL_DATA", "/mnt/data2/Toan/Causal_ML_XAI/data")   # thư mục chứa data_ISIC2024/ và PAD-UFES-20
EMB = os.path.join(RES, "embeddings")
SEED = 42
log = SV.log
save_json = SV.save_json

# Đặc trưng TBP dùng làm x dạng bảng: mọi cột tbp_lv_* số, trừ toạ độ cơ thể và các điểm số DNN
# của chính hệ TBP (hai điểm này là đầu ra của một mô hình khác, không phải đo lường ngoại hình).
TBP_DROP = {"tbp_lv_x", "tbp_lv_y", "tbp_lv_z", "tbp_lv_dnn_lesion_confidence", "tbp_lv_location",
            "tbp_lv_location_simple"}
# Năm concept của proposal_v2 §10, đúng cột của v3.
CONCEPTS = {"color_variegation": "tbp_lv_norm_color", "size": "tbp_lv_areaMM2",
            "lesion_skin_contrast": "tbp_lv_deltaLBnorm", "asymmetry": "tbp_lv_symm_2axis",
            "border_irregularity": "tbp_lv_norm_border"}


# ----------------------------------------------------------------------------------
# Dữ liệu ISIC-2024
# ----------------------------------------------------------------------------------
def tbp_columns(df):
    return [c for c in df.columns if c.startswith("tbp_lv_") and c not in TBP_DROP
            and pd.api.types.is_numeric_dtype(df[c])]


def tabular_X(df):
    """x dạng bảng = đặc trưng TBP (chuẩn hoá) + Z (tuổi, giới, vị trí, cơ sở, ITA)."""
    cols = tbp_columns(df)
    T = df[cols].astype(float)
    T = (T - T.mean()) / T.std().replace(0, 1)
    T = T.fillna(0.0)
    Z = SV.build_Z(df)
    return np.hstack([T.to_numpy(np.float32), Z.to_numpy(np.float32)]), cols + list(Z.columns)


def make_split(df, frac=(0.6, 0.2, 0.2), seed=SEED):
    """Chia theo bệnh nhân, phân tầng theo việc bệnh nhân có ca ác tính hay không. Đóng băng lần đầu."""
    path = os.path.join(RES, "vl_split.json")
    if os.path.exists(path):
        sp = json.load(open(path))
    else:
        rng = np.random.default_rng(seed)
        g = df.groupby(SV.GROUP)["Y"].max()
        sp = {}
        for lab in (0, 1):
            p = g[g == lab].index.to_numpy().copy(); rng.shuffle(p)
            a, b = int(frac[0] * len(p)), int((frac[0] + frac[1]) * len(p))
            for name, part in (("train", p[:a]), ("val", p[a:b]), ("test", p[b:])):
                sp.setdefault(name, []).extend(part.tolist())
        os.makedirs(RES, exist_ok=True)
        json.dump(sp, open(path, "w"))
        log(f"wrote {path}")
    fold = pd.Series("none", index=df.index)
    for name in ("train", "val", "test"):
        fold[df[SV.GROUP].isin(set(sp[name]))] = name
    assert (fold != "none").all()
    return fold.to_numpy()


# ----------------------------------------------------------------------------------
# Định lý 2 và 3
# ----------------------------------------------------------------------------------
def B_interval(v0, v1, pL, pU):
    """Định lý 2. B = a₁/a₀ với aₜ = Pr(V=1|t)/(1−pₜ). Khi pₜ ∈ [pL_t, pU_t]:
    B ∈ [(v₁/v₀)(1−pU₀)/(1−pL₁), (v₁/v₀)(1−pL₀)/(1−pU₁)]. Sắc: hai đầu mút đạt được tại góc hộp."""
    r = v1 / v0
    return r * (1 - pU[0]) / (1 - pL[1]), r * (1 - pL[0]) / (1 - pU[1])


def disease_interval(q, sigma, s_min):
    """Định lý 3. Với q(x) = P(Y=1|x), σ(x) = P(S=1|x) và s(x) = P(S=1|D=1,x) ≥ s_min:
        p(x) = q(x)/s(x) ∈ [ q(x), min( q(x)/s_min, q(x) + 1 − σ(x) ) ].
    Cận trên thứ hai đến từ P(D=1, S=0 | x) ≤ P(S=0 | x). Cả hai đầu mút đạt được, nên khoảng sắc."""
    L = np.asarray(q, float)
    U = np.minimum(L / s_min, L + 1 - np.asarray(sigma, float))
    return L, np.clip(U, L, 1.0)


# ----------------------------------------------------------------------------------
# Độ đo
# ----------------------------------------------------------------------------------
def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def ece(y, p, bins=15):
    """ECE theo phân vị (bin đều số mẫu), vì p lệch nặng về 0 khi bệnh hiếm."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def worst_case_risk(p_hat, L, U):
    """Rủi ro log-loss xấu nhất trên ℋ = Π_i [L_i, U_i]. CE(p, f) tuyến tính theo p nên cực đại tại
    một đầu mút. ℋ hình chữ nhật là tập NGOÀI của tập thừa nhận (bỏ ràng buộc ghép giữa các tổn thương)."""
    f = np.clip(np.asarray(p_hat, float), 1e-7, 1 - 1e-7)
    ce = lambda p: -(p * np.log(f) + (1 - p) * np.log(1 - f))
    return float(np.mean(np.maximum(ce(L), ce(U))))


def coverage(p_hat, L, U, rel=1e-3):
    p = np.asarray(p_hat)
    return float(np.mean((p >= L * (1 - rel)) & (p <= U * (1 + rel))))


# ----------------------------------------------------------------------------------
# Nuisance cross-fit: q(x) = P(Y=1|x), σ(x) = P(S=1|x)
# ----------------------------------------------------------------------------------
def crossfit_prob(X, y, groups, n_folds=5, seed=SEED):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    out = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n_folds).split(X, y, groups):
        m = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.05,
                                           l2_regularization=1.0, random_state=seed).fit(X[tr], y[tr])
        out[te] = m.predict_proba(X[te])[:, 1]
    return out
