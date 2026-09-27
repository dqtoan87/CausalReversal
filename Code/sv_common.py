#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sv_common — module dùng chung cho mô hình Staged-Verification (SV) của Causal_XAI_v4.

Biến quan sát trên ISIC-2024 (401,059 tile, 1,042 bệnh nhân):
  F = 1{lesion_id có mặt}   tổn thương được bác sĩ gắn cờ "lesion of interest" (22,058)
  S = 1{iddx_2 có mặt}      tổn thương có chẩn đoán mô bệnh học (1,068), S ⊂ F
  Y = target                nhãn ác tính ghi nhận, Y = D·S (393)
  V = S·(1−Y)               tổn thương ĐÃ xác minh và không ác tính (675, gồm 114 indeterminate)

v3 (paper1, Statement 3) coi tỉ số xác minh π₁/π₀ là không quan sát được. Metadata thực ra ghi lại
cả hai giai đoạn chọn (F rồi S), nên phần của tỉ số đó đi qua đường "ngoại hình" ước lượng được
qua V. Module này chuẩn bị dữ liệu, các proxy và tertile giống hệt v3 (c1_tier1_ate, c41), cùng hai
bộ ước lượng có hàm ảnh hưởng theo bệnh nhân để lấy hiệp phương sai CHUNG giữa các mô hình:
logistic có hiệu chỉnh Z, và conditional logistic phân tầng theo bệnh nhân (chép từ v3 c41).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
RES = os.path.join(BASE, "Result")
DATA_CSV = os.path.join(os.environ.get("CAUSALREVERSAL_DATA", "/mnt/data2/Toan/Causal_ML_XAI/data"), "data_ISIC2024", "train-metadata.csv")
GROUP = "patient_id"
SEED = 42

# Đúng bảy proxy của v3 (c1_tier1_ate.CONCEPTS), cùng tên cột và vai trò.
PROXIES = [
    {"name": "color_variegation", "col": "tbp_lv_norm_color", "role": "primary"},
    {"name": "size", "col": "tbp_lv_areaMM2", "role": "primary"},
    {"name": "lesion_skin_contrast", "col": "tbp_lv_deltaLBnorm", "role": "primary"},
    {"name": "color_sd", "col": "tbp_lv_color_std_mean", "role": "exploratory"},
    {"name": "asymmetry", "col": "tbp_lv_symm_2axis", "role": "exploratory"},
    {"name": "border_irregularity", "col": "tbp_lv_norm_border", "role": "exploratory"},
    {"name": "benign_proxy", "col": "tbp_lv_nevi_confidence", "role": "negative_control"},
]
PRIMARY = [p["name"] for p in PROXIES if p["role"] == "primary"]

# Confounder cho mô hình hiệu chỉnh (giống v3) và biến thay đổi trong bệnh nhân (giống v3 c41).
NUM_CONF = ["age_approx"]
CAT_CONF = ["sex", "anatom_site_general", "attribution", "ITA_stratum"]
WITHIN_CONF = ["anatom_site_general", "ITA_stratum", "age_band"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def save_json(obj, name):
    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_np_default)
    log(f"wrote {path}")
    return path


def _np_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ----------------------------------------------------------------------------------
# Dữ liệu
# ----------------------------------------------------------------------------------
def compute_ita(df):
    """ITA trên da nền, chỉ dùng quartile tương đối (giống v3 c1.compute_ita)."""
    L, b = df["tbp_lv_Lext"].astype(float), df["tbp_lv_Bext"].astype(float)
    b = b.replace(0, np.nan)
    ita = np.degrees(np.arctan2(L - 50.0, b))
    q = pd.qcut(ita, 4, labels=["ITA_Q1_dark", "ITA_Q2", "ITA_Q3", "ITA_Q4_light"], duplicates="drop")
    return q.astype("object").fillna("ITA_missing")


def load(path=DATA_CSV):
    df = pd.read_csv(path, low_memory=False)
    df["Y"] = df["target"].astype(int)
    df["F"] = df["lesion_id"].notna().astype(int)
    df["S"] = df["iddx_2"].notna().astype(int)
    df["V"] = (df["S"] * (1 - df["Y"])).astype(int)
    df["indet"] = (df["iddx_1"] == "Indeterminate").astype(int)
    df["ITA_stratum"] = compute_ita(df)
    df["age_band"] = pd.to_numeric(df["age_approx"], errors="coerce")
    # Bất biến cấu trúc: Y ⊂ S ⊂ F. Nếu hỏng thì định nghĩa giai đoạn sai.
    assert (df["Y"] <= df["S"]).all() and (df["S"] <= df["F"]).all()
    return df


def tertile(df, col):
    """T = 1 tertile trên, 0 tertile dưới, NaN tertile giữa. Cắt trên TOÀN cohort như v3."""
    x = pd.to_numeric(df[col], errors="coerce")
    q1, q3 = x.quantile(1 / 3), x.quantile(2 / 3)
    T = pd.Series(np.nan, index=df.index)
    T[x.notna() & (x <= q1)] = 0.0
    T[x.notna() & (x >= q3)] = 1.0
    return T


def build_Z(d, cat_cols=CAT_CONF, min_count=0):
    parts = []
    for c in NUM_CONF:
        x = pd.to_numeric(d[c], errors="coerce")
        parts.append(((x.fillna(x.median()) - 50.0) / 10.0).rename(c))
        parts.append(x.isna().astype(float).rename(f"{c}__isna"))
    cat = d[cat_cols].astype("object").fillna("missing")
    du = pd.get_dummies(cat, prefix=cat_cols, drop_first=True).astype(float)
    if min_count:
        du = du.loc[:, du.sum(0) >= min_count]
    return pd.concat(parts + [du], axis=1)


# ----------------------------------------------------------------------------------
# Logistic hiệu chỉnh, với hàm ảnh hưởng gộp theo bệnh nhân
# ----------------------------------------------------------------------------------
def logit_if(X, y, groups, ridge=1e-6, maxiter=100):
    """Newton cho logistic. Trả (beta, IF) với IF là DataFrame: một hàng mỗi bệnh nhân,
    IF_i = H⁻¹ Σ_{j∈i} x_j (y_j − p_j), sao cho Cov(β̂) = Σ_i IF_i IF_iᵀ (sandwich cụm).
    ridge nhỏ chỉ để ổn định số khi có dummy gần tách biệt; không thay đổi ước lượng đáng kể."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    b = np.zeros(X.shape[1])
    b[0] = np.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6)) if np.allclose(X[:, 0], 1) else 0.0
    for _ in range(maxiter):
        eta = np.clip(X @ b, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        g = X.T @ (y - p) - ridge * b
        H = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(len(b))
        step = np.linalg.solve(H, g)
        b += step
        if np.max(np.abs(step)) < 1e-9:
            break
    eta = np.clip(X @ b, -30, 30); p = 1 / (1 + np.exp(-eta))
    H = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(len(b))
    Hinv = np.linalg.inv(H)
    sc = X * (y - p)[:, None]
    S = pd.DataFrame(sc).groupby(np.asarray(groups)).sum()
    IF = S @ Hinv.T
    IF.columns = range(len(b))
    return b, IF, p


def joint_cov(ifs, idx):
    """ifs: list IF DataFrame (theo bệnh nhân); idx: chỉ số hệ số lấy từ mỗi IF.
    Trả ma trận hiệp phương sai chung của các hệ số đã chọn, căn theo bệnh nhân (bệnh nhân vắng
    mặt trong một mô hình đóng góp 0). Hệ số hiệu chỉnh mẫu nhỏ G/(G−1)."""
    pats = sorted(set().union(*[set(f.index) for f in ifs]))
    M = np.column_stack([f.reindex(pats).fillna(0.0)[j].to_numpy() for f, j in zip(ifs, idx)])
    G = len(pats)
    return M.T @ M * G / max(G - 1, 1)


# ----------------------------------------------------------------------------------
# Conditional logistic (chép nguyên thuật toán từ Causal_XAI_v3/code/c41_within_patient.py)
# ----------------------------------------------------------------------------------
def _esp_and_incl(w, k):
    """e_k(w) và xác suất bao hàm p_i = w_i·e_{k−1}(w bỏ i)/e_k. Cùng DP prefix/suffix như v3
    c41, chỉ vector hoá theo bậc j (vòng lặp v3 thuần Python chậm khi k lớn, như với V)."""
    n = len(w)
    pre = np.zeros((k + 1, n + 1)); pre[0, :] = 1.0
    for i in range(1, n + 1):
        pre[1:, i] = pre[1:, i - 1] + w[i - 1] * pre[:-1, i - 1]
    suf = np.zeros((k + 1, n + 2)); suf[0, :] = 1.0
    for i in range(n, 0, -1):
        suf[1:, i] = suf[1:, i + 1] + w[i - 1] * suf[:-1, i + 1]
    ek = pre[k, n]
    if not np.isfinite(ek) or ek <= 0:
        return ek, np.zeros(n)
    # e_{k−1}(w bỏ i) = Σ_{j<k} pre[j, i−1]·suf[k−1−j, i+1]
    ekm1 = (pre[:k, :n] * suf[k - 1::-1, 2:n + 2]).sum(0)
    return ek, w * ekm1 / ek


def clogit_nll(beta, X, y, strata):
    eta = X @ beta
    nll = 0.0
    grad = np.zeros_like(beta)
    scores = []
    for idx in strata:
        e = eta[idx]; e = e - e.max()
        w = np.exp(e); yi = y[idx]; k = int(yi.sum())
        # k và n−k đối xứng: đổi vai để DP rẻ hơn khi ca nhiều hơn chứng
        ek, incl = _esp_and_incl(w, k)
        if ek <= 0 or not np.isfinite(ek):
            scores.append(np.zeros_like(beta)); continue
        nll -= float(e[yi == 1].sum() - np.log(ek))
        g = (X[idx] * (yi - incl)[:, None]).sum(0)
        grad -= g
        scores.append(g)
    return nll, grad, np.array(scores)


def clogit_prepare(d, T, outcome, adjust=True):
    """d: các hàng thuộc mẫu tertile; T: 0/1. Chỉ giữ tầng có cả hai lớp outcome. Trả
    (X, y, strata, patient_ids) với cột 0 là T."""
    d = d.copy(); d["_T"] = np.asarray(T, float)
    g = d.groupby(GROUP)[outcome].agg(["sum", "count"])
    keep = g[(g["sum"] > 0) & (g["sum"] < g["count"])].index
    d = d[d[GROUP].isin(keep)]
    cols = [d["_T"].rename("T")]
    if adjust:
        cat = d[WITHIN_CONF].astype("object").fillna("missing").astype(str)
        cols.append(pd.get_dummies(cat, prefix=WITHIN_CONF, drop_first=True).astype(float))
    X = pd.concat(cols, axis=1)
    y = d[outcome].to_numpy().astype(float)
    codes, uniq = pd.factorize(d[GROUP])
    order = np.argsort(codes, kind="stable")
    X = X.to_numpy()[order]; y = y[order]; codes = codes[order]
    bounds = np.searchsorted(codes, np.arange(codes.max() + 2))
    strata = [np.arange(bounds[i], bounds[i + 1]) for i in range(codes.max() + 1)]
    pids = [uniq[i] for i in range(codes.max() + 1)]
    keep_s = [i for i, s in enumerate(strata) if len(s) > 1]
    strata = [strata[i] for i in keep_s]; pids = [pids[i] for i in keep_s]
    if X.shape[1] > 1:
        var = np.zeros(X.shape[1])
        for s in strata:
            var += X[s].var(0) * len(s)
        drop = np.where(var < 1e-10)[0]; drop = drop[drop != 0]
        if len(drop):
            X = np.delete(X, drop, axis=1)
    return X, y, strata, pids


def clogit_if(X, y, strata, pids):
    """Khớp conditional logistic; trả (beta, IF theo bệnh nhân, n_strata, n_cases)."""
    p = X.shape[1]
    f = lambda b: clogit_nll(b, X, y, strata)[:2]
    r = minimize(f, np.zeros(p), jac=True, method="BFGS", options={"maxiter": 300, "gtol": 1e-6})
    beta = r.x
    _, _, S = clogit_nll(beta, X, y, strata)
    eps = 1e-5
    H = np.zeros((p, p))
    for j in range(p):
        bp = beta.copy(); bp[j] += eps
        bm = beta.copy(); bm[j] -= eps
        H[:, j] = (clogit_nll(bp, X, y, strata)[1] - clogit_nll(bm, X, y, strata)[1]) / (2 * eps)
    H = 0.5 * (H + H.T)
    Hinv = np.linalg.pinv(H)
    IF = pd.DataFrame(S @ Hinv.T, index=pids)
    return beta, IF, len(strata), int(y.sum())


def ci(est_log, se, z=1.959964):
    return [float(np.exp(est_log - z * se)), float(np.exp(est_log + z * se))]
