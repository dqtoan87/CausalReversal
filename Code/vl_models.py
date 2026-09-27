#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vl_models — SVCL và các baseline của proposal_v2 §5, §7, §8, §11.2, cùng một vòng huấn luyện và
bộ đánh giá. Dùng chung cho mô phỏng (vl2) và dữ liệu thật (vl3, vl4).

Phương pháp (khoá `method`):
  erm_y          ERM trên Y, toàn bộ tổn thương                         (M0)
  erm_flag       ERM trên Y, chỉ tổn thương đã gắn cờ F=1               (M1)
  erm_verified   ERM trên Y, chỉ tổn thương đã xác minh S=1              (M2, biopsy-only)
  ipw_verified   như M2, trọng số 1/σ̂(x)                                  (baseline 3)
  mi_pseudo      nhãn mềm: Y nếu S=1, m̂(x) = P̂(Y|x,S=1) nếu S=0          (baseline 4, MAR imputation)
  svcl           mô hình đề xuất, xem dưới
  svcl_noid      bỏ L_ID và L_rob                                          (ablation)
  svcl_nostage   bỏ L_stage
  svcl_nosep     bỏ L_sep
  svcl_norob     bỏ L_rob
  svcl_mar       s(x) cố định bằng σ̂(x): kiểm tra rút gọn của Định lý 5

SVCL. Hai encoder φ_D, φ_V. Head bệnh p_D = σ(g_D(H_D)); head xác minh ác tính
s(x) = s_min + (1 − s_min)·σ(g_s(H_V)) ∈ [s_min, 1]; head gắn cờ g_F(H_V); head sinh thiết g_S(H_V) trên F=1.
  L = BCE(Y, p_D·s)                               likelihood theo giai đoạn: Y = D·S, không điền nhãn D
    + λ_stage [BCE(F, g_F) + BCE_{F=1}(S, g_S)]    dữ liệu về cơ chế thiếu nhãn
    + λ_sep  ‖Cov(H_D, H_V)‖²_F / (d_D·d_V)         tách biểu diễn (HSIC tuyến tính)
    + λ_ID   dist²(log p_D, [log L, log U])        ràng buộc tập định danh (Định lý 3)
    + λ_rob  max_{p∈{L,U}} CE(p, p_D)              rủi ro xấu nhất trên ℋ = Π[L_i, U_i]
Ràng buộc s ≥ s_min trong likelihood chính là giả định L1; do đó cực tiểu của số hạng đầu thoả
p_D ∈ [q, q/s_min] điểm-theo-điểm, và p_D không bao giờ được gán nhãn giả cho tổn thương chưa xác minh.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

import vl_common as V

EPS = 1e-6
SVCL_VARIANTS = {
    "svcl": dict(stage=1.0, sep=1.0, id=1.0, rob=1.0),
    "svcl_noid": dict(stage=1.0, sep=1.0, id=0.0, rob=0.0),
    "svcl_nostage": dict(stage=0.0, sep=1.0, id=1.0, rob=1.0),
    "svcl_nosep": dict(stage=1.0, sep=0.0, id=1.0, rob=1.0),
    "svcl_norob": dict(stage=1.0, sep=1.0, id=1.0, rob=0.0),
    "svcl_mar": dict(stage=1.0, sep=0.0, id=0.0, rob=0.0),
}
BASELINES = ["erm_y", "erm_flag", "erm_verified", "ipw_verified", "mi_pseudo"]
ALL_METHODS = BASELINES + list(SVCL_VARIANTS)


def device():
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free > 2e9:
            return torch.device("cuda")
    return torch.device("cpu")


def mlp(d_in, d_hid, d_out, drop=0.1):
    return nn.Sequential(nn.Linear(d_in, d_hid), nn.GELU(), nn.Dropout(drop),
                         nn.Linear(d_hid, d_hid), nn.GELU(), nn.Dropout(drop), nn.Linear(d_hid, d_out))


class Plain(nn.Module):
    def __init__(self, d_in, hid):
        super().__init__()
        self.net = mlp(d_in, hid, 1)

    def forward(self, x):
        return {"logit_D": self.net(x).squeeze(-1)}


class SVCL(nn.Module):
    def __init__(self, d_in, hid, d_rep=64, s_min=0.5):
        super().__init__()
        self.s_min = s_min
        self.phi_D = mlp(d_in, hid, d_rep)
        self.phi_V = mlp(d_in, hid, d_rep)
        self.g_D = nn.Linear(d_rep, 1)
        self.g_s = nn.Linear(d_rep, 1)
        self.g_F = nn.Linear(d_rep, 1)
        self.g_S = nn.Linear(d_rep + 1, 1)

    def forward(self, x, F=None):
        hD, hV = self.phi_D(x), self.phi_V(x)
        o = {"logit_D": self.g_D(hD).squeeze(-1), "hD": hD, "hV": hV,
             "s": self.s_min + (1 - self.s_min) * torch.sigmoid(self.g_s(hV).squeeze(-1)),
             "logit_F": self.g_F(hV).squeeze(-1)}
        if F is not None:
            o["logit_S"] = self.g_S(torch.cat([hV, F[:, None]], 1)).squeeze(-1)
        return o


def cross_cov(a, b):
    a = (a - a.mean(0)) / (a.std(0) + 1e-4)
    b = (b - b.mean(0)) / (b.std(0) + 1e-4)
    c = a.T @ b / len(a)
    return (c ** 2).mean()


def interval_loss(logp, logL, logU):
    return (Fn.relu(logL - logp) ** 2 + Fn.relu(logp - logU) ** 2).mean()


def rob_loss(p, L, U):
    p = p.clamp(EPS, 1 - EPS)
    ce = lambda t: -(t * torch.log(p) + (1 - t) * torch.log(1 - p))
    return torch.maximum(ce(L), ce(U)).mean()


def batch_loss(method, model, b, w_cfg):
    x = b["x"]
    if method in BASELINES:
        lg = model(x)["logit_D"]
        if method == "erm_y":
            return Fn.binary_cross_entropy_with_logits(lg, b["Y"])
        if method == "mi_pseudo":
            t = torch.where(b["S"] > 0, b["Y"], b["m"])
            return Fn.binary_cross_entropy_with_logits(lg, t)
        m = b["F"] > 0 if method == "erm_flag" else b["S"] > 0
        if m.sum() == 0:
            return lg.sum() * 0
        w = (1.0 / b["sigma"][m]) if method == "ipw_verified" else torch.ones_like(lg[m])
        return (Fn.binary_cross_entropy_with_logits(lg[m], b["Y"][m], reduction="none") * w).sum() / w.sum()
    o = model(x, b["F"])
    pD = torch.sigmoid(o["logit_D"]).clamp(EPS, 1 - EPS)
    s = b["sigma"].clamp(EPS, 1.0) if method == "svcl_mar" else o["s"]
    pY = (pD * s).clamp(EPS, 1 - EPS)
    L = Fn.binary_cross_entropy(pY, b["Y"])
    if w_cfg["stage"]:
        L = L + w_cfg["stage"] * Fn.binary_cross_entropy_with_logits(o["logit_F"], b["F"])
        mf = b["F"] > 0
        if mf.any():
            L = L + w_cfg["stage"] * Fn.binary_cross_entropy_with_logits(o["logit_S"][mf], b["S"][mf])
    if w_cfg["sep"]:
        L = L + w_cfg["sep"] * cross_cov(o["hD"], o["hV"])
    if w_cfg["id"]:
        L = L + w_cfg["id"] * interval_loss(torch.log(pD), torch.log(b["L"]), torch.log(b["U"]))
    if w_cfg["rob"]:
        L = L + w_cfg["rob"] * rob_loss(pD, b["L"], b["U"])
    return L


def to_t(arrs, idx, dev):
    return {k: torch.as_tensor(v[idx], dtype=torch.float32, device=dev) for k, v in arrs.items()}


def train(method, arrs, tr, va, s_min=0.5, hid=128, epochs=30, bs=4096, lr=1e-3, wd=1e-4, seed=0,
          patience=5, dev=None, verbose=False):
    """arrs: dict các mảng numpy độ dài n: x [n,d], Y, F, S, sigma, L, U, m. tr/va: chỉ số.
    Dừng sớm theo chính hàm mục tiêu của phương pháp trên tập val."""
    dev = dev or device()
    torch.manual_seed(seed); np.random.seed(seed)
    d = arrs["x"].shape[1]
    model = (Plain(d, hid) if method in BASELINES else SVCL(d, hid, s_min=s_min)).to(dev)
    w_cfg = SVCL_VARIANTS.get(method)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    # tập con huấn luyện thực sự của baseline chỉ-đã-xác-minh nhỏ: dùng batch nhỏ hơn cho đủ bước
    sub = {"erm_flag": arrs["F"] > 0, "erm_verified": arrs["S"] > 0, "ipw_verified": arrs["S"] > 0}.get(method)
    tr_eff = tr[sub[tr]] if sub is not None else tr
    va_eff = va[sub[va]] if sub is not None else va
    bs_eff = min(bs, max(64, len(tr_eff) // 20))
    best, best_state, bad = np.inf, None, 0
    rng = np.random.default_rng(seed)
    Xva = to_t(arrs, va_eff, dev)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(tr_eff)
        for i in range(0, len(perm), bs_eff):
            b = to_t(arrs, perm[i:i + bs_eff], dev)
            loss = batch_loss(method, model, b, w_cfg)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(batch_loss(method, model, Xva, w_cfg))
        if verbose:
            V.log(f"    {method} ep{ep} val={vl:.5f}")
        if vl < best - 1e-6:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict(model, x, dev=None, bs=65536):
    dev = dev or next(model.parameters()).device
    out = {"pD": [], "s": []}
    for i in range(0, len(x), bs):
        xb = torch.as_tensor(x[i:i + bs], dtype=torch.float32, device=dev)
        o = model(xb)
        out["pD"].append(torch.sigmoid(o["logit_D"]).cpu().numpy())
        if "s" in o:
            out["s"].append(o["s"].cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items() if v}


def cav_sensitivity(model, x, concept, dev=None, n=20000, seed=0):
    """Đạo hàm theo hướng concept (CAV): w = hồi quy ridge của concept chuẩn hoá lên x;
    độ nhạy = E[∇_x logit_D · w/‖w‖]. Dùng cho đặc trưng embedding, nơi không can thiệp trực tiếp được."""
    from sklearn.linear_model import Ridge
    ok = np.isfinite(concept)
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(ok)
    idx = rng.choice(idx, min(n, len(idx)), replace=False)
    c = (concept[idx] - concept[idx].mean()) / (concept[idx].std() + 1e-9)
    w = Ridge(alpha=10.0).fit(x[idx], c).coef_
    w = w / (np.linalg.norm(w) + 1e-12)
    dev = dev or next(model.parameters()).device
    xb = torch.as_tensor(x[idx], dtype=torch.float32, device=dev).requires_grad_(True)
    lg = model(xb)["logit_D"].sum()
    g, = torch.autograd.grad(lg, xb)
    return float((g.cpu().numpy() @ w).mean())
