#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR3 — Identification-Constrained Disease Learning (ICDL), proposal_v3 §4–8, §15–16.

ICDL bỏ L_stage và L_sep của SVCL (ablation của v2 cho thấy chúng không tạo khác biệt). Ba dạng mục tiêu:
  icdl_kl     MỤC TIÊU HỢP NHẤT (§4): ℓ_i = max{ KL(L_i ‖ f_i), KL(U_i ‖ f_i) }, tức regret log-loss xấu nhất
              trên khoảng định danh 𝓘_i = [L_i, U_i]. KL(p‖f) lồi theo p nên cực đại tại đầu mút; ℓ_i nhỏ nhất
              tại "tâm KL" của 𝓘_i và tăng nhanh khi f ra ngoài 𝓘_i. Một số hạng thay cho L_ID + L_rob.
  icdl_hinge  chỉ L_IC của §6: bình phương khoảng cách (thang log) tới 𝓘_i, bằng 0 bên trong
  icdl_hr     L_IC + L_rob cũ (CE xấu nhất tại đầu mút), để so với dạng hợp nhất
Baseline: erm_y (M0) và erm_verified (M2).

𝓘_i = [q̂(x_i), min(q̂/s, q̂ + 1 − σ̂)] (Định lý 3), q̂ và σ̂ cross-fit, không rò rỉ nhãn test.

Phần A — mô phỏng, ma trận s_true × s_assumed (§15). π¹(x) = s_true + (1 − s_true)·expit(w₁ᵀc) ≥ s_true, nên
  s_true là sàn đúng. Đo: sai số log so với p thật, độ phủ p thật trong 𝓘(s_assumed), AUROC_D, rủi ro xấu nhất,
  số concept sai dấu (can thiệp x_k: −1 → +1), dải dự đoán theo giả định [min_s f, max_s f] và độ phủ của nó,
  quyết định so với oracle ở ngưỡng τ.
Phần B — ℋ có cấu trúc (§16): cận của tương phản tertile RR_D dưới
  rect    s(x) tự do trong [s, 1] từng tổn thương            → [RR_Y·s, RR_Y/s] (Định lý 1)
  strata  s chung trong K = 5 tầng (ngũ phân vị σ̂)            → liệt kê 2^K đỉnh
  shared  một s chung cho mọi tổn thương (⇔ A = 1)             → RR_Y (định danh điểm)
  và rủi ro xấu nhất của mô hình dưới cả ba ℋ.
Phần C — ISIC-2024 (tabular và ảnh đóng băng): ICDL theo từng s ∈ 𝒮, dải theo giả định, tỉ lệ quyết định bất
  biến theo giả định, phân phối tipping point cá thể s_i* = q̂_i/τ (§8), cận ℋ có cấu trúc cho năm concept.

Tipping point cá thể (§8). Dưới giả định s_min = s, p(x) ∈ [q, q/s]. Quyết định "ác tính" (p ≥ τ) ổn định trên
mọi cơ chế thừa nhận ⇔ q ≥ τ, không phụ thuộc s; quyết định "lành" ổn định ⇔ q/s < τ ⇔ s > q/τ. Vậy
s_i* = q̂_i/τ là sàn xác minh nhỏ nhất để quyết định "lành" của tổn thương i không đổi.

Out -> Result/vr3_icdl.json
"""
from __future__ import annotations

import argparse
import copy
import itertools
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

import vr_common as R

TAUS = [0.005, 0.01, 0.02]
EPS = 1e-7
ORACLE = {}
BETA = np.array([0.5, 0.4, 0.3, 0.4, 0.4])
FW = np.array([1.2, 0.8, 1.5, -1.0, -1.0])
SW = np.array([0.0, 0.3, -0.6, 0.8, 0.9])
W1 = np.array([0.8, 0.5, 0.8, -0.5, -0.5])
torch.set_num_threads(32)


def mlp(d, hid):
    return nn.Sequential(nn.Linear(d, hid), nn.GELU(), nn.Dropout(0.1), nn.Linear(hid, hid), nn.GELU(),
                         nn.Dropout(0.1), nn.Linear(hid, 1))


def kl(p, f):
    return p * (torch.log(p + EPS) - torch.log(f)) + (1 - p) * (torch.log(1 - p + EPS) - torch.log(1 - f))


def loss_fn(method, lg, b):
    f = torch.sigmoid(lg).clamp(EPS, 1 - EPS)
    if method == "erm_y":
        return Fn.binary_cross_entropy_with_logits(lg, b["Y"])
    if method == "erm_verified":
        m = b["S"] > 0
        return Fn.binary_cross_entropy_with_logits(lg[m], b["Y"][m]) if m.any() else lg.sum() * 0
    L, U = b["L"], b["U"]
    if method == "icdl_kl":
        return torch.maximum(kl(L, f), kl(U, f)).mean()
    lf = torch.log(f)
    hinge = (Fn.relu(torch.log(L) - lf) ** 2 + Fn.relu(lf - torch.log(U)) ** 2).mean()
    if method == "icdl_hinge":
        return hinge
    ce = lambda t: -(t * torch.log(f) + (1 - t) * torch.log(1 - f))
    return hinge + torch.maximum(ce(L), ce(U)).mean()          # icdl_hr


def train(method, arr, tr, va, hid=128, epochs=40, bs=4096, lr=1e-3, seed=0, patience=4):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if method == "erm_verified":
        tr, va = tr[arr["S"][tr] > 0], va[arr["S"][va] > 0]
        bs = min(bs, max(64, len(tr) // 20))
    T = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in arr.items()}
    net = mlp(arr["x"].shape[1], hid)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    vb = {k: v[va] for k, v in T.items()}
    best, state, bad = np.inf, None, 0
    for _ in range(epochs):
        net.train()
        perm = rng.permutation(tr)
        for i in range(0, len(perm), bs):
            b = {k: v[perm[i:i + bs]] for k, v in T.items()}
            loss = loss_fn(method, net(b["x"]).squeeze(-1), b)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(loss_fn(method, net(vb["x"]).squeeze(-1), vb))
        if vl < best - 1e-7:
            best, state, bad = vl, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state); net.eval()
    return net


@torch.no_grad()
def predict(net, x):
    return torch.sigmoid(net(torch.as_tensor(x, dtype=torch.float32)).squeeze(-1)).numpy()


def nuisance(x, Y, S, tv, te, groups):
    from sklearn.ensemble import HistGradientBoostingClassifier
    out = []
    for y in (Y, S):
        p = np.zeros(len(y))
        p[tv] = R.VL.crossfit_prob(x[tv], y[tv], groups[tv])
        m = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.05,
                                           random_state=R.SEED).fit(x[tv], y[tv])
        p[te] = m.predict_proba(x[te])[:, 1]
        out.append(np.clip(p, 1e-6, 1 - 1e-6))
    return out


def wcr_structured(f, q, sig, s, strata):
    """Rủi ro log-loss xấu nhất dưới ba ℋ. Với p = q/s', CE tuyến tính theo 1/s' nên cực trị ở s' ∈ {s, 1}."""
    f = np.clip(f, EPS, 1 - EPS)
    def ce(p):
        return -(p * np.log(f) + (1 - p) * np.log(1 - f))
    L, U = R.disease_interval(q, sig, s)
    rect = float(np.mean(np.maximum(ce(L), ce(U))))
    cL, cU = ce(L), ce(U)
    strat = float(sum(max(cL[strata == k].sum(), cU[strata == k].sum()) for k in np.unique(strata)) / len(f))
    shared = float(max(cL.mean(), cU.mean()))
    return {"rect": rect, "strata": strat, "shared": shared}


def contrast_bounds(q, T, strata, s):
    """Cận của RR_D giữa tertile 1 và 0 dưới ba ℋ (p = q/s_k, s_k ∈ [s, 1])."""
    K = np.unique(strata)
    m1 = np.array([q[(T == 1) & (strata == k)].sum() for k in K]) / max((T == 1).sum(), 1)
    m0 = np.array([q[(T == 0) & (strata == k)].sum() for k in K]) / max((T == 0).sum(), 1)
    rr_y = m1.sum() / m0.sum()
    vals = []
    for corner in itertools.product([s, 1.0], repeat=len(K)):
        inv = 1 / np.array(corner)
        vals.append((m1 * inv).sum() / (m0 * inv).sum())
    return {"rect": [rr_y * s, rr_y / s], "strata": [min(vals), max(vals)], "shared": [rr_y, rr_y]}


def strata_of(sig, K=5):
    return np.searchsorted(np.quantile(sig, np.linspace(0, 1, K + 1)[1:-1]), sig)


# ----------------------------------------------------------------------------------
# A + B: mô phỏng
# ----------------------------------------------------------------------------------
def expit(z):
    return 1 / (1 + np.exp(-z))


def simulate(s_true, n, rng, prev=0.005):
    from scipy.optimize import brentq
    x = rng.standard_normal((n, 8)).astype(np.float32); c = x[:, :5]
    a = brentq(lambda a: expit(a + c @ BETA).mean() - prev, -20, 5)
    p = expit(a + c @ BETA)
    D = rng.random(n) < p
    f0 = brentq(lambda f: expit(f + c @ FW).mean() - 0.05, -30, 10)
    s0 = brentq(lambda s: (expit(f0 + c @ FW) * expit(s + c @ SW)).mean() - 0.003, -30, 10)
    pi0 = expit(f0 + c @ FW) * expit(s0 + c @ SW)
    pi1 = s_true + (1 - s_true) * expit(c @ W1)
    S = rng.random(n) < np.where(D, pi1, pi0)
    Y = D & S
    global ORACLE
    ORACLE = {"q": p * pi1, "sigma": p * pi1 + (1 - p) * pi0}     # nuisance thật, để tách lỗi ước lượng khỏi T3
    return x, p, D.astype(np.float32), S.astype(np.float32), Y.astype(np.float32)


def concept_signs(net, x):
    out = []
    for k in range(5):
        lo, hi = x.copy(), x.copy(); lo[:, k] = -1; hi[:, k] = 1
        out.append(float(np.mean(R.logit(predict(net, hi)) - R.logit(predict(net, lo)))))
    return out


def part_ab(n, rng):
    rows, bands, struct = [], [], []
    for s_true in (0.5, 0.6, 0.7, 0.8, 0.9):
        x, p, D, S, Y = simulate(s_true, n, rng)
        idx = rng.permutation(n); tr, va, te = idx[:int(.6 * n)], idx[int(.6 * n):int(.8 * n)], idx[int(.8 * n):]
        tv = np.concatenate([tr, va])
        q, sig = nuisance(x, Y, S, tv, te, np.arange(n) // 200)
        base = {}
        for meth in ("erm_y", "erm_verified"):
            net = train(meth, dict(x=x, Y=Y, S=S, L=q, U=q), tr, va)
            pr = predict(net, x[te])
            base[meth] = pr
            rows.append({"s_true": s_true, "s_assumed": None, "method": meth, **evaluate(net, pr, x, te, p, D, q, sig, None)})
        preds = {}
        for s_a in R.S_GRID:
            L, U = R.disease_interval(q, sig, s_a)
            arr = dict(x=x, Y=Y, S=S, L=np.clip(L, EPS, 1).astype(np.float32), U=np.clip(U, EPS, 1).astype(np.float32))
            methods = ["icdl_kl"] + (["icdl_hinge", "icdl_hr"] if s_a in (0.5, round(s_true, 1)) else [])
            for meth in methods:
                net = train(meth, arr, tr, va)
                pr = predict(net, x[te])
                if meth == "icdl_kl":
                    preds[s_a] = pr
                r = {"s_true": s_true, "s_assumed": s_a, "method": meth, **evaluate(net, pr, x, te, p, D, q, sig, s_a)}
                rows.append(r)
                R.log(f"[sim s_true={s_true} s_a={s_a}] {meth:10s} |log err|={r['abs_log_err']:.3f} "
                      f"cov_true_p={r['true_p_in_interval']:.3f} (oracle {r['true_p_in_oracle_interval']:.3f}) AUROC_D={r['auroc_D']:.3f} WCR={r['wcr_rect']:.4f} "
                      f"sign_wrong={r['n_sign_wrong']}")
        P = np.column_stack([preds[s] for s in R.S_GRID])
        lo, hi = P.min(1), P.max(1)
        band = {"s_true": s_true, "band_covers_true_p": float(np.mean((p[te] >= lo) & (p[te] <= hi))),
                "median_band_ratio": float(np.median(hi / lo)),
                "decisions": {}}
        for tau in TAUS:
            dec = P >= tau
            inv = dec.all(1) | (~dec).all(1)
            oracle = p[te] >= tau
            band["decisions"][str(tau)] = {
                "frac_assumption_free": float(inv.mean()),
                "assumption_free_agree_oracle": float(np.mean(dec[inv, 0] == oracle[inv])) if inv.any() else None,
                "error_at_true_s": float(np.mean((preds[min(R.S_GRID, key=lambda s: abs(s - s_true))] >= tau) != oracle)),
                "error_erm_y": float(np.mean((base["erm_y"] >= tau) != oracle)),
                "error_erm_verified": float(np.mean((base["erm_verified"] >= tau) != oracle)),
                "oracle_pos_rate": float(oracle.mean())}
        bands.append(band)
        # ℋ có cấu trúc: cận tương phản, độ phủ RR_D thật
        st = strata_of(sig)
        for k, name in enumerate(R.CONCEPTS):
            T = np.where(x[:, k] <= np.quantile(x[:, k], 1 / 3), 0, np.where(x[:, k] >= np.quantile(x[:, k], 2 / 3), 1, -1))
            rr_true = p[T == 1].mean() / p[T == 0].mean()
            for s_a in (0.5, s_true):
                cb = contrast_bounds(q, T, st, s_a)
                struct.append({"s_true": s_true, "s_assumed": s_a, "concept": name, "RR_D_true": rr_true,
                               **{f"{h}_bounds": v for h, v in cb.items()},
                               **{f"{h}_covers": bool(v[0] * 0.999 <= rr_true <= v[1] * 1.001) for h, v in cb.items()}})
        # rủi ro xấu nhất dưới ba ℋ cho ICDL (s_a = 0.5) và ERM-Y
        for name, pr in (("icdl_kl", preds[0.5]), ("erm_y", base["erm_y"])):
            struct.append({"s_true": s_true, "s_assumed": 0.5, "model": name,
                           "wcr": wcr_structured(pr, q[te], sig[te], 0.5, st[te])})
        R.log(f"[sim s_true={s_true}] band covers true p {band['band_covers_true_p']:.3f}, "
              f"median ratio {band['median_band_ratio']:.2f}")
    return rows, bands, struct


def evaluate(net, pr, x, te, p, D, q, sig, s_a):
    s_eval = 0.5 if s_a is None else s_a
    L, U = R.disease_interval(q[te], sig[te], s_eval)
    deltas = concept_signs(net, x[te][:20000])
    Lo, Uo = R.disease_interval(ORACLE["q"][te], ORACLE["sigma"][te], s_eval)
    return {"abs_log_err": float(np.mean(np.abs(np.log(np.clip(pr, EPS, 1)) - np.log(p[te])))),
            "true_p_in_interval": float(np.mean((p[te] >= L) & (p[te] <= U))),
            "true_p_in_oracle_interval": float(np.mean((p[te] >= Lo * (1 - 1e-9)) & (p[te] <= Uo * (1 + 1e-9)))),
            "pred_in_oracle_interval": R.coverage(pr, Lo, Uo),
            "pred_in_interval": R.coverage(pr, L, U), "auroc_D": R.auroc(D[te], pr),
            "wcr_rect": R.worst_case_risk(pr, L, U), "concept_delta": deltas,
            "n_sign_wrong": int(sum(d <= 0 for d in deltas))}


# ----------------------------------------------------------------------------------
# C: ISIC-2024
# ----------------------------------------------------------------------------------
def part_c(features):
    df, fold = R.load_isic()
    tr, va, te = (np.flatnonzero(fold == k) for k in ("train", "val", "test"))
    out = {}
    for kind in features:
        if kind == "tabular":
            x, _ = R.VL.tabular_X(df)
            nz = np.load(os.path.join(R.MISS_RES, "vl_nuisance.npz"))
            hid = 128
        else:
            z = np.load(os.path.join(R.EMB, "isic2024.npz"))
            E = z["emb"].astype(np.float32); x = (E - E.mean(0)) / (E.std(0) + 1e-6)
            nz = np.load(os.path.join(R.MISS_RES, "vl_nuisance_image.npz"))
            hid = 256
        q, sig = nz["q"], nz["sigma"]
        Y, S = df.Y.to_numpy(np.float32), df.S.to_numpy(np.float32)
        preds, res = {}, {"per_s": {}}
        for s in R.S_GRID:
            L, U = R.disease_interval(q, sig, s)
            arr = dict(x=x, Y=Y, S=S, L=np.clip(L, EPS, 1).astype(np.float32), U=np.clip(U, EPS, 1).astype(np.float32))
            net = train("icdl_kl", arr, tr, va, hid=hid)
            pr = predict(net, x[te]); preds[s] = pr
            Tc = {c: R.SV.tertile(df, col).fillna(-1).to_numpy()[te] for c, col in R.CONCEPTS.items()}
            lg = R.logit(pr)
            res["per_s"][str(s)] = {
                "auroc_Y": R.auroc(Y[te], pr), "auroc_D_verified": R.auroc(Y[te][S[te] > 0], pr[S[te] > 0]),
                "pred_in_interval": R.coverage(pr, L[te], U[te]), "wcr_rect": R.worst_case_risk(pr, L[te], U[te]),
                "mean_pred": float(pr.mean()),
                "learned_contrast": {c: float(lg[t == 1].mean() - lg[t == 0].mean()) for c, t in Tc.items()}}
            R.log(f"[ISIC {kind} s={s}] AUROC_Y={res['per_s'][str(s)]['auroc_Y']:.3f} "
                  f"in-set={res['per_s'][str(s)]['pred_in_interval']:.3f} mean p̂={pr.mean():.5f}")
        P = np.column_stack([preds[s] for s in R.S_GRID])
        res["band"] = {"median_ratio": float(np.median(P.max(1) / P.min(1))),
                       "monotone_decreasing_in_s": float(np.mean(np.all(np.diff(P, axis=1) <= 1e-9, axis=1)))}
        res["decisions"] = {}
        qt = q[te]
        for tau in TAUS:
            dec = P >= tau
            inv = dec.all(1) | (~dec).all(1)
            s_star = np.where(qt < tau, qt / tau, 0.0)
            res["decisions"][str(tau)] = {
                "frac_assumption_free_model": float(inv.mean()),
                "frac_positive_at_s0.5": float(dec[:, R.S_GRID.index(0.5)].mean()),
                "s_star_quantiles": {str(k): float(np.quantile(s_star, k)) for k in (0.5, 0.9, 0.99, 0.999)},
                "frac_s_star_above": {str(s): float(np.mean(s_star > s)) for s in (0.3, 0.5, 0.7, 0.9)},
                "frac_robust_malignant": float(np.mean(qt >= tau))}
        st = strata_of(sig)
        res["structured_contrast_s0.5"] = {}
        for c, col in R.CONCEPTS.items():
            T = R.SV.tertile(df, col).fillna(-1).to_numpy()
            res["structured_contrast_s0.5"][c] = contrast_bounds(q, T, st, 0.5)
        res["wcr_structured_s0.5"] = wcr_structured(preds[0.5], q[te], sig[te], 0.5, st[te])
        out[kind] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300_000)
    ap.add_argument("--features", default="tabular,image")
    ap.add_argument("--skip-sim", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(R.SEED)
    out = {}
    if not a.skip_sim:
        rows, bands, struct = part_ab(a.n, rng)
        out.update({"sim_rows": rows, "sim_bands": bands, "sim_structured": struct})
        R.save_json(out, "vr3_icdl.json")
    out["isic"] = part_c(a.features.split(","))
    R.save_json(out, "vr3_icdl.json")


if __name__ == "__main__":
    main()
