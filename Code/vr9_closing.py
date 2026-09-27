#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR9 — Phân tích khép lại: khoảng tin cậy bootstrap theo BỆNH NHÂN cho đảo dấu học được,
và độ nhạy với định nghĩa concept.

Đại lượng. Với một learner m và concept k, tương phản học được trên CÙNG quần thể đánh giá (test, đồng đều):
    Δ_m,k = E[logit f_m(X) | concept cao] − E[logit f_m(X) | concept thấp].
Báo cáo Δ_M0, Δ_M2, Δ_M0 − Δ_M2 và tỉ lệ bản bootstrap có sign(Δ_M0) ≠ sign(Δ_M2). Bootstrap lấy lại BỆNH NHÂN
(2,000 lần), cùng một bộ trọng số cho mọi seed, rồi lấy trung bình Δ qua seed trong từng bản bootstrap; biến
thiên giữa seed được báo cáo riêng.

Bốn họ learner (M0 = ERM trên Y toàn bộ, M2 = chỉ tổn thương đã xác minh):
  ft            ResNet-50 fine-tune (vr4), tập probe đồng đều 30,000 tổn thương test
  lp            linear probe trên ResNet-50 đóng băng (vr4), cùng tập probe
  head_tabular  MLP trên đặc trưng TBP + Z (như vl3), toàn bộ test
  head_image    MLP trên embedding ResNet-50 đóng băng (như vl3), toàn bộ test
Hai họ head được huấn luyện lại ở đây bằng đúng code và seed của vl3 (vl_models.train) để lưu dự đoán
từng tổn thương, vì vl3 không lưu; Δ tertile của chúng phải khớp bảng vl3 (mốc tái lập).

Định nghĩa concept (§8.2), không thêm concept mới:
  tertile (CHÍNH)  tertile trên so với dưới, cắt trên toàn cohort
  quartile, quintile   nhóm cực trên so với cực dưới
  median           trên so với dưới trung vị
  slope            độ dốc OLS của logit theo concept chuẩn hoá
  within_slope     như slope sau khi trừ trung bình trong từng bệnh nhân (bỏ khác biệt giữa bệnh nhân)
  rank_slope       độ dốc theo hạng phân vị của concept (bất biến với biến đổi đơn điệu)

Out -> Result/vr9_closing.json, Result/closing_heads.npz
"""
from __future__ import annotations

import json
import os

import numpy as np

import vr_common as R

B = 2000
DEFS = ["tertile", "quartile", "quintile", "median", "slope", "within_slope", "rank_slope"]
FT = os.path.join(R.RES, "finetune")


def groups_for(x, how, cuts):
    if how in ("tertile", "quartile", "quintile", "median"):
        lo, hi = cuts[how]
        return np.where(x <= lo, 0, np.where(x >= hi, 1, -1)) if how != "median" else np.where(x <= lo, 0, 1)
    return None


def cohort_cuts(xall):
    ok = xall[np.isfinite(xall)]
    q = lambda a: float(np.quantile(ok, a))
    return {"tertile": (q(1 / 3), q(2 / 3)), "quartile": (q(1 / 4), q(3 / 4)), "quintile": (q(1 / 5), q(4 / 5)),
            "median": (q(0.5), q(0.5)), "mean": float(ok.mean()), "sd": float(ok.std()), "sorted": np.sort(ok)}


def patient_stats(lg, x, pid, how, cuts):
    """Thống kê đủ theo bệnh nhân để bootstrap nhanh. Trả (P × k ma trận, hàm tính Δ từ tổng có trọng số)."""
    ok = np.isfinite(x)
    lg, x, pid = lg[ok], x[ok], pid[ok]
    up, inv = np.unique(pid, return_inverse=True)
    agg = lambda v: np.bincount(inv, weights=v, minlength=len(up))
    if how in ("tertile", "quartile", "quintile", "median"):
        g = groups_for(x, how, cuts)
        M = np.column_stack([agg(lg * (g == 1)), agg((g == 1).astype(float)), agg(lg * (g == 0)), agg((g == 0).astype(float))])
        f = lambda s: s[..., 0] / s[..., 1] - s[..., 2] / s[..., 3]
        return up, M, f
    if how == "rank_slope":
        z = np.searchsorted(cuts["sorted"], x) / len(cuts["sorted"])
    else:
        z = (x - cuts["mean"]) / cuts["sd"]
    if how == "within_slope":
        n = agg(np.ones_like(z)); mz = agg(z) / n; ml = agg(lg) / n
        zc, lc = z - mz[inv], lg - ml[inv]
        M = np.column_stack([agg(zc * lc), agg(zc * zc)])
        f = lambda s: s[..., 0] / s[..., 1]
        return up, M, f
    M = np.column_stack([agg(np.ones_like(z)), agg(z), agg(lg), agg(z * lg), agg(z * z)])
    def f(s):
        n, sz, sl, szl, szz = (s[..., i] for i in range(5))
        return (szl - sz * sl / n) / (szz - sz * sz / n)
    return up, M, f


def delta_boot(lgs, x, pid, how, cuts, W, pat_index):
    """lgs: list logit theo seed. W: (B × P_all) trọng số bệnh nhân. Trả Δ điểm (TB seed) và bản bootstrap."""
    pts, bts = [], []
    for lg in lgs:
        up, M, f = patient_stats(lg, x, pid, how, cuts)
        cols = np.array([pat_index[p] for p in up])
        pts.append(float(f(M.sum(0))))
        bts.append(f(W[:, cols] @ M))
    return float(np.mean(pts)), np.mean(bts, 0), pts


def summarise(d0, d2):
    p0, b0, s0 = d0; p2, b2, s2 = d2
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return {"M0": p0, "M0_ci": ci(b0), "M0_seeds": s0, "M2": p2, "M2_ci": ci(b2), "M2_seeds": s2,
            "diff": p0 - p2, "diff_ci": ci(b0 - b2), "p_sign_differ": float(np.mean(np.sign(b0) != np.sign(b2))),
            "point_sign_differ": bool(np.sign(p0) != np.sign(p2))}


def train_heads(df, fold):
    path = os.path.join(R.RES, "closing_heads.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    import vl_models as M
    tr, va, te = (np.flatnonzero(fold == k) for k in ("train", "val", "test"))
    out = {"test_idx": te}
    for kind in ("tabular", "image"):
        if kind == "tabular":
            X, _ = R.VL.tabular_X(df); X = X.astype(np.float32); hid = 128
        else:
            z = np.load(os.path.join(R.EMB, "isic2024.npz"))
            E = z["emb"].astype(np.float32); X = ((E - E.mean(0)) / (E.std(0) + 1e-6)).astype(np.float32); hid = 256
        arrs = dict(x=X, Y=df.Y.to_numpy(np.float32), F=df.F.to_numpy(np.float32), S=df.S.to_numpy(np.float32),
                    sigma=np.ones(len(df), np.float32), L=np.ones(len(df), np.float32), U=np.ones(len(df), np.float32),
                    m=np.zeros(len(df), np.float32))
        for seed in range(3):
            for meth, reg in (("erm_verified", "M2"), ("erm_y", "M0")):
                mdl = M.train(meth, arrs, tr, va, hid=hid, seed=seed)
                out[f"{kind}_{reg}_s{seed}"] = R.logit(M.predict(mdl, X[te])["pD"]).astype(np.float32)
                R.log(f"head {kind} {reg} seed {seed} trained")
    np.savez(path, **out)
    return out


def main():
    df, fold = R.load_isic()
    pid_all = df[R.SV.GROUP].to_numpy()
    pats = np.unique(pid_all); pat_index = {p: i for i, p in enumerate(pats)}
    rng = np.random.default_rng(R.SEED)
    # trọng số bootstrap theo bệnh nhân, riêng cho từng tập đánh giá (bệnh nhân test)
    heads = train_heads(df, fold)
    fams = {}
    for mode in ("ft", "lp"):
        z0 = [np.load(os.path.join(FT, f"{mode}_M0_s{s}.npz")) for s in range(3)]
        z2 = [np.load(os.path.join(FT, f"{mode}_M2_s{s}.npz")) for s in range(3)]
        probe, uni = z0[0]["probe"], z0[0]["is_uniform"]
        assert all((z["probe"] == probe).all() for z in z0 + z2)
        fams[mode] = (probe[uni], [z["logit"][uni] for z in z0], [z["logit"][uni] for z in z2])
    te = heads["test_idx"]
    for kind in ("tabular", "image"):
        fams[f"head_{kind}"] = (te, [heads[f"{kind}_M0_s{s}"] for s in range(3)], [heads[f"{kind}_M2_s{s}"] for s in range(3)])
    out = {"B": B, "definitions": DEFS, "families": {}}
    for fam, (idx, l0, l2) in fams.items():
        pid = pid_all[idx]
        tp = np.unique(pid)
        cnt = np.zeros((B, len(pats)))
        for b in range(B):
            smp = rng.choice(tp, len(tp), replace=True)
            np.add.at(cnt[b], [pat_index[p] for p in smp], 1.0)
        res = {"n_lesions": int(len(idx)), "n_patients": int(len(tp)), "concepts": {}}
        for c, col in R.CONCEPTS.items():
            xall = df[col].to_numpy(float); cuts = cohort_cuts(xall); x = xall[idx]
            res["concepts"][c] = {}
            for how in DEFS:
                d0 = delta_boot(l0, x, pid, how, cuts, cnt, pat_index)
                d2 = delta_boot(l2, x, pid, how, cuts, cnt, pat_index)
                res["concepts"][c][how] = summarise(d0, d2)
            t = res["concepts"][c]["tertile"]
            agree = [res["concepts"][c][h]["point_sign_differ"] for h in DEFS]
            res["concepts"][c]["n_definitions_reversed"] = int(sum(agree))
            R.log(f"[{fam}] {c:22s} Δ_M0={t['M0']:+.2f} {np.round(t['M0_ci'], 2)}  Δ_M2={t['M2']:+.2f} {np.round(t['M2_ci'], 2)}  "
                  f"diff={t['diff']:+.2f} {np.round(t['diff_ci'], 2)}  P(sign≠)={t['p_sign_differ']:.3f}  "
                  f"definitions reversed {sum(agree)}/{len(DEFS)}")
        out["families"][fam] = res
    R.save_json(out, "vr9_closing.json")


if __name__ == "__main__":
    main()
