#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VL3 — Thí nghiệm chính trên ISIC-2024 (proposal_v2 §10, §11, H1–H4).

Cùng một kiến trúc head, huấn luyện dưới mọi phương pháp của vl_models (M0 erm_y, M1 erm_flag,
M2 erm_verified, IPW, MI, SVCL và ablation) trên hai loại x:
  tabular   đặc trưng TBP + Z (47 chiều)
  image     embedding ResNet-50 ImageNet đóng băng (2048 chiều, vl0), chuẩn hoá
Chia theo bệnh nhân đóng băng (Result/vl_split.json). Khoảng định danh [L, U] (Định lý 3) được dựng từ
nuisance TRÊN CÙNG loại x: GBM của vl1 cho tabular, MLP cross-fit cho image, để không phương pháp nào
nhận thông tin mà baseline không có.

Đánh giá trên test:
  auroc_D_verified   AUROC của p̂_D với Y trên tổn thương đã xác minh (nhãn bệnh thật duy nhất có)
  auroc_Y            AUROC với nhãn ghi nhận trên toàn bộ test
  ece_Y              hiệu chuẩn của P̂(Y|x) mà mô hình ngụ ý (p̂ cho baseline, p̂_D·ŝ cho SVCL)
  coverage, WCR      độ phủ trong [L, U] và rủi ro xấu nhất trên ℋ, ở s_min = 0.5
  delta_assoc[E]     logit p̂ trung bình tertile trên − tertile dưới của concept TBP, trong E0/E1/E2
  delta_int          (tabular) can thiệp: đặt cột concept bằng giá trị Q3 so với Q1, giữ nguyên phần còn lại
  cav                (image) đạo hàm theo hướng concept
H2/H3 = so dấu delta giữa M0 và M2; §9 = độ ổn định của delta qua E0 → E1 → E2.

Out -> Result/vl3_learning.json (+ .partial sau mỗi lần huấn luyện)
Chạy:  python3 vl3_learning.py [--features tabular,image] [--seeds 3]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import sv_common as SV
import vl_common as V
import vl_models as M

S_MAIN = 0.5
S_EXTRA = [0.3, 0.9]


def load_features(df, kind):
    if kind == "tabular":
        X, _ = V.tabular_X(df)
        return X.astype(np.float32)
    z = np.load(os.path.join(V.EMB, "isic2024.npz"))
    assert (z["ids"] == df["isic_id"].to_numpy()).all(), "embedding order differs from metadata"
    E = z["emb"].astype(np.float32)
    return ((E - E.mean(0)) / (E.std(0) + 1e-6)).astype(np.float32)


def nuisance_for(kind, X, df, fold):
    nu = np.load(os.path.join(V.RES, "vl_nuisance.npz"), allow_pickle=True)
    if kind == "tabular":
        return nu["q"], nu["sigma"]
    path = os.path.join(V.RES, "vl_nuisance_image.npz")
    if os.path.exists(path):
        z = np.load(path); return z["q"], z["sigma"]
    from sklearn.model_selection import GroupKFold
    g = df[SV.GROUP].to_numpy()
    tv = np.flatnonzero(fold != "test"); te = np.flatnonzero(fold == "test")
    out = {}
    for name in ("Y", "S"):
        y = df[name].to_numpy().astype(np.float32)
        arrs = dict(x=X, Y=y, F=df.F.to_numpy(np.float32), S=df.S.to_numpy(np.float32))
        p = np.zeros(len(y))
        for k, (a, b) in enumerate(GroupKFold(5).split(tv, groups=g[tv])):
            tr_i, ho = tv[a], tv[b]
            rng = np.random.default_rng(k)
            va_i = tr_i[np.isin(g[tr_i], rng.choice(np.unique(g[tr_i]), 80, replace=False))]
            tr_i = np.setdiff1d(tr_i, va_i)
            m = M.train("erm_y", arrs, tr_i, va_i, hid=256, seed=k)
            p[ho] = M.predict(m, X[ho])["pD"]
        va_i = np.flatnonzero(fold == "val"); tr_i = np.flatnonzero(fold == "train")
        m = M.train("erm_y", arrs, tr_i, va_i, hid=256, seed=0)
        p[te] = M.predict(m, X[te])["pD"]
        out[name] = np.clip(p, 1e-6, 1 - 1e-6)
        V.log(f"image nuisance {name}: AUROC(test)={V.auroc(y[te], p[te]):.3f}")
    np.savez(path, q=out["Y"], sigma=out["S"])
    return out["Y"], out["S"]


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def evaluate(model, meth, kind, X, df, te, L, U, Tc):
    pr = M.predict(model, X[te])
    pD = pr["pD"]
    pY = pD * pr["s"] if "s" in pr else pD
    Y, S, F = df.Y.to_numpy()[te], df.S.to_numpy()[te], df.F.to_numpy()[te]
    r = {"auroc_D_verified": V.auroc(Y[S > 0], pD[S > 0]), "auroc_Y": V.auroc(Y, pD),
         "ece_Y": V.ece(Y, pY), "mean_pD": float(pD.mean()),
         "coverage": V.coverage(pD, L[te], U[te]), "worst_case_risk": V.worst_case_risk(pD, L[te], U[te]),
         "delta_assoc": {}, "delta_int": {}, "cav": {}}
    lg = logit(pD)
    env = {"E0": np.ones(len(te), bool), "E1": F > 0, "E2": S > 0}
    for cname, T in Tc.items():
        t = T[te]
        r["delta_assoc"][cname] = {e: float(lg[m & (t == 1)].mean() - lg[m & (t == 0)].mean())
                                   for e, m in env.items()}
        if kind == "tabular":
            j = Tc_cols[cname]
            col = X[:, j]
            lo_v, hi_v = np.nanquantile(col, 1 / 6), np.nanquantile(col, 5 / 6)   # tâm tertile dưới/trên
            xl, xh = X[te].copy(), X[te].copy()
            xl[:, j] = lo_v; xh[:, j] = hi_v
            r["delta_int"][cname] = float(np.mean(logit(M.predict(model, xh)["pD"]) - logit(M.predict(model, xl)["pD"])))
        else:
            conc = df[V.CONCEPTS[cname]].to_numpy(float)[te]
            r["cav"][cname] = M.cav_sensitivity(model, X[te], conc)
    return r


Tc_cols = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="tabular,image")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--methods", default=",".join(M.ALL_METHODS))
    a = ap.parse_args()
    df = SV.load()
    fold = V.make_split(df)
    tr, va, te = (np.flatnonzero(fold == k) for k in ("train", "val", "test"))
    Tc = {c: SV.tertile(df, col).fillna(-1).to_numpy() for c, col in V.CONCEPTS.items()}
    _, names = V.tabular_X(df.head(1000))
    for c, col in V.CONCEPTS.items():
        Tc_cols[c] = names.index(col)
    part = os.path.join(V.RES, "vl3_learning.json.partial")
    out = json.load(open(part)) if os.path.exists(part) else {"runs": []}
    done = {(r["features"], r["method"], r["s_min"], r["seed"]) for r in out["runs"]}
    for kind in a.features.split(","):
        X = load_features(df, kind)
        q, sig = nuisance_for(kind, X, df, fold)
        hid = 128 if kind == "tabular" else 256
        for s_min in [S_MAIN] + S_EXTRA:
            L, U = V.disease_interval(q, sig, s_min)
            arrs = dict(x=X, Y=df.Y.to_numpy(np.float32), F=df.F.to_numpy(np.float32),
                        S=df.S.to_numpy(np.float32), sigma=sig.astype(np.float32),
                        L=np.clip(L, 1e-7, 1).astype(np.float32), U=np.clip(U, 1e-7, 1).astype(np.float32),
                        m=np.zeros(len(df), np.float32))
            LU_main = V.disease_interval(q, sig, S_MAIN)
            methods = a.methods.split(",") if s_min == S_MAIN else ["svcl"]
            for seed in range(a.seeds):
                mv = None
                for meth in methods:
                    key = (kind, meth, s_min, seed)
                    if key in done:
                        continue
                    if meth in ("mi_pseudo", "erm_verified") and mv is None:
                        mv = M.train("erm_verified", arrs, tr, va, hid=hid, seed=seed)
                        arrs["m"] = M.predict(mv, X)["pD"].astype(np.float32)
                    model = mv if meth == "erm_verified" else M.train(meth, arrs, tr, va, s_min=s_min, hid=hid,
                                                                      seed=seed)
                    r = evaluate(model, meth, kind, X, df, te, LU_main[0], LU_main[1], Tc)
                    r.update({"features": kind, "method": meth, "s_min": s_min, "seed": seed})
                    out["runs"].append(r)
                    json.dump(out, open(part, "w"), default=SV._np_default)
                    key_c = "delta_int" if kind == "tabular" else "cav"
                    V.log(f"[{kind}|s={s_min}|seed{seed}] {meth:14s} AUROC_Dver={r['auroc_D_verified']:.3f} "
                          f"AUROC_Y={r['auroc_Y']:.3f} ECE_Y={r['ece_Y']:.5f} cover={r['coverage']:.2f} "
                          f"WCR={r['worst_case_risk']:.4f} {key_c}="
                          f"{ {k[:5]: round(v, 2) for k, v in r[key_c].items()} }")
    out["summary"] = summarise(out["runs"])
    V.save_json(out, "vl3_learning.json")


def summarise(runs):
    s = {}
    for kind in sorted({r["features"] for r in runs}):
        s[kind] = {}
        for meth in M.ALL_METHODS:
            rr = [r for r in runs if r["features"] == kind and r["method"] == meth and r["s_min"] == S_MAIN]
            if not rr:
                continue
            key_c = "delta_int" if kind == "tabular" else "cav"
            agg = {k: float(np.mean([r[k] for r in rr if r[k] is not None]))
                   for k in ("auroc_D_verified", "auroc_Y", "ece_Y", "coverage", "worst_case_risk")}
            agg["sensitivity"] = {c: float(np.mean([r[key_c][c] for r in rr])) for c in V.CONCEPTS}
            agg["delta_assoc"] = {c: {e: float(np.mean([r["delta_assoc"][c][e] for r in rr])) for e in ("E0", "E1", "E2")}
                                  for c in V.CONCEPTS}
            # §9: số concept mà dấu của delta_assoc đổi giữa E0 và E2
            agg["n_sign_flip_E0_E2"] = int(sum(np.sign(agg["delta_assoc"][c]["E0"]) != np.sign(agg["delta_assoc"][c]["E2"])
                                               for c in V.CONCEPTS))
            s[kind][meth] = agg
        if "erm_y" in s[kind] and "erm_verified" in s[kind]:
            a, b = s[kind]["erm_y"]["sensitivity"], s[kind]["erm_verified"]["sensitivity"]
            s[kind]["H2_sign_reversal_M0_vs_M2"] = {c: bool(np.sign(a[c]) != np.sign(b[c])) for c in V.CONCEPTS}
    return s


if __name__ == "__main__":
    main()
