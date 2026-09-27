#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR19 — Suy luận chính: bootstrap đồng thời train + test, 1,000 lần, cho hai họ learner chính (tabular, ảnh đóng băng).

Mỗi lần lặp: lấy lại bệnh nhân train (có hoàn lại), huấn luyện lại M0 và M2 đúng như phân tích chính, hiệu chỉnh
Platt từng learner trên tập val cố định của regime (M0: val toàn bộ; M2: val đã xác minh), lấy lại bệnh nhân test.
Ghi lại cho năm concept:
  (1) Δ_M0, Δ_M2 thô (estimand chính) và đã hiệu chỉnh;
  (2) tập định danh của mục tiêu bệnh trên CÙNG functional và CÙNG quần thể với Δ (Proposition 2):
        ψ_k = E[logit p(X̃) | t=1] − E[logit p(X̃) | t=0],  p(x̃) = P(D=1 | x̃) = q(x̃)/s(x̃),  s(x̃) ≥ s_min,
      ψ_L(s) = E₁[logit q] − E₀[logit U_s],  ψ_U(s) = E₁[logit U_s] − E₀[logit q],  U_s = min(q/s, q + 1 − σ),
      với q̂ = xác suất M0 đã hiệu chỉnh (ước lượng P(Y=1 | x̃)), σ̂ = nuisance tabular out-of-fold;
      tipping point s* = sàn nhỏ nhất để dấu của ψ_k được định danh;
  (3) Δ thô trên quần thể test bị giới hạn vào vùng có hỗ trợ: σ̂ ≥ phân vị 1%, 5%, 10% của σ̂ trên tổn thương
      train đã xác minh (phân tích độ nhạy overlap, không phải chứng minh positivity).
Một lần chạy "gốc" (không lấy lại) cho ước lượng điểm của các đại lượng đã hiệu chỉnh và của ψ.

Chạy song song:  python3 vr19_primary_bootstrap.py --family tabular --start 0 --end 500
                 ... rồi  python3 vr19_primary_bootstrap.py --summarize
Out -> Result/vr19/{family}_{start}_{end}.json ; Result/vr19_primary_bootstrap.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

import vr_common as R
import gpu_setup  # noqa: F401  (ép GPU, giới hạn luồng)

OUTD = os.path.join(R.RES, "vr19")
S_FINE = np.round(np.arange(0.01, 1.0001, 0.01), 2)
S_REPORT = [0.3, 0.5, 0.7, 0.9]
SUPPORT_Q = [0.01, 0.05, 0.10]
PRIMARY = ["color_variegation", "size", "lesion_skin_contrast"]


def features(df, kind):
    if kind == "tabular":
        X, _ = R.VL.tabular_X(df)
        return X.astype(np.float32), 128
    z = np.load(os.path.join(R.EMB, "isic2024.npz"))
    E = z["emb"].astype(np.float32)
    return ((E - E.mean(0)) / (E.std(0) + 1e-6)).astype(np.float32), 256


def arrays(df, X):
    n = len(df)
    S = df.S.to_numpy(np.float32)
    return dict(x=X, Y=df.Y.to_numpy(np.float32), F=S, S=S, sigma=np.ones(n, np.float32), L=np.ones(n, np.float32),
                U=np.ones(n, np.float32), m=np.zeros(n, np.float32))


def platt_fit(lg, y):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1e6, max_iter=2000).fit(lg[:, None], y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def psi_bounds(q, sig, T):
    """ψ_L(s), ψ_U(s) trên lưới S_FINE; q, sig trên quần thể đánh giá."""
    lq = R.logit(q)
    out_L, out_U = [], []
    m1, m0 = T == 1, T == 0
    for s in S_FINE:
        U = np.minimum(q / s, q + 1 - sig)
        lU = R.logit(np.clip(U, 1e-7, 1 - 1e-7))
        out_L.append(lq[m1].mean() - lU[m0].mean())
        out_U.append(lU[m1].mean() - lq[m0].mean())
    return np.array(out_L), np.array(out_U)


def tipping(psiL, psiU):
    """Sàn nhỏ nhất để dấu ψ được định danh (dương nếu ψ_L > 0, âm nếu ψ_U < 0); None nếu không có trên [0.01, 1]."""
    pos = np.flatnonzero(psiL > 0); neg = np.flatnonzero(psiU < 0)
    return (float(S_FINE[pos[0]]) if len(pos) else None, float(S_FINE[neg[0]]) if len(neg) else None)


def one_rep(M, df, X, hid, A, tr, va, va2, te, seed, sig, Tfull, sup_masks):
    """Huấn luyện đúng như phân tích chính (dừng sớm trên toàn bộ val; M2 dùng phần đã xác minh của val),
    rồi hiệu chỉnh Platt: M0 trên toàn bộ val, M2 trên val đã xác minh."""
    Y = df.Y.to_numpy()
    m2 = M.train("erm_verified", A, tr, va, hid=hid, seed=seed)
    m0 = M.train("erm_y", A, tr, va, hid=hid, seed=seed)
    a0, b0 = platt_fit(R.logit(M.predict(m0, X[va])["pD"]), Y[va])
    a2, b2 = platt_fit(R.logit(M.predict(m2, X[va2])["pD"]), Y[va2])
    ute, inv = np.unique(te, return_inverse=True)
    l0 = R.logit(M.predict(m0, X[ute])["pD"])[inv]; l2 = R.logit(M.predict(m2, X[ute])["pD"])[inv]
    c0, c2 = a0 * l0 + b0, a2 * l2 + b2
    q = 1 / (1 + np.exp(-c0)); sg = sig[te]
    out = {"platt": [a0, a2], "concepts": {}}
    for c, Tf in Tfull.items():
        T = Tf[te]
        D = lambda v, m=np.ones(len(te), bool): float(v[m & (T == 1)].mean() - v[m & (T == 0)].mean())
        pL, pU = psi_bounds(q, sg, T)
        sp, sn = tipping(pL, pU)
        r = {"d0": D(l0), "d2": D(l2), "d0c": D(c0), "d2c": D(c2), "s_pos": sp, "s_neg": sn,
             "psiL": {str(s): float(pL[S_FINE.tolist().index(s)]) for s in S_REPORT},
             "psiU": {str(s): float(pU[S_FINE.tolist().index(s)]) for s in S_REPORT},
             "support": {}}
        for qn, msk in sup_masks.items():
            m = msk[te]
            r["support"][qn] = [D(l0, m), D(l2, m)]
        out["concepts"][c] = r
    return out


def run(family, start, end):
    import vl_models as M
    df, fold = R.load_isic()
    X, hid = features(df, family)
    A = arrays(df, X)
    Y, S = df.Y.to_numpy(), df.S.to_numpy()
    tr_all, va, te_all = (np.flatnonzero(fold == k) for k in ("train", "val", "test"))
    va2 = va[S[va] == 1]
    nz = np.load(os.path.join(R.MISS_RES, "vl_nuisance.npz"))
    sig = nz["sigma"]
    thr = {str(qq): float(np.quantile(sig[tr_all[S[tr_all] == 1]], qq)) for qq in SUPPORT_Q}
    sup_masks = {k: sig >= v for k, v in thr.items()}
    Tfull = {c: R.SV.tertile(df, col).fillna(-1).to_numpy() for c, col in R.CONCEPTS.items()}
    pid = df[R.SV.GROUP].to_numpy()
    trp, tep = np.unique(pid[tr_all]), np.unique(pid[te_all])
    tr_by = {p: tr_all[pid[tr_all] == p] for p in trp}; te_by = {p: te_all[pid[te_all] == p] for p in tep}
    os.makedirs(OUTD, exist_ok=True)
    path = os.path.join(OUTD, f"{family}_{start}_{end}.json")
    res = json.load(open(path)) if os.path.exists(path) else {"family": family, "support_thresholds": thr, "reps": {}}
    for b in range(start, end):
        if str(b) in res["reps"]:
            continue
        if b < 0:                                               # lần chạy gốc: b = −1, −2, −3 là seed 0, 1, 2
            tr, te = tr_all, te_all
        else:
            rng = np.random.default_rng(50_000 + b)
            tr = np.concatenate([tr_by[p] for p in rng.choice(trp, len(trp))])
            te = np.concatenate([te_by[p] for p in rng.choice(tep, len(tep))])
        res["reps"][str(b)] = one_rep(M, df, X, hid, A, tr, va, va2, te, (-b - 1) if b < 0 else b, sig, Tfull, sup_masks)
        json.dump(res, open(path + ".tmp", "w")); os.replace(path + ".tmp", path)
        if b % 25 == 0:
            c = res["reps"][str(b)]["concepts"]["color_variegation"]
            R.log(f"{family} rep {b}: color d0 {c['d0']:+.2f} d2 {c['d2']:+.2f} s* {c['s_pos']}")


def average_originals(origs):
    """Ước lượng điểm = trung bình qua ba lần chạy gốc (seed 0–2), như Table 1; s* tính lại trên ψ trung bình."""
    avg = {"platt": np.mean([o["platt"] for o in origs], 0).tolist(), "concepts": {}}
    for c in origs[0]["concepts"]:
        cs = [o["concepts"][c] for o in origs]
        e = {k: float(np.mean([x[k] for x in cs])) for k in ("d0", "d2", "d0c", "d2c")}
        e["psiL"] = {s: float(np.mean([x["psiL"][s] for x in cs])) for s in cs[0]["psiL"]}
        e["psiU"] = {s: float(np.mean([x["psiU"][s] for x in cs])) for s in cs[0]["psiU"]}
        sp = [x["s_pos"] for x in cs]; sn = [x["s_neg"] for x in cs]
        e["s_pos"] = None if any(v is None for v in sp) else float(np.max(sp))
        e["s_neg"] = None if any(v is None for v in sn) else float(np.max(sn))
        e["support"] = {q: np.mean([x["support"][q] for x in cs], 0).tolist() for q in cs[0]["support"]}
        avg["concepts"][c] = e
    avg["n_seeds"] = len(origs)
    return avg


def q95_floor(sp):
    """Phân vị 95 của s*, coi lần lặp không có s* (không sàn nào ≤ 1 làm ψ_L dương) là trên 1; phân vị không nội suy.
    Vì ψ_L tăng theo s, đại lượng này bằng sàn nhỏ nhất mà ψ_L dương trong ít nhất 95% lần lặp."""
    v = np.where(np.isfinite(np.asarray(sp, float)), np.asarray(sp, float), 9.0)
    x = float(np.quantile(v, 0.95, method="inverted_cdf"))
    return None if x > 1 else x


def summarize():
    out = {"B": {}, "families": {}}
    for fam in ("tabular", "image"):
        reps, origs, thr = [], [], None
        for f in glob.glob(os.path.join(OUTD, f"{fam}_*.json")):
            d = json.load(open(f)); thr = d["support_thresholds"]
            for k, v in d["reps"].items():
                (origs if int(k) < 0 else reps).append(v)
        orig = average_originals(origs) if origs else None
        B = len(reps); out["B"][fam] = B
        alpha_b = 0.05 / 6
        fo = {"support_thresholds": thr, "platt_original": orig["platt"] if orig else None, "concepts": {}}
        for c in R.CONCEPTS:
            g = lambda key: np.array([r["concepts"][c][key] for r in reps], float)
            d0, d2, d0c, d2c = g("d0"), g("d2"), g("d0c"), g("d2c")
            pct = lambda v, a: [float(np.percentile(v, 100 * a / 2)), float(np.percentile(v, 100 * (1 - a / 2)))]
            n_diff = int(np.sum(np.sign(d0) == np.sign(d2)))
            rob = lambda a: bool((pct(d0, a)[0] > 0 and pct(d2, a)[1] < 0) or (pct(d0, a)[1] < 0 and pct(d2, a)[0] > 0))
            sp = [r["concepts"][c]["s_pos"] for r in reps]; sn = [r["concepts"][c]["s_neg"] for r in reps]
            s_pos = np.array([np.nan if v is None else v for v in sp]); s_neg = np.array([np.nan if v is None else v for v in sn])
            oc = orig["concepts"][c] if orig else {}
            e = {"d0_orig": oc.get("d0"), "d2_orig": oc.get("d2"), "d0c_orig": oc.get("d0c"), "d2c_orig": oc.get("d2c"),
                 "d0_ci95": pct(d0, 0.05), "d2_ci95": pct(d2, 0.05), "d0_ci_bonf": pct(d0, alpha_b), "d2_ci_bonf": pct(d2, alpha_b),
                 "diff_ci95": pct(d0 - d2, 0.05), "d0c_ci95": pct(d0c, 0.05), "d2c_ci95": pct(d2c, 0.05),
                 "n_same_sign": n_diff, "p_signs_differ": 1 - n_diff / B,
                 "robust_95": rob(0.05), "robust_bonf": rob(alpha_b),
                 "s_pos_orig": oc.get("s_pos"), "s_neg_orig": oc.get("s_neg"),
                 "s_pos_q95": q95_floor(s_pos),
                 "s_pos_defined_frac": float(np.isfinite(s_pos).mean()),
                 "psiL_orig": oc.get("psiL"), "psiU_orig": oc.get("psiU"),
                 "support": {}}
            for qn in (thr or {}):
                a = np.array([r["concepts"][c]["support"][qn] for r in reps])
                e["support"][qn] = {"d0_orig": oc["support"][qn][0] if orig else None, "d2_orig": oc["support"][qn][1] if orig else None,
                                    "d0_ci95": pct(a[:, 0], 0.05), "d2_ci95": pct(a[:, 1], 0.05),
                                    "robust_95": bool((pct(a[:, 0], 0.05)[0] > 0 and pct(a[:, 1], 0.05)[1] < 0) or
                                                      (pct(a[:, 0], 0.05)[1] < 0 and pct(a[:, 1], 0.05)[0] > 0))}
            fo["concepts"][c] = e
            R.log(f"{fam} {c:22s} d0 {e['d0_orig']:+.2f} {np.round(e['d0_ci_bonf'], 2)} d2 {e['d2_orig']:+.2f} {np.round(e['d2_ci_bonf'], 2)} "
                  f"rob95={e['robust_95']} robB={e['robust_bonf']} s*={e['s_pos_orig']} s*95={e['s_pos_q95']}")
        out["families"][fam] = fo
    R.save_json(out, "vr19_primary_bootstrap.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family"); ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=1000)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        summarize()
    else:
        run(a.family, a.start, a.end)


if __name__ == "__main__":
    main()
