#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR40 — Tách thay đổi thành phần quần thể khỏi phần còn lại của sự suy giảm trong phân tích overlap.

Giới hạn quần thể test vào σ̂ ≥ ngưỡng làm đổi phân phối các concept bên trong mỗi tertile, nên Δ có thể đổi ngay cả khi
hàm dự đoán không đổi. Với M0 và M2 (logit, trung bình seed 0–2 của lần chạy gốc) script báo:
  full        Δ trên toàn quần thể test;
  support     Δ thô trên vùng hỗ trợ (ba ước lượng σ̂ × ngưỡng 1/5/10%);
  transported Δ trên vùng hỗ trợ sau khi, trong từng tertile, trọng số hóa tổn thương có hỗ trợ bằng 1/P̂(có hỗ trợ | năm
              concept) (logistic trên tổn thương của tertile đó trong toàn quần thể, cắt ở phân vị 99), để phân phối năm concept
              trong mỗi tertile khớp với toàn quần thể. Phần chênh full − transported là phần không giải thích được bằng
              thành phần concept; phần support − transported là phần do thành phần.
Chẩn đoán phụ trên toàn quần thể: tương phản có điều kiện khi cân bằng hai tertile theo bốn concept còn lại (IPW), một
estimand khác (tương phản giữ các concept khác cố định), chỉ để cho thấy tương phản biên chứa phần chung với concept tương quan.
Khoảng 95%: 300 lần lấy lại bệnh nhân test, khớp lại mô hình xu hướng mỗi lần, có điều kiện trên learner đã khớp.

Out -> Result/vr40_support_composition.json ; σ̂ perceptron lưu ở Result/vr40_sigma.npz
"""
from __future__ import annotations

import os

import numpy as np
from sklearn.linear_model import LogisticRegression

import vr_common as R

QS = [0.01, 0.05, 0.10]
B = 300
PRIMARY = ["color_variegation", "size", "lesion_skin_contrast"]


def sigmas(df, fold):
    path = os.path.join(R.RES, "vr40_sigma.npz")
    S = df.S.to_numpy()
    out = {"nuisance": np.load(os.path.join(R.MISS_RES, "vl_nuisance.npz"))["sigma"]}
    if os.path.exists(path):
        z = np.load(path); out.update({k: z[k] for k in ("mlp_tab", "mlp_img")}); return out
    import gpu_setup  # noqa: F401
    import vl_models as M
    from vr19_primary_bootstrap import features, arrays, platt_fit
    tr, va = np.flatnonzero(fold == "train"), np.flatnonzero(fold == "val")
    for kind, name in (("tabular", "mlp_tab"), ("image", "mlp_img")):
        X, hid = features(df, kind); A = arrays(df, X); A["Y"] = S.astype(np.float32)
        ps = []
        for seed in range(3):
            m = M.train("erm_y", A, tr, va, hid=hid, seed=seed)
            a, b = platt_fit(R.logit(M.predict(m, X[va])["pD"]), S[va])
            ps.append(1 / (1 + np.exp(-(a * R.logit(M.predict(m, X)["pD"]) + b))))
        out[name] = np.mean(ps, 0)
        R.log(f"{name}: fitted")
    np.savez(path, mlp_tab=out["mlp_tab"], mlp_img=out["mlp_img"])
    return out


def wmean_diff(l, T, w):
    return float(np.average(l[T == 1], weights=w[T == 1]) - np.average(l[T == 0], weights=w[T == 0]))


def transport_weights(T, Z, sup, w):
    """Trọng số cho tổn thương có hỗ trợ để phân phối Z trong mỗi tertile khớp toàn quần thể (tertile ngoài)."""
    out = np.zeros(len(T))
    for t in (0, 1):
        k = T == t
        p = LogisticRegression(C=1.0, max_iter=500).fit(Z[k], sup[k], sample_weight=w[k]).predict_proba(Z[k])[:, 1]
        p = np.clip(p, 1e-3, 1.0)
        p = np.maximum(p, np.quantile(p[sup[k]], 0.01))            # cắt trọng số ở phân vị 99
        out[k] = np.where(sup[k], w[k] / p, 0.0)
    return out


def main():
    df, fold = R.load_isic()
    S = df.S.to_numpy()
    heads = np.load(os.path.join(R.RES, "closing_heads.npz"))
    te = heads["test_idx"]
    sig = sigmas(df, fold)
    tr = np.flatnonzero(fold == "train"); trS = tr[S[tr] == 1]
    Zc = {c: ((df[col] - df[col].mean()) / df[col].std()).fillna(0.0).to_numpy()[te] for c, col in R.CONCEPTS.items()}
    Z5 = np.column_stack([Zc[c] for c in R.CONCEPTS])
    Tt = {c: R.SV.tertile(df, R.CONCEPTS[c]).fillna(-1).to_numpy()[te] for c in PRIMARY}
    L = {f"{k}|{m}": np.mean([heads[f"{k}_{m}_s{s}"] for s in range(3)], 0) for k in ("tabular", "image") for m in ("M0", "M2")}
    sup = {}
    for nm, sg in sig.items():
        for q in QS:
            sup[f"{nm}|{q}"] = sg[te] >= float(np.quantile(sg[trS], q))
    pid = df[R.SV.GROUP].to_numpy()[te]; pats = np.unique(pid); pidx = {p: i for i, p in enumerate(pats)}
    pc = np.array([pidx[p] for p in pid])
    rng = np.random.default_rng(40)
    W = [np.ones(len(pats))] + [np.bincount(rng.integers(0, len(pats), len(pats)), minlength=len(pats)).astype(float) for _ in range(B)]
    res = {"full": {}, "support": {}, "partial_full": {}}
    for c in PRIMARY:
        m = (Tt[c] == 0) | (Tt[c] == 1)
        T = Tt[c][m]; Zm = Z5[m]; others = np.column_stack([Zc[o] for o in R.CONCEPTS if o != c])[m]
        vals = {"full": [], "partial": [], **{k: [] for k in sup}}
        for w in W:
            wl = w[pc][m]
            vals["full"].append({fm: wmean_diff(l[m], T, wl) for fm, l in L.items()})
            e = LogisticRegression(C=1.0, max_iter=500).fit(others, T, sample_weight=wl).predict_proba(others)[:, 1]
            e = np.clip(e, 1e-3, 1 - 1e-3); wp = wl * np.where(T == 1, 1 / e, 1 / (1 - e))
            vals["partial"].append({fm: wmean_diff(l[m], T, wp) for fm, l in L.items()})
            for k, s in sup.items():
                sm = s[m]
                wt = transport_weights(T, Zm, sm, wl)
                vals[k].append({fm: {"raw": wmean_diff(l[m][sm], T[sm], wl[sm]), "transported": wmean_diff(l[m], T, wt)} for fm, l in L.items()})
        for fm in L:
            bs = np.array([v[fm] for v in vals["full"][1:]])
            res["full"][f"{fm}|{c}"] = {"est": vals["full"][0][fm], "ci": np.percentile(bs, [2.5, 97.5]).tolist()}
            bs = np.array([v[fm] for v in vals["partial"][1:]])
            res["partial_full"][f"{fm}|{c}"] = {"est": vals["partial"][0][fm], "ci": np.percentile(bs, [2.5, 97.5]).tolist()}
            for k in sup:
                e = {"share": float(sup[k].mean())}
                for kk in ("raw", "transported"):
                    bs = np.array([v[fm][kk] for v in vals[k][1:]])
                    e[kk] = vals[k][0][fm][kk]; e[kk + "_ci"] = np.percentile(bs, [2.5, 97.5]).tolist()
                res["support"][f"{fm}|{k}|{c}"] = e
        R.log(f"{c}: " + " ".join(f"{fm} full {res['full'][f'{fm}|{c}']['est']:+.2f} "
                                  + " ".join(f"{k}: {res['support'][f'{fm}|{k}|{c}']['raw']:+.2f}/{res['support'][f'{fm}|{k}|{c}']['transported']:+.2f}"
                                             for k in sup if k.endswith("0.05")) for fm in L))
        R.save_json({"B": B, **res}, "vr40_support_composition.json")


if __name__ == "__main__":
    main()
