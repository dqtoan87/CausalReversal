#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR38 — Vì sao ψ nhạy với cách ước lượng q: trung bình trên thang logit ở đuôi rủi ro hiếm.

Trên các lần chạy gốc (seed 0–2, trung bình), cho ba ước lượng q (perceptron Platt, perceptron isotonic, gradient boosting
Platt) và hai họ chính, trong tertile dưới và trên của color variegation trên test: số tổn thương, số ác tính quan sát,
tổng q (số sự kiện kỳ vọng), trung bình q, trung bình logit q, và tỉ lệ tổn thương có q < 0.0005. Hiệu chỉnh ràng buộc
tổng q theo khoảng, không ràng buộc trung bình logit q; vì logit lõm và rất dốc gần 0, cùng một số sự kiện kỳ vọng có thể
đi với trung bình logit rất khác nhau.

Out -> Result/vr38_logit_tail.json
"""
from __future__ import annotations

import numpy as np

import vr_common as R
import gpu_setup  # noqa: F401
from vr19_primary_bootstrap import features, arrays, platt_fit
from vr24_full_bootstrap import iso_fit
from vr30_q_bootstrap import gbm_input


def main():
    import vl_models as M
    from sklearn.ensemble import HistGradientBoostingClassifier
    df, fold = R.load_isic()
    Y = df.Y.to_numpy()
    tr, va, te = (np.flatnonzero(fold == k) for k in ("train", "val", "test"))
    T = R.SV.tertile(df, R.CONCEPTS["color_variegation"]).fillna(-1).to_numpy()[te]
    out = {"families": {}}
    for kind in ("tabular", "image"):
        X, hid = features(df, kind); A = arrays(df, X); Xg = gbm_input(df, X, kind, tr)
        acc = {k: [] for k in ("mlp_platt", "mlp_iso", "gbm_platt")}
        for seed in range(3):
            m0 = M.train("erm_y", A, tr, va, hid=hid, seed=seed)
            pv, pt = M.predict(m0, X[va])["pD"], M.predict(m0, X[te])["pD"]
            a, b = platt_fit(R.logit(pv), Y[va]); acc["mlp_platt"].append(1 / (1 + np.exp(-(a * R.logit(pt) + b))))
            acc["mlp_iso"].append(np.clip(iso_fit(pv, Y[va]).predict(pt), 1e-6, 1 - 1e-6))
            gb = HistGradientBoostingClassifier(max_depth=4, max_iter=400, learning_rate=0.05, random_state=seed).fit(Xg[tr], Y[tr])
            lv = R.logit(np.clip(gb.predict_proba(Xg[va])[:, 1], 1e-7, 1 - 1e-7)); a, b = platt_fit(lv, Y[va])
            acc["gbm_platt"].append(1 / (1 + np.exp(-(a * R.logit(np.clip(gb.predict_proba(Xg[te])[:, 1], 1e-7, 1 - 1e-7)) + b))))
        fo = {}
        for k, qs in acc.items():
            rows = {}
            for t in (0, 1):
                m = T == t
                per = [{"sum_q": float(q[m].sum()), "mean_q": float(q[m].mean()), "mean_logit_q": float(R.logit(np.clip(q[m], 1e-7, 1 - 1e-7)).mean()),
                        "frac_below_5e-4": float(np.mean(q[m] < 5e-4))} for q in qs]
                rows[str(t)] = {kk: float(np.mean([p[kk] for p in per])) for kk in per[0]} | {"n": int(m.sum()), "observed": int(Y[te][m].sum())}
            rows["contrast_logit"] = rows["1"]["mean_logit_q"] - rows["0"]["mean_logit_q"]
            rows["contrast_log_mean"] = float(np.log(rows["1"]["mean_q"] / rows["0"]["mean_q"]))
            fo[k] = rows
            R.log(f"{kind} {k}: lower tertile obs {rows['0']['observed']} expected {rows['0']['sum_q']:.1f} mean logit {rows['0']['mean_logit_q']:.2f}; "
                  f"upper obs {rows['1']['observed']} expected {rows['1']['sum_q']:.1f} mean logit {rows['1']['mean_logit_q']:.2f}; "
                  f"contrast logit {rows['contrast_logit']:.2f} vs log mean {rows['contrast_log_mean']:.2f}")
        out["families"][kind] = fo
    R.save_json(out, "vr38_logit_tail.json")


if __name__ == "__main__":
    main()
