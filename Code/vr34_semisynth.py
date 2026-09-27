#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR34 — Bán mô phỏng trên đặc trưng THẬT của ISIC-2024: kiểm ước lượng plug-in của Proposition 2 và cầu nối Proposition 1.

Với mỗi họ chính, trên input thật x̃ và chia bệnh nhân thật, dựng các hàm của x̃ từ dữ liệu thật (seed 0):
  ℓ(x̃)   logit M0 thật đã hiệu chỉnh (chuẩn hóa), giữ hình học nguy cơ hiếm của cohort;
  m(x̃)   ridge dự đoán z của color variegation từ x̃;
  π₀(x̃)  xác suất xác minh thật trong nhãn âm (MLP học S trên Y = 0, hiệu chỉnh Platt).
Cơ chế thật (biết trước):  logit p(x̃) = α + ℓ(x̃) + b·m(x̃), α chọn theo tỉ lệ bệnh; b chọn để ψ_color thật ≈ 0.6
(cùng chiều), ≈ 0 (không), ≈ −0.3 (ngược).  s(x̃) = s_true + (1 − s_true)·expit(−1 + 1.5 m(x̃)) (xác minh ác tính phụ
thuộc concept);  D ~ Bern(p), S | D=1 ~ Bern(s), S | D=0 ~ Bern(π₀), Y = D·S.
Lưới: 3 kịch bản × tỉ lệ bệnh {0.2%, 0.5%} × s_true {0.5, 0.8} × 2 lần sinh.
Chạy nguyên quy trình: M0 (MLP) Platt và isotonic, GBM Platt cho q; σ̂ bằng MLP học S (Platt). So với sự thật: ψ_L tại
s = s_true (sai lệch so với ψ_L oracle), độ bao phủ ψ thật bởi tập ước lượng tại s_true, sàn tới hạn ước lượng so với
oracle, chứng nhận dấu tại s_true (đúng/sai). Cầu nối: M2 và ĝ (MLP) trên dữ liệu sinh, độ dốc khoảng cách logit lên log ĝ
khi identity quần thể đúng chính xác.

Đối số `local`: rủi ro bệnh như thiết kế perceptron, nhưng sàn bị vi phạm cục bộ: trong LOCAL_SHARE tổn thương có xác suất
xác minh tổng thể thật thấp nhất, s(x̃) = LOCAL_S dưới sàn giả định s_true; tập vẫn được tính tại s_true. Ghi thêm tập oracle
(nuisance thật, sàn giả định) để tách lỗi do giả định khỏi lỗi do ước lượng. Bỏ phần cầu nối.

Out -> Result/vr34_semisynth.json (mlp), vr34_semisynth_gbm.json, vr34_semisynth_local.json
"""
from __future__ import annotations

import os

import numpy as np
from scipy.optimize import brentq

import vr_common as R
import gpu_setup  # noqa: F401
from vr19_primary_bootstrap import features, arrays, platt_fit, S_FINE
from vr24_full_bootstrap import iso_fit, tip
from vr30_q_bootstrap import gbm_input

SCEN = {"aligned": 0.6, "null": 0.0, "reversed": -0.3}
PREV = [0.002, 0.005]
S_TRUE = [0.5, 0.8]
NDRAW = 2
LOCAL_SHARE, LOCAL_S = 0.30, 0.20


def expit(z):
    return 1 / (1 + np.exp(-z))


def curves(q, sig, T):
    lq = R.logit(np.clip(q, 1e-7, 1 - 1e-7)); L, U = [], []
    for s in S_FINE:
        lU = R.logit(np.clip(np.minimum(q / s, q + 1 - sig), 1e-7, 1 - 1e-7))
        L.append(lq[T == 1].mean() - lU[T == 0].mean()); U.append(lU[T == 1].mean() - lq[T == 0].mean())
    return np.array(L), np.array(U)


def main():
    import sys
    dgp = sys.argv[1] if len(sys.argv) > 1 else "mlp"
    fname = "vr34_semisynth.json" if dgp == "mlp" else f"vr34_semisynth_{dgp}.json"
    import vl_models as M
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingClassifier
    df, fold = R.load_isic()
    Yr, Sr = df.Y.to_numpy(), df.S.to_numpy()
    tr, va, te = (np.flatnonzero(fold == k) for k in ("train", "val", "test"))
    zc = ((df[R.CONCEPTS["color_variegation"]] - df[R.CONCEPTS["color_variegation"]].mean()) / df[R.CONCEPTS["color_variegation"]].std()).fillna(0.0).to_numpy()
    T = R.SV.tertile(df, R.CONCEPTS["color_variegation"]).fillna(-1).to_numpy()
    Tt = T[te]
    si = S_FINE.tolist().index
    out = {"families": {}}
    for kind in ("tabular", "image"):
        X, hid = features(df, kind); Xg = gbm_input(df, X, kind, tr)
        n = len(df)
        # hàm thật của x̃
        A = arrays(df, X)
        m0 = M.train("erm_y", A, tr, va, hid=hid, seed=0)
        a, b = platt_fit(R.logit(M.predict(m0, X[va])["pD"]), Yr[va])
        ell = a * R.logit(M.predict(m0, X)["pD"]) + b
        if dgp == "gbm":                        # thành phần rủi ro lấy từ gradient boosting thay vì perceptron
            gb0 = HistGradientBoostingClassifier(max_depth=4, max_iter=400, learning_rate=0.05, random_state=0).fit(Xg[tr], Yr[tr])
            lg0 = R.logit(np.clip(gb0.predict_proba(Xg[va])[:, 1], 1e-7, 1 - 1e-7)); a, b = platt_fit(lg0, Yr[va])
            ell = a * R.logit(np.clip(gb0.predict_proba(Xg)[:, 1], 1e-7, 1 - 1e-7)) + b
        ell = (ell - ell.mean()) / ell.std()
        sub = np.random.default_rng(0).choice(tr, 60000, replace=False)
        mfun = Ridge(alpha=10.0).fit(X[sub], zc[sub]).predict(X); mfun = (mfun - mfun.mean()) / mfun.std()
        negm = (Yr == 0).astype(np.float32)
        base = dict(sigma=np.ones(n, np.float32), L=np.ones(n, np.float32), U=np.ones(n, np.float32), m=np.zeros(n, np.float32))
        mg = M.train("erm_verified", dict(x=X, Y=Sr.astype(np.float32), F=negm, S=negm, **base), tr, va, hid=hid, seed=0)
        vneg = va[Yr[va] == 0]
        ag, bg = platt_fit(R.logit(M.predict(mg, X[vneg])["pD"]), Sr[vneg])
        pi0 = np.clip(expit(ag * R.logit(M.predict(mg, X)["pD"]) + bg), 1e-5, 1.0)
        rows = []
        for scen, target in SCEN.items():
            for prev in PREV:
                def truth(bb):
                    al = brentq(lambda al: expit(al + ell + bb * mfun).mean() - prev, -80, 30)
                    p = expit(al + ell + bb * mfun); lp = R.logit(p[te])
                    return p, float(lp[Tt == 1].mean() - lp[Tt == 0].mean())
                try:
                    bb = brentq(lambda bb: truth(bb)[1] - target, -4, 4)
                except ValueError:                      # không bao được: chọn b trên lưới gần mục tiêu nhất
                    grid = np.linspace(-6, 6, 49)
                    bb = float(grid[np.argmin([abs(truth(g)[1] - target) for g in grid])])
                p, psi = truth(bb)
                for s_true in S_TRUE:
                    s = s_true + (1 - s_true) * expit(-1 + 1.5 * mfun)
                    if dgp == "local":                  # vi phạm sàn chỉ ở vùng xác minh tổng thể thấp nhất
                        sig0 = p * s + (1 - p) * pi0
                        low = sig0 <= np.quantile(sig0, LOCAL_SHARE)
                        s = np.where(low, LOCAL_S, s)
                    q = p * s; sig = q + (1 - p) * pi0
                    oL, oU = curves(q[te], sig[te], Tt)
                    for d in range(NDRAW):
                        rng = np.random.default_rng(1000 * d + int(1e4 * prev) + int(10 * s_true) + 97 * list(SCEN).index(scen))
                        D = rng.random(n) < p; S = rng.random(n) < np.where(D, s, pi0); Y = (D & S)
                        As = dict(x=X, Y=Y.astype(np.float32), F=S.astype(np.float32), S=S.astype(np.float32), **base)
                        seed = 10 * d + 1
                        f0 = M.train("erm_y", As, tr, va, hid=hid, seed=seed); f2 = M.train("erm_verified", As, tr, va, hid=hid, seed=seed)
                        pv = M.predict(f0, X[va])["pD"]; pt = M.predict(f0, X[te])["pD"]
                        a0, b0 = platt_fit(R.logit(pv), Y[va]); iso = iso_fit(pv, Y[va].astype(float))
                        gb = HistGradientBoostingClassifier(max_depth=4, max_iter=400, learning_rate=0.05, random_state=seed).fit(Xg[tr], Y[tr])
                        lgv = R.logit(np.clip(gb.predict_proba(Xg[va])[:, 1], 1e-7, 1 - 1e-7)); a3, b3 = platt_fit(lgv, Y[va])
                        qs = {"mlp_platt": expit(a0 * R.logit(pt) + b0), "mlp_iso": np.clip(iso.predict(pt), 1e-6, 1 - 1e-6),
                              "gbm_platt": expit(a3 * R.logit(np.clip(gb.predict_proba(Xg[te])[:, 1], 1e-7, 1 - 1e-7)) + b3)}
                        Asg = dict(As); Asg["Y"] = S.astype(np.float32)
                        fs = M.train("erm_y", Asg, tr, va, hid=hid, seed=seed)
                        a4, b4 = platt_fit(R.logit(M.predict(fs, X[va])["pD"]), S[va]); sh = expit(a4 * R.logit(M.predict(fs, X[te])["pD"]) + b4)
                        rec = {"family": kind, "scenario": scen, "prev": prev, "s_true": s_true, "draw": d, "psi": psi, "b": bb,
                               "n_pos_train": int(Y[tr].sum()), "oracle_L": float(oL[si(s_true)]), "oracle_U": float(oU[si(s_true)]),
                               "oracle_tip": tip(oL), "est": {},
                               "oracle_covers": bool(oL[si(s_true)] <= psi <= oU[si(s_true)]),
                               "oracle_wrong": bool((oL[si(s_true)] > 0 and psi < -0.05) or (oU[si(s_true)] < 0 and psi > 0.05))}
                        if dgp == "local":
                            Tte = Tt; lowte = low[te]
                            rec["low_share_by_tertile"] = [float(lowte[Tte == 0].mean()), float(lowte[Tte == 1].mean())]
                        for k, qh in qs.items():
                            L, U = curves(qh, np.maximum(sh, qh), Tt)
                            rec["est"][k] = {"L": float(L[si(s_true)]), "U": float(U[si(s_true)]), "tip": tip(L),
                                             "covers": bool(L[si(s_true)] <= psi <= U[si(s_true)]),
                                             "certifies": bool(L[si(s_true)] > 0 or U[si(s_true)] < 0),
                                             "wrong": bool((L[si(s_true)] > 0 and psi < -0.05) or (U[si(s_true)] < 0 and psi > 0.05))}
                        if dgp == "local":
                            rows.append(rec)
                            R.log(f"local {kind} {scen} prev={prev} s={s_true} d={d}: ψ={psi:+.2f} oracle L {rec['oracle_L']:+.2f} "
                                  + " ".join(f"{k} L {v['L']:+.2f}" for k, v in rec["est"].items()))
                            out["families"][kind] = {"rows": rows}
                            R.save_json(out, fname)
                            continue
                        # cầu nối: identity quần thể đúng chính xác
                        vneg_s = va[Y[va] == 0]; negs = (~Y).astype(np.float32)
                        fg = M.train("erm_verified", dict(x=X, Y=S.astype(np.float32), F=negs, S=negs, **base), tr, va, hid=hid, seed=seed)
                        va2 = va[S[va]]
                        c2a, c2b = platt_fit(R.logit(M.predict(f2, X[va2])["pD"]), Y[va2])
                        cga, cgb = platt_fit(R.logit(M.predict(fg, X[vneg_s])["pD"]), S[vneg_s])
                        gap = (a0 * R.logit(pt) + b0) - (c2a * R.logit(M.predict(f2, X[te])["pD"]) + c2b)
                        lgh = np.log(expit(cga * R.logit(M.predict(fg, X[te])["pD"]) + cgb))
                        rec["bridge_slope"] = float(np.polyfit(lgh, gap, 1)[0])
                        g_true = np.log((1 - p) * pi0 / (1 - p * s))[te]      # P(S=1 | x̃, Y=0)
                        rec["bridge_slope_true_g"] = float(np.polyfit(g_true, gap, 1)[0])
                        trn = tr[~Y[tr]]
                        gbg = HistGradientBoostingClassifier(max_depth=4, max_iter=400, learning_rate=0.05, random_state=seed).fit(Xg[trn], S[trn])
                        ab_, bb_ = platt_fit(R.logit(np.clip(gbg.predict_proba(Xg[vneg_s])[:, 1], 1e-7, 1 - 1e-7)), S[vneg_s])
                        lgb = np.log(expit(ab_ * R.logit(np.clip(gbg.predict_proba(Xg[te])[:, 1], 1e-7, 1 - 1e-7)) + bb_))
                        rec["bridge_slope_gbm"] = float(np.polyfit(lgb, gap, 1)[0])
                        rows.append(rec)
                        R.log(f"{kind} {scen} prev={prev} s={s_true} d={d}: ψ={psi:+.2f} pos_train={rec['n_pos_train']} oracle tip {rec['oracle_tip']} "
                              + " ".join(f"{k} L {v['L']:+.2f} tip {v['tip']}" for k, v in rec["est"].items())
                              + f" bridge {rec['bridge_slope']:.2f} (true g {rec['bridge_slope_true_g']:.2f})")
                        out["families"][kind] = {"rows": rows}
                        R.save_json(out, fname)
        summ = {}
        for k in ("mlp_platt", "mlp_iso", "gbm_platt"):
            e = [r["est"][k] for r in rows]
            summ[k] = {"bias_L": float(np.mean([x["L"] - r["oracle_L"] for x, r in zip(e, rows)])),
                       "mad_tip": float(np.nanmean([abs(x["tip"] - r["oracle_tip"]) for x, r in zip(e, rows)])),
                       "covers": float(np.mean([x["covers"] for x in e])), "certifies": float(np.mean([x["certifies"] for x in e])),
                       "wrong": float(np.mean([x["wrong"] for x in e]))}
            for scen in SCEN:
                ee = [(x, r) for x, r in zip(e, rows) if r["scenario"] == scen]
                summ[k][scen] = {"covers": float(np.mean([x["covers"] for x, _ in ee])), "certifies": float(np.mean([x["certifies"] for x, _ in ee])),
                                 "wrong": float(np.mean([x["wrong"] for x, _ in ee])), "bias_L": float(np.mean([x["L"] - r["oracle_L"] for x, r in ee]))}
        summ["oracle"] = {"covers": float(np.mean([r["oracle_covers"] for r in rows])), "wrong": float(np.mean([r["oracle_wrong"] for r in rows]))}
        if dgp == "local":
            out["families"][kind] = {"rows": rows, "summary": summ}
            R.log(f"{kind} summary: {summ}")
            R.save_json(out, fname)
            continue
        summ["bridge_slope_gbm"] = {"median": float(np.median([r["bridge_slope_gbm"] for r in rows])),
                                    "range": [float(np.min([r["bridge_slope_gbm"] for r in rows])), float(np.max([r["bridge_slope_gbm"] for r in rows]))]}
        summ["bridge_slope"] = {"median": float(np.median([r["bridge_slope"] for r in rows])),
                                "range": [float(np.min([r["bridge_slope"] for r in rows])), float(np.max([r["bridge_slope"] for r in rows]))],
                                "true_g_median": float(np.median([r["bridge_slope_true_g"] for r in rows]))}
        out["families"][kind] = {"rows": rows, "summary": summ}
        R.log(f"{kind} summary: {summ}")
        R.save_json(out, fname)


if __name__ == "__main__":
    main()
