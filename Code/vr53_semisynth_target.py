#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR53 — The semi-synthetic target and oracle set of vr34 on the unclipped logit scale of Proposition 2.

vr34 computed the true ψ and the oracle endpoints with R.logit, which clips probabilities to [1e-7, 1 − 1e-7]. Section III-B
defines ψ on the unclipped logit, so that ψ = Δ_t ℓ + b·Δ_t m exactly. This script rebuilds the fixed functions ℓ, m and π₀
of vr34 (same seeds, same fits), checks that the stored clipped ψ and oracle endpoints are reproduced from the stored b and
prevalence, and recomputes, for every stored setting:
  psi_unclipped                  mean of α + ℓ + b·m in the upper minus the lower tertile (test population)
  oracle L, U at the true floor  stratum formula of Proposition 2 with the true q and σ, no clipping
  oracle tipping floor           on the grid S_FINE, no clipping
The plug-in endpoints are the stored ones: the plug-in estimator clips q̂ at 1e-7, as in the analysis of ISIC-2024. Each plug-in
set is then classified against the unclipped target and oracle:
  null_exclusion        |ψ| ≤ 0.05 and the plug-in set excludes zero
  same_side_oracle0     the plug-in set excludes zero on the side of ψ (|ψ| > 0.05) while the oracle set contains zero
  wrong                 the plug-in set excludes zero on the side opposite to ψ, |ψ| > 0.05
and the plug-in tipping floor is compared with the oracle one (lower, equal or higher; none when ψ_L never turns positive).
Out -> Result/vr53_semisynth_target.json
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

import vr_common as R
import gpu_setup  # noqa: F401
from vr19_primary_bootstrap import features, arrays, platt_fit, S_FINE
from vr24_full_bootstrap import tip
from vr30_q_bootstrap import gbm_input
from vr34_semisynth import expit, curves, LOCAL_SHARE, LOCAL_S

FILES = {"mlp": "vr34_semisynth.json", "gbm": "vr34_semisynth_gbm.json", "local": "vr34_semisynth_local.json"}
EST = ("mlp_platt", "mlp_iso", "gbm_platt")


def curves_raw(q, omp, pi0, T):
    """Unclipped stratum endpoints. q = p·s, omp = 1 − p computed as expit(−η), and 1 − σ + q = 1 − (1 − p)·π₀, so the
    second branch has log-odds log(1 − (1 − p)π₀) − log((1 − p)π₀), finite even where p rounds to one."""
    lq = np.log(q) - np.log1p(-q); L, U = [], []
    v = omp * pi0                                     # 1 − u on the second branch
    l2 = np.log1p(-v) - np.log(v)
    for s in S_FINE:
        r = q / s
        with np.errstate(invalid="ignore", divide="ignore"):
            l1 = np.log(r) - np.log1p(-r)
        lU = np.where(r < 1 - v, l1, l2)
        L.append(lq[T == 1].mean() - lU[T == 0].mean()); U.append(lU[T == 1].mean() - lq[T == 0].mean())
    return np.array(L), np.array(U)


def main():
    import json, os
    import vl_models as M
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingClassifier
    df, fold = R.load_isic()
    Yr, Sr = df.Y.to_numpy(), df.S.to_numpy()
    tr, va, te = (np.flatnonzero(fold == k) for k in ("train", "val", "test"))
    cv = df[R.CONCEPTS["color_variegation"]]
    zc = ((cv - cv.mean()) / cv.std()).fillna(0.0).to_numpy()
    Tt = R.SV.tertile(df, R.CONCEPTS["color_variegation"]).fillna(-1).to_numpy()[te]
    si = S_FINE.tolist().index
    stored = {k: json.load(open(os.path.join(R.RES, f)))["families"] for k, f in FILES.items()}
    out = {"files": FILES, "designs": {}}
    for kind in ("tabular", "image"):
        X, hid = features(df, kind); Xg = gbm_input(df, X, kind, tr)
        n = len(df)
        A = arrays(df, X)
        m0 = M.train("erm_y", A, tr, va, hid=hid, seed=0)
        a, b = platt_fit(R.logit(M.predict(m0, X[va])["pD"]), Yr[va])
        ells = {"mlp": a * R.logit(M.predict(m0, X)["pD"]) + b}
        gb0 = HistGradientBoostingClassifier(max_depth=4, max_iter=400, learning_rate=0.05, random_state=0).fit(Xg[tr], Yr[tr])
        lg0 = R.logit(np.clip(gb0.predict_proba(Xg[va])[:, 1], 1e-7, 1 - 1e-7)); a, b = platt_fit(lg0, Yr[va])
        ells["gbm"] = a * R.logit(np.clip(gb0.predict_proba(Xg)[:, 1], 1e-7, 1 - 1e-7)) + b
        ells["local"] = ells["mlp"]
        ells = {k: (v - v.mean()) / v.std() for k, v in ells.items()}
        sub = np.random.default_rng(0).choice(tr, 60000, replace=False)
        mfun = Ridge(alpha=10.0).fit(X[sub], zc[sub]).predict(X); mfun = (mfun - mfun.mean()) / mfun.std()
        negm = (Yr == 0).astype(np.float32)
        base = dict(sigma=np.ones(n, np.float32), L=np.ones(n, np.float32), U=np.ones(n, np.float32), m=np.zeros(n, np.float32))
        mg = M.train("erm_verified", dict(x=X, Y=Sr.astype(np.float32), F=negm, S=negm, **base), tr, va, hid=hid, seed=0)
        vneg = va[Yr[va] == 0]
        ag, bg = platt_fit(R.logit(M.predict(mg, X[vneg])["pD"]), Sr[vneg])
        pi0 = np.clip(expit(ag * R.logit(M.predict(mg, X)["pD"]) + bg), 1e-5, 1.0)
        for dgp in FILES:
            ell = ells[dgp]
            rows_out, err = [], []
            cache = {}
            for r in stored[dgp][kind]["rows"]:
                key = (r["scenario"], r["prev"], r["s_true"])
                if key not in cache:
                    bb, prev = r["b"], r["prev"]
                    al = brentq(lambda al: expit(al + ell + bb * mfun).mean() - prev, -80, 30)
                    eta = al + ell + bb * mfun; p = expit(eta)
                    lp = R.logit(p[te])
                    psi_c = float(lp[Tt == 1].mean() - lp[Tt == 0].mean())
                    psi_u = float(eta[te][Tt == 1].mean() - eta[te][Tt == 0].mean())
                    s = r["s_true"] + (1 - r["s_true"]) * expit(-1 + 1.5 * mfun)
                    if dgp == "local":
                        sig0 = p * s + (1 - p) * pi0
                        s = np.where(sig0 <= np.quantile(sig0, LOCAL_SHARE), LOCAL_S, s)
                    q = p * s; sig = q + (1 - p) * pi0
                    oL, oU = curves(q[te], sig[te], Tt)
                    rL, rU = curves_raw(q[te], expit(-eta)[te], pi0[te], Tt)
                    assert np.all(np.isfinite(rL)) and np.all(np.isfinite(rU))
                    cache[key] = dict(psi_clip=psi_c, psi=psi_u, oL_c=float(oL[si(r["s_true"])]), oU_c=float(oU[si(r["s_true"])]),
                                      oL=float(rL[si(r["s_true"])]), oU=float(rU[si(r["s_true"])]), otip=tip(rL),
                                      share_p_clipped=float(np.mean(p[te] < 1e-7)), share_q_clipped=float(np.mean(q[te] < 1e-7)),
                                      share_p_one=float(np.mean(p[te] > 1 - 1e-7)))
                c = cache[key]
                err.append(max(abs(c["psi_clip"] - r["psi"]), abs(c["oL_c"] - r["oracle_L"]), abs(c["oU_c"] - r["oracle_U"])))
                psi = c["psi"]
                o0 = c["oL"] <= 0 <= c["oU"]
                row = {"scenario": r["scenario"], "prev": r["prev"], "s_true": r["s_true"], "draw": r["draw"], "b": r["b"],
                       "psi_stored": r["psi"], "psi_unclipped": psi, "psi_clipped_rebuilt": c["psi_clip"],
                       "oracle_L_clipped_rebuilt": c["oL_c"], "oracle_U_clipped_rebuilt": c["oU_c"], "oracle_L": c["oL"], "oracle_U": c["oU"], "oracle_tip": c["otip"],
                       "oracle_covers": bool(c["oL"] <= psi <= c["oU"]), "oracle_excludes_zero": not o0,
                       "share_p_clipped": c["share_p_clipped"], "share_q_clipped": c["share_q_clipped"],
                       "share_p_one": c["share_p_one"], "est": {}}
                for k in EST:
                    e = r["est"].get(k)
                    if e is None:
                        continue
                    L, U = e["L"], e["U"]
                    ex = L > 0 or U < 0
                    row["est"][k] = {"L": L, "U": U, "tip": e["tip"], "covers": bool(L <= psi <= U), "excludes_zero": ex,
                                     "null_exclusion": bool(abs(psi) <= 0.05 and ex),
                                     "same_side_oracle0": bool(abs(psi) > 0.05 and o0 and ((L > 0 and psi > 0) or (U < 0 and psi < 0))),
                                     "wrong": bool(abs(psi) > 0.05 and ((L > 0 and psi < 0) or (U < 0 and psi > 0))),
                                     "dL": L - c["oL"], "dU": U - c["oU"]}
                rows_out.append(row)
            out["designs"].setdefault(dgp, {})[kind] = {"max_reproduction_error": float(max(err)), "rows": rows_out}
            R.log(f"{dgp} {kind}: reproduction error {max(err):.2e}; ψ stored vs unclipped max |diff| "
                  f"{max(abs(x['psi_stored'] - x['psi_unclipped']) for x in rows_out):.4f}")
        R.save_json(out, "vr53_semisynth_target.json")
    summarize(out)
    R.save_json(out, "vr53_semisynth_target.json")


def summarize(out):
    sm = {}
    for dgp, fams in out["designs"].items():
        for kind, d in fams.items():
            for sc in ("aligned", "null", "reversed"):
                rr = [r for r in d["rows"] if r["scenario"] == sc]
                if not rr:
                    continue
                e = [(r, r["est"][k]) for r in rr for k in EST if k in r["est"]]
                fin = lambda t: t is not None and np.isfinite(t)          # tip() is NaN when ψ_L never turns positive
                tips = [(x["tip"] if fin(x["tip"]) else None, r["oracle_tip"] if fin(r["oracle_tip"]) else None) for r, x in e]
                sm[f"{dgp}|{kind}|{sc}"] = {
                    "n_sets": len(e), "n_oracle": len(rr),
                    "psi_unclipped_range": [min(r["psi_unclipped"] for r in rr), max(r["psi_unclipped"] for r in rr)],
                    "psi_shift_max": max(abs(r["psi_stored"] - r["psi_unclipped"]) for r in rr),
                    "clip_effect_psi_max": max(abs(r["psi_clipped_rebuilt"] - r["psi_unclipped"]) for r in rr),
                    "clip_effect_oracle_max": max(max(abs(r["oracle_L_clipped_rebuilt"] - r["oracle_L"]),
                                                      abs(r["oracle_U_clipped_rebuilt"] - r["oracle_U"])) for r in rr),
                    "share_p_clipped_max": max(r["share_p_clipped"] for r in rr),
                    "share_q_clipped_max": max(r["share_q_clipped"] for r in rr),
                    "share_p_one_max": max(r["share_p_one"] for r in rr),
                    "oracle_covers": sum(r["oracle_covers"] for r in rr), "oracle_excludes_zero": sum(r["oracle_excludes_zero"] for r in rr),
                    "covers": sum(x["covers"] for _, x in e), "excludes_zero": sum(x["excludes_zero"] for _, x in e),
                    "null_exclusion": sum(x["null_exclusion"] for _, x in e),
                    "null_pos": sum(x["null_exclusion"] and x["L"] > 0 for _, x in e),
                    "null_neg": sum(x["null_exclusion"] and x["U"] < 0 for _, x in e),
                    "same_side_oracle0": sum(x["same_side_oracle0"] for _, x in e),
                    "same_side_dU": [x["dU"] for _, x in e if x["same_side_oracle0"] and x["U"] < 0],
                    "same_side_dL": [x["dL"] for _, x in e if x["same_side_oracle0"] and x["L"] > 0],
                    "wrong": sum(x["wrong"] for _, x in e),
                    "tip_lower": sum(t is not None and (o is None or t < o - 1e-9) for t, o in tips),
                    "tip_equal": sum(t is not None and o is not None and abs(t - o) < 1e-9 for t, o in tips),
                    "tip_higher": sum(o is not None and (t is None or t > o + 1e-9) for t, o in tips),
                    "tip_both_none": sum(t is None and o is None for t, o in tips),
                    "mean_dL": float(np.mean([x["dL"] for _, x in e]))}
    out["summary"] = sm
    for k, v in sm.items():
        R.log(f"{k}: " + ", ".join(f"{a} {b}" for a, b in v.items() if not isinstance(b, list)))


if __name__ == "__main__":
    import sys
    if "--summarize" in sys.argv:
        import json, os
        o = json.load(open(os.path.join(R.RES, "vr53_semisynth_target.json")))
        summarize(o); R.save_json(o, "vr53_semisynth_target.json")
    else:
        main()
