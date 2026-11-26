#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR48 — Inputs, branches and clipping of the plug-in bounds of Proposition 2, on the original fits (seeds 0 to 2).

For each frozen-feature family, M0 is refitted with seeds 0 to 2 as in vr25, then:
  q̂: Platt (primary) and isotonic calibration on the validation set; σ̂: the common out-of-fold gradient-boosting model on
  the tabular features (used by every plug-in in the paper) and a Platt-calibrated perceptron on the learner's own input
  (vr40_sigma.npz, the input-matched estimate).
Reported on the test population, for color variegation:
  - share of lesions where q̂ > σ̂ (impossible in the population, since Y ⊂ S), for both σ̂;
  - share of lower-tertile lesions where the second branch q̂ + 1 − σ̂ of U is active at the point tipping floor;
  - share of lesions at the clipping bounds (Platt and the internal logit clip at 1e-7; isotonic at 1e-6);
  - plug-in tipping floor on the seed-averaged curve: primary; σ̂ on the learner's input without any max rule; isotonic with
    lower bound 1e-5 and 1e-7 instead of 1e-6.
Out -> Result/vr48_nuisance_audit.json
"""
from __future__ import annotations

import os

import numpy as np

import vr_common as R
import gpu_setup  # noqa: F401
from vr19_primary_bootstrap import features, arrays, platt_fit, S_FINE
from vr24_full_bootstrap import psiL_curve, tip

C = "color_variegation"


def iso(pv, y, lo):
    from sklearn.isotonic import IsotonicRegression
    return IsotonicRegression(out_of_bounds="clip", y_min=lo, y_max=1 - lo).fit(pv, y)


def main():
    import vl_models as M
    df, fold = R.load_isic()
    Y, S = df.Y.to_numpy(), df.S.to_numpy()
    tr, va, te = (np.flatnonzero(fold == k) for k in ("train", "val", "test"))
    sig_tab = np.load(os.path.join(R.MISS_RES, "vl_nuisance.npz"))["sigma"][te]
    sig_in = np.load(os.path.join(R.RES, "vr40_sigma.npz"))
    T = R.SV.tertile(df, R.CONCEPTS[C]).fillna(-1).to_numpy()[te]
    out = {"families": {}}
    for fam, sk in (("tabular", "mlp_tab"), ("image", "mlp_img")):
        X, hid = features(df, fam)
        A = arrays(df, X)
        sg_in = sig_in[sk][te]
        curves = {k: [] for k in ("platt", "platt_sigma_input", "iso_1e-6", "iso_1e-5", "iso_1e-7")}
        st = {k: [] for k in ("q_gt_sigma_common", "q_gt_sigma_input", "platt_at_clip", "iso_at_lower", "iso_at_upper")}
        qs_pl, qs_iso = [], []
        for seed in range(3):
            m0 = M.train("erm_y", A, tr, va, hid=hid, seed=seed)
            pv, pt = M.predict(m0, X[va])["pD"], M.predict(m0, X[te])["pD"]
            a, b = platt_fit(R.logit(pv), Y[va]); q_pl = 1 / (1 + np.exp(-(a * R.logit(pt) + b)))
            qi = {lo: iso(pv, Y[va], lo).predict(pt) for lo in (1e-6, 1e-5, 1e-7)}
            curves["platt"].append(psiL_curve(q_pl, sig_tab, T))
            curves["platt_sigma_input"].append(psiL_curve(q_pl, sg_in, T))
            for lo in (1e-6, 1e-5, 1e-7):
                curves[f"iso_{lo:.0e}".replace("e-0", "e-")].append(psiL_curve(qi[lo], sig_tab, T))
            st["q_gt_sigma_common"].append(float(np.mean(q_pl > sig_tab)))
            st["q_gt_sigma_input"].append(float(np.mean(q_pl > sg_in)))
            st["platt_at_clip"].append(float(np.mean((q_pl <= 1e-7) | (q_pl >= 1 - 1e-7))))
            st["iso_at_lower"].append(float(np.mean(qi[1e-6] <= 1e-6)))
            st["iso_at_upper"].append(float(np.mean(qi[1e-6] >= 1 - 1e-6)))
            qs_pl.append(q_pl); qs_iso.append(qi[1e-6])
            R.log(f"{fam} seed {seed}: s* platt {tip(curves['platt'][-1])} σ input {tip(curves['platt_sigma_input'][-1])} "
                  f"iso {tip(curves['iso_1e-6'][-1])}; iso at lower bound {st['iso_at_lower'][-1]:.3f}")
        sstar = {k: tip(np.mean(v, 0)) for k, v in curves.items()}
        branch = {}
        for nm, qs, s in (("platt", qs_pl, sstar["platt"]), ("iso_1e-6", qs_iso, sstar["iso_1e-6"])):
            for snm, sg in (("sigma_common", sig_tab), ("sigma_input", sg_in)):
                act = [float(np.mean((q + 1 - sg < q / s)[T == 0])) for q in qs]
                branch[f"{nm}|{snm}"] = {"floor": s, "share_lower_tertile_second_branch": float(np.mean(act))}
        out["families"][fam] = {"s_star": sstar, "stats_mean_over_seeds": {k: float(np.mean(v)) for k, v in st.items()},
                                "branch": branch, "n_lower": int((T == 0).sum())}
        R.log(f"{fam}: {out['families'][fam]}")
        R.save_json(out, "vr48_nuisance_audit.json")


def iso_zero():
    """Isotonic fit of the tabular M0 without clipping: share of test lesions whose fitted value is exactly 0, or positive
    but below 1e-6, by outer tertile of color variegation. Adds key "iso_zero" to the existing output."""
    import json
    import vl_models as M
    df, fold = R.load_isic()
    Y = df.Y.to_numpy()
    tr, va, te = (np.flatnonzero(fold == k) for k in ("train", "val", "test"))
    T = R.SV.tertile(df, R.CONCEPTS[C]).fillna(-1).to_numpy()[te]
    X, hid = features(df, "tabular"); A = arrays(df, X)
    rows = []
    for seed in range(3):
        m0 = M.train("erm_y", A, tr, va, hid=hid, seed=seed)
        pv, pt = M.predict(m0, X[va])["pD"], M.predict(m0, X[te])["pD"]
        q = iso(pv, Y[va], 0.0).predict(pt)
        rows.append({f"t{t}": {"zero": float(np.mean(q[T == t] == 0)), "below": float(np.mean((q[T == t] > 0) & (q[T == t] < 1e-6)))}
                     for t in (0, 1)} | {"all": {"zero": float(np.mean(q == 0)), "below": float(np.mean((q > 0) & (q < 1e-6)))}})
        R.log(f"iso zero seed {seed}: {rows[-1]}")
    path = os.path.join(R.RES, "vr48_nuisance_audit.json"); out = json.load(open(path))
    out["iso_zero"] = {k: {kk: float(np.mean([r[k][kk] for r in rows])) for kk in ("zero", "below")} for k in rows[0]}
    R.save_json(out, "vr48_nuisance_audit.json")


if __name__ == "__main__":
    import sys
    iso_zero() if "--iso-zero" in sys.argv else main()
