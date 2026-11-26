#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR49 — Indeterminate histopathology and the disease-side count analyses.

ISIC-2024 records three histopathology classes in iddx_1: Malignant, Benign and Indeterminate. The binary target is 1 for
Malignant only, so the 114 Indeterminate lesions (all biopsied) are biopsied recorded negatives. The learner analyses use the
recorded label and are unaffected. For the disease side, three codings are compared on counts:
  dataset  D = 1 for Malignant, 0 for Benign and Indeterminate (the binary rule of the release; primary)
  exclude  Indeterminate lesions are treated as unverified for disease: S_det = 1 for Malignant or Benign histopathology
  malign   Indeterminate lesions are counted as malignant (an extreme coding)
For each coding: the cross-table by split and outer tertile, RR of the (coded) label between the outer tertiles, the
verified-only log odds ratio, the observed benign ratio B_V and the 95 percent resampling stability threshold of the marginal
association reversal (lower-tertile floor grid step 0.05, 2,000 patient-cluster resamples), as in vr35.
Out -> Result/vr49_indeterminate.json
"""
from __future__ import annotations

import numpy as np

import vr_common as R

B = 2000
FLOORS = np.round(np.arange(0.5, 1.0001, 0.05), 2)


def main():
    df, fold = R.load_isic()
    dx = df["iddx_1"].fillna("none").to_numpy()
    S = df.S.to_numpy().astype(float); Y = df.Y.to_numpy().astype(float)
    ind = (dx == "Indeterminate").astype(float)
    assert np.all(S[ind == 1] == 1) and np.all(Y[ind == 1] == 0) and np.all((dx == "Malignant") == (Y == 1))
    codings = {"dataset": (Y, S), "exclude": (Y, S * (1 - ind)), "malign": (np.maximum(Y, ind), S)}
    pid = df[R.SV.GROUP].to_numpy()
    out = {"B": B, "crosstab": {}, "codings": {}}
    for k in ("train", "val", "test"):
        m = fold == k
        out["crosstab"][k] = {"malignant": int(Y[m].sum()), "benign_biopsied": int(((dx == "Benign") & (S == 1) & m).sum()),
                              "indeterminate": int(ind[m].sum())}
    for c in ("color_variegation", "size", "lesion_skin_contrast"):
        T = R.SV.tertile(df, R.CONCEPTS[c]).to_numpy()
        out["crosstab"][c] = {f"t{t}": {"malignant": int(Y[T == t].sum()), "benign_biopsied": int(((dx == "Benign") & (S == 1) & (T == t)).sum()),
                                        "indeterminate": int(ind[T == t].sum())} for t in (0, 1)}
    rng = np.random.default_rng(49)
    for popn, mask in (("cohort", np.ones(len(df), bool)), ("test", fold == "test")):
        ix = np.flatnonzero(mask); pats = np.unique(pid[ix]); pidx = {p: i for i, p in enumerate(pats)}
        pc = np.array([pidx[p] for p in pid[ix]])
        W = np.zeros((B, len(pats)))
        for b in range(B):
            np.add.at(W[b], rng.integers(0, len(pats), len(pats)), 1.0)
        for cod, (yy, ss) in codings.items():
            for c in ("color_variegation", "size", "lesion_skin_contrast"):
                T = R.SV.tertile(df, R.CONCEPTS[c]).to_numpy()[ix]
                y, s = yy[ix], ss[ix]

                def stats(w):
                    wl = w[pc]
                    p1 = (wl * y * (T == 1)).sum() / (wl * (T == 1)).sum(); p0 = (wl * y * (T == 0)).sum() / (wl * (T == 0)).sum()
                    a = (wl * s * y * (T == 1)).sum(); bb = (wl * s * (1 - y) * (T == 1)).sum()
                    cc = (wl * s * y * (T == 0)).sum(); dd = (wl * s * (1 - y) * (T == 0)).sum()
                    n1 = (wl * (1 - y) * (T == 1)).sum(); n0 = (wl * (1 - y) * (T == 0)).sum()
                    return p1 / p0, np.log((a / bb) / (cc / dd)), (bb / n1) / (dd / n0)
                rr0, lor0, bv0 = stats(np.ones(len(pats)))
                bs = np.array([stats(W[b]) for b in range(B)])
                jc = {f"{f:.2f}": int(np.sum((bs[:, 0] * f > 1) & (bs[:, 1] < 0))) for f in FLOORS}
                jf = next((float(k) for k, v in jc.items() if v >= 0.95 * B), None)
                out["codings"][f"{popn}|{cod}|{c}"] = {
                    "rr": float(rr0), "rr_ci95": np.percentile(bs[:, 0], [2.5, 97.5]).tolist(), "point_floor": float(1 / rr0) if rr0 > 1 else None,
                    "log_or_verified": float(lor0), "log_or_verified_ci95": np.percentile(bs[:, 1], [2.5, 97.5]).tolist(),
                    "B_V": float(bv0), "joint_floor": jf}
                R.log(f"{popn} {cod} {c}: RR {rr0:.2f} logOR|S {lor0:+.2f} {np.round(np.percentile(bs[:, 1], [2.5, 97.5]), 2)} B_V {bv0:.2f} floor {jf}")
    R.save_json(out, "vr49_indeterminate.json")


if __name__ == "__main__":
    main()
