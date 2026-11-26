#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR47 — Count-based checks of Lemma 1 and the marginal analysis, and a Monte Carlo audit of the all-splits labels.

(a) Exact sensitivity paths for Fig. 4. For the whole cohort and each primary concept, the counts fix q_t = P(Y=1 | t),
    v_t = P(S=1, Y=0 | t) and the verified-only log odds ratio c. For a pair (A, π₀¹) with π₁¹ = A·π₀¹ ≤ 1, the disease risk is
    p_t = q_t/π_t¹, benign verification is π_t⁰ = v_t/(1 − p_t), θ = logit p₁ − logit p₀ and δ = log B − log A with
    B = π₁⁰/π₀⁰. Lemma 1 then gives θ − δ = c exactly, so every path lies on the line δ = θ − c. θ = 0 exactly at A = RR_Y.
    The script also checks that the floor branch q_t/s of the upper bound is the active one, so that the interval
    [s·RR_Y, RR_Y/s] for RR_D coincides with the sharp interval [q₁/U₀, U₁/q₀] at the floors used in the paper.
(b) Site standardization with common weights on both sides. On sites with verified lesions in both outer tertiles, the
    standardized recorded-label risk ratio and the standardized verified-only risk ratio use the same weights w_g (share of
    outer-tertile lesions, renormalized over the supported sites). Joint event per patient-cluster resample: s·RR_Y,std > 1
    and the standardized verified-only log risk ratio < 0. A site without verified lesions in one tertile of a resample is
    dropped from both sides of that resample.
(c) Monte Carlo audit of the labels of the all-splits bootstrap (vr30, one seed per replicate, 1,000 replicates per family):
    2,000 resamples of the stored replicates, share reproducing the label.
Out -> Result/vr47_counts_audit.json
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

import vr_common as R

B = 2000
FLOORS = np.round(np.arange(0.40, 1.0001, 0.01), 2)
PRIMARY = ("color_variegation", "size", "lesion_skin_contrast")
logit = lambda p: np.log(p / (1 - p))


def part_a(df):
    Y, S = df.Y.to_numpy().astype(float), df.S.to_numpy().astype(float)
    out = {}
    for c in PRIMARY:
        T = R.SV.tertile(df, R.CONCEPTS[c]).to_numpy()
        N = np.array([(T == t).sum() for t in (0, 1)], float)
        Yt = np.array([Y[T == t].sum() for t in (0, 1)])
        St = np.array([S[T == t].sum() for t in (0, 1)])
        VB = np.array([(S * (1 - Y))[T == t].sum() for t in (0, 1)])
        q, sig, v = Yt / N, St / N, VB / N
        cc = float(np.log((Yt[1] / VB[1]) / (Yt[0] / VB[0])))
        paths, max_err = {}, 0.0
        for p0 in (0.25, 0.5, 1.0):
            rows = []
            for A in np.round(np.arange(0.5, 2.0001, 0.01), 2):
                pi = np.array([p0, A * p0])
                if pi[1] > 1:
                    continue
                p = q / pi
                assert np.all(p <= q + 1 - sig + 1e-12)
                th = logit(p[1]) - logit(p[0])
                pib = v / (1 - p)
                de = np.log(pib[1] / pib[0]) - np.log(A)
                max_err = max(max_err, abs(th - de - cc))
                rows.append({"A": float(A), "theta": float(th), "delta": float(de)})
            paths[str(p0)] = rows
        ref = {k: next(r for r in v_ if r["A"] == 1.0) for k, v_ in paths.items()}
        # floor branch of U_t = min(q_t/s, q_t + 1 − σ_t) active for every s ≥ 0.01 in both tertiles
        s_switch = [float(q[t] / (q[t] + 1 - sig[t])) for t in (0, 1)]
        out[c] = {"N": N.tolist(), "Y": Yt.tolist(), "S": St.tolist(), "verified_benign": VB.tolist(), "q": q.tolist(),
                  "sigma": sig.tolist(), "rr_y": float(q[1] / q[0]), "log_or_verified": cc, "max_identity_error": max_err,
                  "theta_zero_at_A": float(q[1] / q[0]), "reference_A1": ref, "paths": paths,
                  "floor_branch_active_above": max(s_switch)}
        R.log(f"(a) {c}: c {cc:+.3f}, RR_Y {q[1] / q[0]:.3f}, θ(A=1) " + " ".join(f"{k}:{r['theta']:.3f}" for k, r in ref.items())
              + f", identity error {max_err:.1e}, floor branch active for s > {max(s_switch):.4f}")
    return out


def part_b(df):
    Y, S = df.Y.to_numpy().astype(float), df.S.to_numpy().astype(float)
    site = df["attribution"].astype(str).to_numpy()
    sites = sorted(np.unique(site)); sidx = np.array([sites.index(x) for x in site])
    pid = df[R.SV.GROUP].to_numpy(); pats = np.unique(pid); pidx = {p: i for i, p in enumerate(pats)}
    pc = np.array([pidx[p] for p in pid])
    rng = np.random.default_rng(47)
    W = [np.ones(len(pats))] + [np.bincount(rng.integers(0, len(pats), len(pats)), minlength=len(pats)).astype(float) for _ in range(B)]
    out = {"sites": sites}
    for c in ("color_variegation", "size"):
        T = R.SV.tertile(df, R.CONCEPTS[c]).to_numpy()
        m = (T == 0) | (T == 1)
        sup = [g for g in range(len(sites)) if all((S * m * (sidx == g) * (T == t)).sum() > 0 for t in (0, 1))]
        wg = np.array([(m & (sidx == g)).sum() for g in range(len(sites))], float)
        wsup = {sites[g]: float(wg[g] / wg[sup].sum()) for g in sup}

        def stats(wp):
            wl = wp[pc] * m
            num = den = vn = vd = wsum = 0.0; dropped = 0
            for g in sup:
                k = sidx == g
                n1, n0 = (wl * k * (T == 1)).sum(), (wl * k * (T == 0)).sum()
                s1, s0 = (wl * k * (T == 1) * S).sum(), (wl * k * (T == 0) * S).sum()
                if min(n1, n0, s1, s0) == 0:
                    dropped += 1
                    continue
                num += wg[g] * (wl * k * (T == 1) * Y).sum() / n1; den += wg[g] * (wl * k * (T == 0) * Y).sum() / n0
                vn += wg[g] * (wl * k * (T == 1) * Y * S).sum() / s1; vd += wg[g] * (wl * k * (T == 0) * Y * S).sum() / s0
            rr = num / den if den > 0 else np.nan
            lrv = float(np.log(vn / vd)) if vn > 0 and vd > 0 else np.nan
            return rr, lrv, dropped

        bs = np.array([stats(w) for w in W])
        rr0, lrv0, _ = bs[0]; bb = bs[1:]
        ok = np.isfinite(bb[:, 0]) & np.isfinite(bb[:, 1])
        jc = {f"{s:.2f}": int(np.sum(ok & (bb[:, 0] * s > 1) & (bb[:, 1] < 0))) for s in FLOORS}
        jf = next((float(k) for k, v in jc.items() if v >= 0.95 * B), None)
        out[c] = {"supported_sites": [sites[g] for g in sup], "excluded_sites": [s for g, s in enumerate(sites) if g not in sup],
                  "weights": wsup, "rr_y_std": float(rr0), "rr_y_std_ci95": np.nanpercentile(bb[:, 0], [2.5, 97.5]).tolist(),
                  "point_floor": float(1 / rr0) if rr0 > 1 else None,
                  "log_rr_verified_std": float(lrv0), "log_rr_verified_std_ci95": np.nanpercentile(bb[:, 1], [2.5, 97.5]).tolist(),
                  "share_verified_negative": float(np.mean(bb[ok, 1] < 0)), "resamples_with_dropped_site": int(np.sum(bb[:, 2] > 0)),
                  "joint_counts": jc, "joint_floor": jf}
        R.log(f"(b) {c}: sites {out[c]['supported_sites']}, RR_Y,std {rr0:.2f}, verified-only log RR,std {lrv0:+.2f}, "
              f"joint floor {jf}, resamples with a dropped site {out[c]['resamples_with_dropped_site']}")
    return out


def part_c():
    rng = np.random.default_rng(470)
    a = 0.05 / 6
    out = {}
    for fam in ("tabular", "image"):
        reps = []
        for f in glob.glob(os.path.join(R.RES, "vr30", f"{fam}_*.json")):
            reps += [v for k, v in json.load(open(f))["reps"].items() if int(k) >= 0]
        out[fam] = {"B": len(reps)}
        for c in PRIMARY:
            d0 = np.array([r["concepts"][c]["d0"] for r in reps]); d2 = np.array([r["concepts"][c]["d2"] for r in reps])

            def label(x0, x2):
                b0, b2 = np.percentile(x0, [100 * a / 2, 100 * (1 - a / 2)]), np.percentile(x2, [100 * a / 2, 100 * (1 - a / 2)])
                n0, n2 = np.percentile(x0, [2.5, 97.5]), np.percentile(x2, [2.5, 97.5])
                sep = lambda u, w: (u[0] > 0 and w[1] < 0) or (u[1] < 0 and w[0] > 0)
                return "Bonferroni" if sep(b0, b2) else ("95" if sep(n0, n2) else "no"), b0.tolist()
            lab, b0 = label(d0, d2)
            rep = []
            for _ in range(2000):
                i = rng.integers(0, len(d0), len(d0))
                rep.append(label(d0[i], d2[i])[0] == lab)
            out[fam][c] = {"label": lab, "d0_bonf": b0, "reproduced": float(np.mean(rep)),
                           "tail_count": int(np.floor(len(d0) * a / 2))}
            R.log(f"(c) {fam} {c}: label {lab}, Δ_M0 Bonferroni {np.round(b0, 2)}, reproduced {np.mean(rep):.2f}")
    return out


def main():
    df, fold = R.load_isic()
    out = {"fig4_paths": part_a(df), "site_common_weights": part_b(df), "allsplits_mc": part_c()}
    R.save_json(out, "vr47_counts_audit.json")


if __name__ == "__main__":
    main()
