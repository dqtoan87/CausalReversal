#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR51 — Primary bootstrap on the direct logit scale, and M2 retrained without the indeterminate biopsies.

Replicate b reuses the patient resample of vr19 (rng(50_000 + b): training patients, then test patients) and the three seeds
of the primary three-seed bootstrap (b, b + 100_000, b + 200_000; vr19 and vr39). For each seed it refits
  M0     on all training lesions with the released label (image family only: the logit cap binds only there, vr50)
  M2     on biopsied lesions, early-stopped on biopsied validation lesions (as in the paper)
  M2x    on biopsied lesions with a determinate histopathology (indeterminate excluded from training and early stopping)
and stores, for the five concepts on the resampled test population, the tertile contrast with the clipped logit (the measure
of the paper) and with the direct pre-sigmoid logit. The clipped values reproduce the stored vr19 / vr39 replicates, which is
checked in the summary.
Modes:
  --family F --start s --end e      replicates s..e−1 (resumable; one JSON per lane)
  --points                          original fits (seeds 0 to 2, unresampled), both families, plus the coding that counts the
                                    indeterminate biopsies as malignant (M0 and M2 refitted with that label)
  --summarize                       intervals, Bonferroni labels and Monte Carlo reproduction, as vr43
Out -> Result/vr51/{family}_{s}_{e}.json, Result/vr51/points.json, Result/vr51_direct_indeterminate.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

import vr_common as R
import gpu_setup  # noqa: F401
from vr19_primary_bootstrap import features, arrays

OUTD = os.path.join(R.RES, "vr51")
CONC = ("color_variegation", "size", "lesion_skin_contrast", "asymmetry", "border_irregularity")
PRIMARY = ("color_variegation", "size", "lesion_skin_contrast")
SEEDS = (0, 100_000, 200_000)
A_B = 0.05 / 6


@torch.no_grad()
def logits(model, x, bs=65536):
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(x), bs):
        out.append(model(torch.as_tensor(x[i:i + bs], dtype=torch.float32, device=dev))["logit_D"].float().cpu().numpy())
    ld = np.concatenate(out).astype(np.float64)
    pc = (1 / (1 + np.exp(-ld.astype(np.float32)))).astype(np.float32)     # float32 sigmoid, as predict(): then clipped logit
    return R.logit(pc.astype(np.float64)), ld


def setup(family):
    df, fold = R.load_isic()
    X, hid = features(df, family)
    A = arrays(df, X)
    S = df.S.to_numpy().astype(np.float32)
    ind = (df["iddx_1"].fillna("") == "Indeterminate").to_numpy()
    Ax = dict(A); Sx = S * (~ind); Ax["S"] = Sx.astype(np.float32); Ax["F"] = Ax["S"]
    Am = dict(A); Am["Y"] = np.maximum(A["Y"], ind.astype(np.float32))
    T = {c: R.SV.tertile(df, R.CONCEPTS[c]).fillna(-1).to_numpy() for c in CONC}
    tr_all, va, te_all = (np.flatnonzero(fold == k) for k in ("train", "val", "test"))
    pid = df[R.SV.GROUP].to_numpy()
    trp, tep = np.unique(pid[tr_all]), np.unique(pid[te_all])
    tr_by = {p: tr_all[pid[tr_all] == p] for p in trp}; te_by = {p: te_all[pid[te_all] == p] for p in tep}
    return dict(df=df, X=X, hid=hid, A=A, Ax=Ax, Am=Am, T=T, tr_all=tr_all, va=va, te_all=te_all, trp=trp, tep=tep,
                tr_by=tr_by, te_by=te_by)


def contrasts(lc, ld, T, te):
    return {c: {"clip": float(lc[T[c][te] == 1].mean() - lc[T[c][te] == 0].mean()),
                "direct": float(ld[T[c][te] == 1].mean() - ld[T[c][te] == 0].mean())} for c in CONC}


def fit_eval(M, E, arrs, meth, tr, te, seed):
    m = M.train(meth, arrs, tr, E["va"], hid=E["hid"], seed=seed)
    ute, inv = np.unique(te, return_inverse=True)
    lc, ld = logits(m, E["X"][ute])
    return contrasts(lc[inv], ld[inv], E["T"], te), float(np.mean(lc[inv] <= -16.1))


def run(family, start, end):
    import vl_models as M
    E = setup(family)
    os.makedirs(OUTD, exist_ok=True)
    path = os.path.join(OUTD, f"{family}_{start}_{end}.json")
    res = json.load(open(path)) if os.path.exists(path) else {"family": family, "reps": {}}
    for b in range(start, end):
        rng = np.random.default_rng(50_000 + b)
        tr = np.concatenate([E["tr_by"][p] for p in rng.choice(E["trp"], len(E["trp"]))])
        te = np.concatenate([E["te_by"][p] for p in rng.choice(E["tep"], len(E["tep"]))])
        for off in SEEDS:
            key = f"{b}|{off}"
            if key in res["reps"]:
                continue
            r = {}
            if family == "image":
                r["M0"], r["M0_cap"] = fit_eval(M, E, E["A"], "erm_y", tr, te, b + off)
            r["M2"], _ = fit_eval(M, E, E["A"], "erm_verified", tr, te, b + off)
            r["M2x"], _ = fit_eval(M, E, E["Ax"], "erm_verified", tr, te, b + off)
            res["reps"][key] = r
        json.dump(res, open(path + ".tmp", "w")); os.replace(path + ".tmp", path)
        if b % 20 == 0:
            R.log(f"vr51 {family} rep {b} done")


def points():
    import vl_models as M
    out = {}
    for family in ("tabular", "image"):
        E = setup(family); tr, te = E["tr_all"], E["te_all"]
        rows = []
        for seed in range(3):
            r = {}
            for name, arrs, meth in (("M0", E["A"], "erm_y"), ("M2", E["A"], "erm_verified"), ("M2x", E["Ax"], "erm_verified"),
                                     ("M0m", E["Am"], "erm_y"), ("M2m", E["Am"], "erm_verified")):
                r[name], _ = fit_eval(M, E, arrs, meth, tr, te, seed)
            rows.append(r)
            R.log(f"points {family} seed {seed}: color M2 {r['M2']['color_variegation']['clip']:+.3f} M2x "
                  f"{r['M2x']['color_variegation']['clip']:+.3f} M0m {r['M0m']['color_variegation']['clip']:+.3f} "
                  f"M2m {r['M2m']['color_variegation']['clip']:+.3f}")
        out[family] = {k: {c: {s: float(np.mean([r[k][c][s] for r in rows])) for s in ("clip", "direct")} for c in CONC}
                       for k in rows[0]}
        os.makedirs(OUTD, exist_ok=True)
        json.dump(out, open(os.path.join(OUTD, "points.json"), "w"), indent=1)


def summarize():
    pct = lambda v, a: [float(np.percentile(v, 100 * a / 2)), float(np.percentile(v, 100 * (1 - a / 2)))]
    sep = lambda a, b: bool((a[0] > 0 and b[1] < 0) or (a[1] < 0 and b[0] > 0))
    rng = np.random.default_rng(510)
    out = {"families": {}}
    for fam in ("tabular", "image"):
        reps = {}
        for f in glob.glob(os.path.join(OUTD, f"{fam}_*.json")):
            reps.update(json.load(open(f))["reps"])
        bs = sorted({int(k.split("|")[0]) for k in reps if all(f"{k.split('|')[0]}|{o}" in reps for o in SEEDS)})
        v19 = {}
        for f in glob.glob(os.path.join(R.RES, "vr19", f"{fam}_*.json")):
            for k, v in json.load(open(f))["reps"].items():
                if int(k) >= 0:
                    v19[int(k)] = v["concepts"]
        v39 = {}
        for f in glob.glob(os.path.join(R.RES, "vr39", f"{fam}_*.json")):
            v39.update(json.load(open(f))["reps"])
        fo = {"B": len(bs), "concepts": {}}
        stored = lambda b, o, c, k: v19[b][c][k] if o == 0 else v39[f"{b}|{o}"][c][k]
        err = []
        for b in bs[:200]:
            for o in SEEDS:
                err.append(abs(reps[f"{b}|{o}"]["M2"]["color_variegation"]["clip"] - stored(b, o, "color_variegation", "d2")))
                if fam == "image":
                    err.append(abs(reps[f"{b}|{o}"]["M0"]["color_variegation"]["clip"] - stored(b, o, "color_variegation", "d0")))
        fo["max_reproduction_error"] = float(max(err)) if err else None
        for c in CONC:
            avg = lambda key, scale: np.array([np.mean([reps[f"{b}|{o}"][key][c][scale] for o in SEEDS]) for b in bs])
            d0_stored = np.array([np.mean([stored(b, o, c, "d0") for o in SEEDS]) for b in bs])
            versions = {"clipped": (d0_stored if fam == "tabular" else avg("M0", "clip"), avg("M2", "clip")),
                        "excluded": (d0_stored, avg("M2x", "clip"))}
            if fam == "image":
                versions["direct"] = (avg("M0", "direct"), avg("M2", "direct"))
            e = {}
            for name, (d0, d2) in versions.items():
                b0, b2 = pct(d0, A_B), pct(d2, A_B)
                lab = sep(b0, b2)
                rep = []
                for _ in range(2000):
                    ix = rng.integers(0, len(d0), len(d0))
                    rep.append(sep(pct(d0[ix], A_B), pct(d2[ix], A_B)) == lab)
                e[name] = {"d0_ci_bonf": b0, "d2_ci_bonf": b2, "d0_ci95": pct(d0, 0.05), "d2_ci95": pct(d2, 0.05),
                           "diff_ci95": pct(d0 - d2, 0.05), "bonf": lab, "robust_95": sep(pct(d0, 0.05), pct(d2, 0.05)),
                           "reproduced": float(np.mean(rep)), "n_same_sign": int(np.sum(np.sign(d0) == np.sign(d2)))}
            fo["concepts"][c] = e
            R.log(f"{fam} {c}: " + "; ".join(f"{k} bonf {v['bonf']} ({v['reproduced']:.2f}) d0 {np.round(v['d0_ci_bonf'], 2)} "
                                             f"d2 {np.round(v['d2_ci_bonf'], 2)}" for k, v in e.items()))
        out["families"][fam] = fo
    pp = os.path.join(OUTD, "points.json")
    out["points"] = json.load(open(pp)) if os.path.exists(pp) else None
    R.save_json(out, "vr51_direct_indeterminate.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family"); ap.add_argument("--start", type=int); ap.add_argument("--end", type=int)
    ap.add_argument("--points", action="store_true"); ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        summarize()
    elif a.points:
        points()
    else:
        run(a.family, a.start, a.end)
