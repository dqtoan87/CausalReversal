#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR50 — Direct network logits against the logit of the clipped predicted probability.

The learned contrasts of the paper use logit(clip(sigmoid(logit_D), 1e-7, 1 − 1e-7)), which caps the logit at ±16.12. This
script refits M0 and M2 with seeds 0 to 2 for both frozen-feature families (same settings as the original fits) and computes,
on the test population, the share of lesions at the cap and the tertile contrast of all five concepts under both measures:
  clipped  logit of the clipped probability (the measure of the paper)
  direct   the pre-sigmoid output logit_D
Out -> Result/vr50_logit_clip.json
"""
from __future__ import annotations

import numpy as np
import torch

import vr_common as R
import gpu_setup  # noqa: F401
from vr19_primary_bootstrap import features, arrays

CONC = ("color_variegation", "size", "lesion_skin_contrast", "asymmetry", "border_irregularity")


@torch.no_grad()
def direct_logit(model, x, bs=65536):
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(x), bs):
        out.append(model(torch.as_tensor(x[i:i + bs], dtype=torch.float32, device=dev))["logit_D"].cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def main():
    import vl_models as M
    df, fold = R.load_isic()
    tr, va, te = (np.flatnonzero(fold == k) for k in ("train", "val", "test"))
    T = {c: R.SV.tertile(df, R.CONCEPTS[c]).to_numpy()[te] for c in CONC}
    cap = float(R.logit(np.array([0.0]))[0])
    out = {"cap": cap, "families": {}}
    for fam in ("tabular", "image"):
        X, hid = features(df, fam); A = arrays(df, X)
        res = {}
        for reg, meth in (("M0", "erm_y"), ("M2", "erm_verified")):
            rows = []
            for seed in range(3):
                m = M.train(meth, A, tr, va, hid=hid, seed=seed)
                ld = direct_logit(m, X[te])
                lc = R.logit(M.predict(m, X[te])["pD"].astype(np.float64))
                r = {"share_at_cap": float(np.mean(lc <= cap + 1e-6) + np.mean(lc >= -cap - 1e-6)),
                     "min_direct": float(ld.min()), "contrasts": {}}
                for c in CONC:
                    t = T[c]
                    r["contrasts"][c] = {"clipped": float(lc[t == 1].mean() - lc[t == 0].mean()),
                                         "direct": float(ld[t == 1].mean() - ld[t == 0].mean()),
                                         "share_at_cap_lower": float(np.mean(lc[t == 0] <= cap + 1e-6)),
                                         "share_at_cap_upper": float(np.mean(lc[t == 1] <= cap + 1e-6))}
                rows.append(r)
                R.log(f"{fam} {reg} seed {seed}: at cap {r['share_at_cap']:.4f}, min direct {r['min_direct']:.1f}, color "
                      f"clipped {r['contrasts']['color_variegation']['clipped']:+.3f} direct {r['contrasts']['color_variegation']['direct']:+.3f}")
            res[reg] = {"share_at_cap": float(np.mean([r["share_at_cap"] for r in rows])),
                        "min_direct": float(min(r["min_direct"] for r in rows)),
                        "contrasts": {c: {k: float(np.mean([r["contrasts"][c][k] for r in rows]))
                                          for k in ("clipped", "direct", "share_at_cap_lower", "share_at_cap_upper")} for c in CONC}}
        out["families"][fam] = res
        R.save_json(out, "vr50_logit_clip.json")


if __name__ == "__main__":
    main()
