#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR32 — Liều chọn mẫu Poisson (như vr26) với 20 lần lặp và hai đặc tả nuisance cho dự đoán theo input của learner ảnh.

Cho mỗi họ chính: ba concept chính với n = 320 nhãn âm, và color variegation với n = 1,000 và 3,000.
Dự đoán độ dốc:
  tabular  chính xác: −(tương phản tertile của log π_η(z_k)) trên test;
  image    bậc một, m_k(x̃) = E[z_k | x̃, Y=0] ước lượng bằng (a) ridge và (b) MLP cùng không gian đặc trưng, cả hai khớp
           lại trên nhãn âm của bệnh nhân train lấy lại ở mỗi lần lặp.
Lưu độ dốc từng lần lặp (quan sát và dự đoán) để tính khoảng của tỉ số quan sát/dự đoán bằng bootstrap qua lần lặp, và
độ dốc tràn sang của năm concept (quan sát và dự đoán) để so độ lớn.

Out -> Result/vr32_dose_nuisance.json
"""
from __future__ import annotations

import numpy as np

import vr_common as R
import gpu_setup  # noqa: F401
from vr21_dose_calibrated import features, _fit, N_MAL
from vr26_dose_poisson import poisson_probs, ETAS

N_REP = 20
CELLS = [("color_variegation", 320), ("size", 320), ("lesion_skin_contrast", 320), ("color_variegation", 1000), ("color_variegation", 3000)]


def mlp_regress(X, y, tr, va, seed):
    import torch
    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.as_tensor(X[tr]); yt = torch.as_tensor(y[tr], dtype=torch.float32)          # giữ trên CPU, chuyển từng batch
    Xv = torch.as_tensor(X[va]); yv = torch.as_tensor(y[va], dtype=torch.float32)
    best, state, bad = np.inf, None, 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(40):
        net.train()
        perm = torch.randperm(len(tr), generator=g)
        for i in range(0, len(tr), 2048):
            j = perm[i:i + 2048]
            loss = torch.mean((net(Xt[j].to(dev)).squeeze(1) - yt[j].to(dev)) ** 2); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(sum(torch.sum((net(Xv[i:i + 8192].to(dev)).squeeze(1) - yv[i:i + 8192].to(dev)) ** 2) for i in range(0, len(va), 8192)) / len(va))
        if vl < best - 1e-5:
            best, state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 4:
                break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        return np.concatenate([net(torch.as_tensor(X[i:i + 65536], device=dev)).squeeze(1).cpu().numpy() for i in range(0, len(X), 65536)])


def main():
    import vl_models as M
    from sklearn.linear_model import Ridge
    df, fold = R.load_isic()
    Y = df.Y.to_numpy()
    Zs = {c: ((df[col] - df[col].mean()) / df[col].std()).fillna(0.0).to_numpy() for c, col in R.CONCEPTS.items()}
    te = np.flatnonzero(fold == "test")
    Tc = {c: R.SV.tertile(df, col).fillna(-1).to_numpy()[te] for c, col in R.CONCEPTS.items()}
    D = lambda v, j: float(v[Tc[j] == 1].mean() - v[Tc[j] == 0].mean())
    pools = {p: {"mal": np.flatnonzero((fold == p) & (Y == 1)), "neg": np.flatnonzero((fold == p) & (Y == 0))} for p in ("train", "val")}
    nva_m = int(round(0.8 * len(pools["val"]["mal"])))
    tr_all, va_all = np.flatnonzero(fold == "train"), np.flatnonzero(fold == "val")
    pid = df[R.SV.GROUP].to_numpy()
    neg = pools["train"]["neg"]; neg_p = np.unique(pid[neg]); neg_by = {p: neg[pid[neg] == p] for p in neg_p}
    vneg = pools["val"]["neg"]
    import json, os
    prev = os.path.join(R.RES, "vr32_dose_nuisance.json")
    out = json.load(open(prev)) if os.path.exists(prev) else {"etas": ETAS, "n_rep": N_REP, "families": {}}
    e = np.array(ETAS)
    for kind in ("tabular", "image"):
        X, hid = features(df, kind)
        fam = out["families"].get(kind, {})
        for k, n_ben in CELLS:
            if f"{k}|{n_ben}" in fam:
                continue
            reps = []
            for r in range(N_REP):
                rng = np.random.default_rng(500 + 1000 * r + 7 * CELLS.index((k, n_ben)))
                mtr = rng.choice(pools["train"]["mal"], N_MAL, replace=False); mva = rng.choice(pools["val"]["mal"], nva_m, replace=False)
                nva_b = int(round(111 * n_ben / 320))
                pred = {}
                if kind == "image":
                    ix = np.concatenate([neg_by[p] for p in rng.choice(neg_p, len(neg_p))])
                    ix = rng.choice(ix, min(60000, len(ix)), replace=False)
                    m_r = Ridge(alpha=10.0).fit(X[ix], Zs[k][ix]).predict(X[te])
                    vsub = rng.choice(vneg, 20000, replace=False)
                    m_m = mlp_regress(X, Zs[k].astype(np.float32), ix, vsub, r)[te]
                    pred["ridge"] = {j: -D(m_r, j) for j in R.CONCEPTS}; pred["mlp"] = {j: -D(m_m, j) for j in R.CONCEPTS}
                obs = []
                exact = []
                for eta in ETAS:
                    ptr, kap = poisson_probs(np.exp(eta * Zs[k][pools["train"]["neg"]]), n_ben)
                    pva, _ = poisson_probs(np.exp(eta * Zs[k][pools["val"]["neg"]]), nva_b)
                    btr = pools["train"]["neg"][rng.random(len(ptr)) < ptr]; bva = pools["val"]["neg"][rng.random(len(pva)) < pva]
                    _, cal = _fit(M, df, X, hid, np.concatenate([mtr, btr]), np.concatenate([mva, bva]), tr_all, va_all, te, r)
                    obs.append({j: D(cal, j) for j in R.CONCEPTS})
                    lpi = np.log(np.minimum(1.0, kap * np.exp(eta * Zs[k][te])))
                    exact.append({j: -D(lpi, j) for j in R.CONCEPTS})
                rep = {"obs": {j: float(np.polyfit(e, [o[j] for o in obs], 1)[0]) for j in R.CONCEPTS},
                       "mean_by_eta": [o[k] for o in obs]}
                if kind == "tabular":
                    rep["pred"] = {"exact": {j: float(np.polyfit(e, [x[j] for x in exact], 1)[0]) for j in R.CONCEPTS}}
                else:
                    rep["pred"] = pred
                reps.append(rep)
            key = f"{k}|{n_ben}"
            specs = list(reps[0]["pred"])
            so = np.array([rp["obs"][k] for rp in reps])
            cell = {"obs_slope": float(so.mean()), "obs_range95": np.percentile(so, [2.5, 97.5]).tolist(), "per_rep": reps, "pred": {}}
            for sp in specs:
                pr = np.array([rp["pred"][sp][k] for rp in reps])
                boot = []
                for _ in range(2000):
                    ii = rng.integers(0, N_REP, N_REP); boot.append(so[ii].mean() / pr[ii].mean())
                cell["pred"][sp] = {"slope": float(pr.mean()), "range95": np.percentile(pr, [2.5, 97.5]).tolist(),
                                    "ratio": float(so.mean() / pr.mean()), "ratio_ci95": np.percentile(boot, [2.5, 97.5]).tolist(),
                                    "spill": {j: [float(np.mean([rp["obs"][j] for rp in reps])), float(np.mean([rp["pred"][sp][j] for rp in reps]))] for j in R.CONCEPTS}}
            fam[key] = cell
            R.log(f"{kind} {key}: obs {cell['obs_slope']:+.2f} " + " ".join(f"{sp} pred {cell['pred'][sp]['slope']:+.2f} ratio {cell['pred'][sp]['ratio']:.2f} "
                                                                     f"{np.round(cell['pred'][sp]['ratio_ci95'], 2)}" for sp in specs))
            out["families"][kind] = fam
            R.save_json(out, "vr32_dose_nuisance.json")


if __name__ == "__main__":
    main()
