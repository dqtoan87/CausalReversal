#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR13 — Phép thử đổi head (head swap), kiểm chứng claim định vị đảo dấu ở tầng quyết định.

Với mỗi seed s (0..4) và mỗi encoder fine-tune e ∈ {M0, M2} (vr4, cùng khởi tạo và lịch), đóng băng encoder,
trích đặc trưng áp chót, rồi học lại MỘT head tuyến tính dưới mỗi regime nhãn r ∈ {M0, M2}:
      Encoder M0 + Head M0,  Encoder M0 + Head M2,  Encoder M2 + Head M0,  Encoder M2 + Head M2.
Head học theo đúng công thức linear probe của vr4: BCE, AdamW lr 1e-3 one-cycle, 2,000 bước batch 64, tỉ lệ dương
tối thiểu 10%, dừng sớm theo AUROC val của regime. Đánh giá Δ tertile trên cùng tập probe đồng đều của test.
Nếu dấu đi theo regime của HEAD chứ không theo nguồn ENCODER, đảo dấu định vị ở tầng quyết định. Đây là bằng
chứng, không phải chứng minh, rằng representation không đảo dấu.

Out -> Result/vr13_headswap.json, Result/headswap/feat_{enc}_s{seed}.npy (đặc trưng train + val)
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn

import vr_common as R
import vr4_finetune as V4

OUT = os.path.join(R.RES, "headswap")
SEEDS = [0, 1, 2, 3, 4]
STEPS, BS = 2000, 64


def load_encoder(tag, dev):
    net = V4.build_model()
    sd = torch.load(os.path.join(V4.OUT, f"{tag}.pt"), map_location="cpu")
    net.load_state_dict({k: v.float() for k, v in sd.items()})
    return net.to(dev).eval().to(memory_format=torch.channels_last)


def train_head(Htr, ytr, Hva, yva, seed, dev):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    pf = ytr.mean()
    w = np.where(ytr == 1, V4.POS_FLOOR / ytr.sum(), (1 - V4.POS_FLOOR) / (len(ytr) - ytr.sum())) if pf < V4.POS_FLOOR \
        else np.ones(len(ytr))
    idx = torch.multinomial(torch.as_tensor(w / w.sum()), STEPS * BS, replacement=True, generator=g).view(STEPS, BS)
    Ht = torch.as_tensor(Htr, dtype=torch.float32, device=dev); yt = torch.as_tensor(ytr, dtype=torch.float32, device=dev)
    Hv = torch.as_tensor(Hva, dtype=torch.float32, device=dev)
    head = nn.Linear(Htr.shape[1], 1).to(dev)
    nn.init.zeros_(head.bias); nn.init.normal_(head.weight, std=0.01)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=STEPS, pct_start=0.1)
    lossf = nn.BCEWithLogitsLoss()
    best, state = -1, None
    for s in range(STEPS):
        b = idx[s].to(dev)
        loss = lossf(head(Ht[b]).squeeze(-1), yt[b])
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if (s + 1) % V4.EVAL_EVERY == 0:
            with torch.no_grad():
                auc = R.auroc(yva, head(Hv).squeeze(-1).cpu().numpy())
            if auc is not None and auc > best:
                best, state = auc, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(state)
    return head, best


def main():
    dev = torch.device("cuda")
    df, fold = R.load_isic()
    ids = df["isic_id"].to_numpy(); Y = df.Y.to_numpy(); S = df.S.to_numpy()
    tr = np.flatnonzero(fold == "train"); va = np.flatnonzero(fold == "val")
    rng = np.random.default_rng(0)
    va0 = np.concatenate([va[Y[va] == 1], rng.choice(va[Y[va] == 0], 10000, replace=False)])   # val de M0 (vr4, seed 0)
    va2 = va[S[va] == 1]
    vall = np.union1d(va0, va2)
    os.makedirs(OUT, exist_ok=True)
    runs = []
    for seed in SEEDS:
        for enc in ("M0", "M2"):
            tag = f"ft_{enc}_s{seed}"
            if not os.path.exists(os.path.join(V4.OUT, f"{tag}.pt")):
                R.log(f"{tag}: missing encoder, skip"); continue
            fpath = os.path.join(OUT, f"feat_{enc}_s{seed}.npz")
            if os.path.exists(fpath):
                z = np.load(fpath); Ftr, Fva = z["tr"], z["va"]
            else:
                net = load_encoder(tag, dev)
                Ftr = V4.run_eval(net, ids[tr], dev, 24, want_feats=True)["h"]
                Fva = V4.run_eval(net, ids[vall], dev, 24, want_feats=True)["h"]
                np.savez(fpath, tr=Ftr, va=Fva)
                del net; torch.cuda.empty_cache()
            pz = np.load(os.path.join(V4.OUT, f"{tag}.npz"))
            probe, uni = pz["probe"], pz["is_uniform"]
            Hp = pz["h"][uni].astype(np.float32)
            Tc = {c: R.SV.tertile(df, col).fillna(-1).to_numpy()[probe[uni]] for c, col in R.CONCEPTS.items()}
            # chuẩn hoá theo thống kê train của chính encoder này
            mu, sd = Ftr.astype(np.float32).mean(0), Ftr.astype(np.float32).std(0) + 1e-6
            nz = lambda A: (A.astype(np.float32) - mu) / sd
            pos = {p: i for i, p in enumerate(vall)}
            for head_reg in ("M0", "M2"):
                if head_reg == "M0":
                    tri, vai = np.arange(len(tr)), np.array([pos[p] for p in va0])
                    ytr, yva = Y[tr], Y[va0]
                else:
                    tri, vai = np.flatnonzero(S[tr] == 1), np.array([pos[p] for p in va2])
                    ytr, yva = Y[tr][tri], Y[va2]
                head, best = train_head(nz(Ftr[tri]), ytr, nz(Fva[vai]), yva, seed, dev)
                with torch.no_grad():
                    lg = head(torch.as_tensor(nz(Hp), device=dev)).squeeze(-1).cpu().numpy()
                r = {"seed": seed, "encoder": enc, "head": head_reg, "val_auroc": best,
                     "test_auroc_Y": R.auroc(Y[probe[uni]], lg),
                     "delta": {c: float(lg[t == 1].mean() - lg[t == 0].mean()) for c, t in Tc.items()}}
                runs.append(r)
                R.log(f"seed {seed} encoder {enc} head {head_reg}: AUROC_Y={r['test_auroc_Y']:.3f} "
                      f"Δ={ {k[:5]: round(v, 2) for k, v in r['delta'].items()} }")
            R.save_json({"runs": runs}, "vr13_headswap.json.partial")
    summ = {}
    for c in R.CONCEPTS:
        cell = {}
        for enc in ("M0", "M2"):
            for hr in ("M0", "M2"):
                v = [r["delta"][c] for r in runs if r["encoder"] == enc and r["head"] == hr]
                cell[f"enc{enc}_head{hr}"] = {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)),
                                              "n": len(v), "values": v}
        # dấu đi theo head: cùng head → cùng dấu qua hai encoder; khác head → khác dấu, ở mọi seed
        seeds = sorted({r["seed"] for r in runs})
        follows_head = []
        for s in seeds:
            d = {(r["encoder"], r["head"]): r["delta"][c] for r in runs if r["seed"] == s}
            if len(d) == 4:
                follows_head.append(bool(np.sign(d[("M0", "M0")]) == np.sign(d[("M2", "M0")]) and
                                         np.sign(d[("M0", "M2")]) == np.sign(d[("M2", "M2")]) and
                                         np.sign(d[("M0", "M0")]) != np.sign(d[("M0", "M2")])))
        cell["seeds_sign_follows_head"] = int(sum(follows_head)); cell["n_seeds"] = len(follows_head)
        summ[c] = cell
        R.log(f"{c}: sign follows head in {sum(follows_head)}/{len(follows_head)} seeds; "
              + ", ".join(f"{k}={v['mean']:+.2f}" for k, v in cell.items() if k.startswith("enc")))
    R.save_json({"runs": runs, "summary": summ, "seeds": SEEDS}, "vr13_headswap.json")


if __name__ == "__main__":
    main()
