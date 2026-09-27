#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR4 — Fine-tune M0 và M2 (proposal_v3 §11), thí nghiệm ưu tiên số 1.

  M0   X → encoder → head, trên nhãn ghi nhận Y, toàn bộ tổn thương train
  M2   X → encoder → head, trên Y, chỉ tổn thương train đã xác minh (S=1)

Giống hệt nhau giữa M0 và M2: khởi tạo (ResNet-50 ImageNet V2), kiến trúc, augmentation, optimizer, lịch
học, số bước, quy tắc lấy mẫu, chia bệnh nhân đóng băng, tập đánh giá. Chỉ tập huấn luyện khác nhau.

Hai chế độ:
  ft   fine-tune toàn bộ (backbone lr 1e-4, head lr 1e-3)
  lp   linear probe: backbone đóng băng (BN ở eval), chỉ head được học — ĐỐI CHỨNG: mọi đảo dấu ở đây
       chỉ có thể nằm ở head, nên so ft với lp tách được "representation reversal" khỏi "head reversal".

Augmentation chỉ hình học (lật, xoay 90°, crop). Không color jitter: nó sẽ phá chính concept màu.
Lấy mẫu: có hoàn lại, tỉ lệ dương tối thiểu 10% (M2 vốn ~37% nên không đổi; M0 được nâng). Cùng quy tắc
cho cả hai regime. Dừng sớm theo AUROC trên val của chính regime (M0: val toàn bộ; M2: val đã xác minh).

Sau huấn luyện, trên TẬP PROBE CHUNG (test: 30,000 tổn thương ngẫu nhiên cố định ∪ mọi tổn thương test F=1):
  lưu logit, đặc trưng áp chót h (2048) và layer3 gộp không gian (1024);
  can thiệp biểu diễn ở layer3 (§12.2): hướng concept v_C = ridge(layer3 gộp → concept TBP chuẩn hoá),
  cộng ±1 SD của hình chiếu vào MỌI vị trí không gian, truyền qua layer4 + pool + fc, Δ = logit(+) − logit(−),
  trên 3,000 tổn thương ngẫu nhiên của phần đồng đều.

Out -> Result/finetune/{mode}_{regime}_s{seed}.npz, .pt ; Result/vr4_finetune.json (tổng hợp nhẹ)
Chạy:  python3 vr4_finetune.py [--modes ft,lp] [--regimes M0,M2] [--seeds 0,1,2]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import vr_common as R

H5 = os.path.join(R.VL.DATA, "data_ISIC2024/train-image.hdf5")
OUT = os.path.join(R.RES, "finetune")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
RESIZE, CROP, BS = 144, 128, 64
STEPS = {"ft": 4000, "lp": 2000}
EVAL_EVERY = 500
POS_FLOOR = 0.10
N_PROBE_UNIFORM, N_INTERV = 30000, 3000


class Tiles(Dataset):
    def __init__(self, ids, train):
        self.ids, self.train, self.h5 = ids, train, None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        if self.h5 is None:
            self.h5 = h5py.File(H5, "r")
        im = Image.open(io.BytesIO(self.h5[self.ids[i]][()])).convert("RGB").resize((RESIZE, RESIZE), Image.BILINEAR)
        x = np.asarray(im, np.float32) / 255.0
        if self.train:
            r = np.random.randint(0, RESIZE - CROP + 1, 2)
            x = x[r[0]:r[0] + CROP, r[1]:r[1] + CROP]
            if np.random.rand() < 0.5:
                x = x[:, ::-1]
            if np.random.rand() < 0.5:
                x = x[::-1]
            x = np.rot90(x, np.random.randint(4))
        else:
            o = (RESIZE - CROP) // 2
            x = x[o:o + CROP, o:o + CROP]
        x = (x - MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), i


def build_model():
    net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    net.fc = nn.Linear(2048, 1)
    nn.init.zeros_(net.fc.bias); nn.init.normal_(net.fc.weight, std=0.01)
    return net


def trunk_to_layer3(net, x):
    x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
    return net.layer3(net.layer2(net.layer1(x)))


def head_from_layer3(net, a3):
    h = torch.flatten(net.avgpool(net.layer4(a3)), 1)
    return h, net.fc(h).squeeze(-1)


@torch.no_grad()
def run_eval(net, ids, dev, workers, want_feats=False):
    net.eval()
    dl = DataLoader(Tiles(ids, False), batch_size=256, num_workers=workers, pin_memory=True)
    lg, H, A3 = [], [], []
    for x, _ in dl:
        x = x.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            a3 = trunk_to_layer3(net, x)
            h, l = head_from_layer3(net, a3)
        lg.append(l.float().cpu().numpy())
        if want_feats:
            H.append(h.half().cpu().numpy()); A3.append(a3.float().mean((2, 3)).half().cpu().numpy())
    out = {"logit": np.concatenate(lg)}
    if want_feats:
        out["h"] = np.concatenate(H); out["a3"] = np.concatenate(A3)
    return out


@torch.no_grad()
def layer3_intervention(net, ids, a3_pooled_probe, concept_probe, a3_pooled_sub, dev, workers):
    """Δ logit khi dịch layer3 theo hướng concept ±1 SD (tại mọi vị trí không gian)."""
    from sklearn.linear_model import Ridge
    ok = np.isfinite(concept_probe)
    A = a3_pooled_probe[ok].astype(np.float32)
    mu, sd = A.mean(0), A.std(0) + 1e-6
    c = (concept_probe[ok] - concept_probe[ok].mean()) / (concept_probe[ok].std() + 1e-9)
    w = Ridge(alpha=10.0).fit((A - mu) / sd, c).coef_ / sd          # hướng trong không gian layer3 gốc
    v = w / (np.linalg.norm(w) + 1e-12)
    step = float(np.std(A @ v))                                     # 1 SD của hình chiếu
    vt = torch.tensor(v * step, dtype=torch.float32, device=dev)[None, :, None, None]
    dl = DataLoader(Tiles(ids, False), batch_size=128, num_workers=workers, pin_memory=True)
    net.eval()
    d = []
    for x, _ in dl:
        x = x.to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            a3 = trunk_to_layer3(net, x).float()
        _, lp = head_from_layer3(net, a3 + vt)
        _, lm = head_from_layer3(net, a3 - vt)
        d.append((lp - lm).float().cpu().numpy() / 2)
    d = np.concatenate(d)
    return {"delta_mean": float(d.mean()), "delta_se": float(d.std() / np.sqrt(len(d))),
            "frac_positive": float((d > 0).mean()), "step_sd": step}


def train_one(mode, regime, seed, df, fold, workers, dev):
    tag = f"{mode}_{regime}_s{seed}"
    path = os.path.join(OUT, f"{tag}.npz")
    if os.path.exists(path):
        R.log(f"{tag}: exists, skip"); return
    torch.manual_seed(seed); np.random.seed(seed)
    ids_all = df["isic_id"].to_numpy()
    Y = df.Y.to_numpy(); S = df.S.to_numpy(); F = df.F.to_numpy()
    tr = np.flatnonzero(fold == "train"); va = np.flatnonzero(fold == "val")
    if regime == "M2":
        tr, va = tr[S[tr] == 1], va[S[va] == 1]
    else:
        rng = np.random.default_rng(seed)
        vneg = va[Y[va] == 0]
        va = np.concatenate([va[Y[va] == 1], rng.choice(vneg, 10000, replace=False)])
    ytr = Y[tr]
    pfrac = ytr.mean()
    if pfrac < POS_FLOOR:
        w = np.where(ytr == 1, POS_FLOOR / ytr.sum(), (1 - POS_FLOOR) / (len(ytr) - ytr.sum()))
    else:
        w = np.ones(len(ytr))
    steps = STEPS[mode]
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), steps * BS, replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    dl = DataLoader(Tiles(ids_all[tr], True), batch_size=BS, sampler=sampler, num_workers=workers,
                    pin_memory=True, drop_last=True, persistent_workers=True)
    net = build_model().to(dev).to(memory_format=torch.channels_last)
    if mode == "lp":
        for p in net.parameters():
            p.requires_grad_(False)
        for p in net.fc.parameters():
            p.requires_grad_(True)
        groups = [{"params": net.fc.parameters(), "lr": 1e-3}]
    else:
        groups = [{"params": [p for n, p in net.named_parameters() if not n.startswith("fc.")], "lr": 1e-4},
                  {"params": net.fc.parameters(), "lr": 1e-3}]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps,
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler()
    lossf = nn.BCEWithLogitsLoss()
    Yt = torch.as_tensor(ytr, dtype=torch.float32)
    best, best_state, hist = -1, None, []
    t0 = time.time()
    R.log(f"{tag}: n_train={len(tr)} pos={int(ytr.sum())} natural_pos_frac={pfrac:.4f} n_val={len(va)}")
    for step, (x, i) in enumerate(dl, 1):
        net.train()
        if mode == "lp":
            net.eval()                                           # BN statistics frozen
        x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
        y = Yt[i].to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = lossf(net(x).squeeze(-1), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if step % EVAL_EVERY == 0 or step == steps:
            ev = run_eval(net, ids_all[va], dev, workers)
            auc = R.auroc(Y[va], ev["logit"])
            hist.append({"step": step, "loss": float(loss.detach()), "val_auroc": auc})
            R.log(f"  {tag} step {step}/{steps} loss={float(loss):.4f} val AUROC={auc:.4f} ({time.time() - t0:.0f}s)")
            if auc is not None and auc > best:
                best = auc
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    os.makedirs(OUT, exist_ok=True)
    torch.save({k: v.half() for k, v in best_state.items()}, os.path.join(OUT, f"{tag}.pt"))

    # tập probe chung, cố định bằng SEED toàn cục (giống nhau cho mọi run)
    te = np.flatnonzero(fold == "test")
    rng = np.random.default_rng(R.SEED)
    uni = np.sort(rng.choice(te, N_PROBE_UNIFORM, replace=False))
    probe = np.union1d(uni, te[F[te] == 1])
    ev = run_eval(net, ids_all[probe], dev, workers, want_feats=True)
    is_uni = np.isin(probe, uni)
    sub = np.sort(rng.choice(np.flatnonzero(is_uni), N_INTERV, replace=False))
    interv = {}
    for c in R.IMAGE_CONCEPTS:
        conc = df[R.CONCEPTS[c]].to_numpy(float)[probe]
        interv[c] = layer3_intervention(net, ids_all[probe[sub]], ev["a3"], conc, None, dev, workers)
        R.log(f"  {tag} layer3 intervention {c}: Δ={interv[c]['delta_mean']:+.4f} ± {interv[c]['delta_se']:.4f}")
    test_auroc_Y = R.auroc(Y[probe][is_uni], ev["logit"][is_uni])
    verified = S[probe] == 1
    test_auroc_Dver = R.auroc(Y[probe][verified], ev["logit"][verified])
    np.savez(path, probe=probe, is_uniform=is_uni, logit=ev["logit"], h=ev["h"], a3=ev["a3"],
             fc_w=net.fc.weight.detach().float().cpu().numpy().ravel(), fc_b=float(net.fc.bias.item()),
             interv=json.dumps(interv), hist=json.dumps(hist),
             meta=json.dumps({"mode": mode, "regime": regime, "seed": seed, "best_val_auroc": best,
                              "test_auroc_Y_uniform": test_auroc_Y, "test_auroc_D_verified": test_auroc_Dver,
                              "n_train": int(len(tr)), "minutes": (time.time() - t0) / 60}))
    R.log(f"{tag}: done best val={best:.4f} test AUROC_Y={test_auroc_Y} AUROC_Dver={test_auroc_Dver} "
          f"({(time.time() - t0) / 60:.1f} min)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="ft,lp")
    ap.add_argument("--regimes", default="M0,M2")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    df, fold = R.load_isic()
    for seed in [int(s) for s in a.seeds.split(",")]:
        for mode in a.modes.split(","):
            for regime in a.regimes.split(","):
                train_one(mode, regime, seed, df, fold, a.workers, dev)
    summ = []
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".npz"):
            z = np.load(os.path.join(OUT, f))
            summ.append({**json.loads(str(z["meta"])), "interv": json.loads(str(z["interv"]))})
    R.save_json({"runs": summ}, "vr4_finetune.json")


if __name__ == "__main__":
    main()
