#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VL0 — Trích embedding ảnh bằng encoder ĐÓNG BĂNG và concept proxy đo trực tiếp trên ảnh.

Vì sao đóng băng. Encoder là ResNet-50 ImageNet, chưa từng thấy nhãn da liễu nào. Mọi khác biệt về
concept sensitivity giữa các chế độ huấn luyện (M0/M1/M2, SVCL...) vì vậy chỉ đến từ head được huấn
luyện, không đến từ một encoder đã học quá trình xác minh. Checkpoint của v3 KHÔNG dùng được ở đây, vì
chúng đã được huấn luyện trên Y của ISIC-2024.

Concept proxy từ ảnh (cùng một định nghĩa trên mọi bộ dữ liệu, để so sánh được giữa cohort):
  mask      L* < ngưỡng Otsu, giữ thành phần liên thông chứa tâm ảnh (hoặc lớn nhất)
  size      tỉ lệ diện tích mask
  color_var sqrt(var a* + var b*) trong mask        (đối ứng tbp_lv_norm_color)
  contrast  L*(da) − L*(tổn thương)                  (đối ứng tbp_lv_deltaLBnorm)
  asym      1 − IoU(mask, mask lật quanh trục chính)
  border    chu vi² / (4π·diện tích)                  (đối ứng tbp_lv_norm_border)

Out -> Result/embeddings/{isic2024,isic2019,padufes}.npz  (emb float16 [n,2048], concepts, ids)
Chạy:  python3 vl0_embed.py [--datasets isic2024,isic2019,padufes] [--batch 256] [--workers 24]
"""
from __future__ import annotations

import argparse
import glob
import io
import os

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import vl_common as V

SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
CONCEPT_NAMES = ["size", "color_var", "contrast", "asym", "border"]


def image_concepts(rgb):
    """rgb uint8 [H,W,3] ở độ phân giải 224. Trả 5 concept proxy (NaN nếu mask suy biến).
    L* được làm mịn (σ=2) trước Otsu, mask được mở hình thái và lấp lỗ, để lông và kết cấu da không
    làm vỡ đường biên."""
    from scipy import ndimage as ndi
    from skimage import color, filters, measure, morphology
    lab = color.rgb2lab(rgb)
    L = lab[..., 0]
    Ls = ndi.gaussian_filter(L, 2)
    try:
        th = filters.threshold_otsu(Ls)
    except ValueError:
        return [np.nan] * 5
    m = morphology.binary_opening(Ls < th, morphology.disk(3))
    lb = measure.label(m)
    if lb.max() == 0:
        return [np.nan] * 5
    c = lb[lb.shape[0] // 2, lb.shape[1] // 2]
    if c == 0:
        c = np.argmax(np.bincount(lb.ravel())[1:]) + 1
    m = ndi.binary_fill_holes(lb == c)
    area = m.sum()
    if area < 30 or area > 0.98 * m.size:
        return [np.nan] * 5
    size = area / m.size
    cv = float(np.sqrt(lab[..., 1][m].var() + lab[..., 2][m].var()))
    contrast = float(L[~m].mean() - L[m].mean())
    props = measure.regionprops(m.astype(np.uint8))[0]
    # phản chiếu từng pixel qua trục chính: (u, v) -> (u, -v) trong hệ trục chính
    r, cc = np.nonzero(m)
    r0, c0 = props.centroid
    th_ = props.orientation                        # góc trục chính so với trục hàng
    dr, dc = r - r0, cc - c0
    u = dr * np.cos(th_) + dc * np.sin(th_)
    v = -dr * np.sin(th_) + dc * np.cos(th_)
    rr = np.rint(r0 + u * np.cos(th_) + v * np.sin(th_)).astype(int)   # v -> -v
    rc = np.rint(c0 + u * np.sin(th_) - v * np.cos(th_)).astype(int)
    ok = (rr >= 0) & (rr < m.shape[0]) & (rc >= 0) & (rc < m.shape[1])
    inside = np.zeros(len(r), bool); inside[ok] = m[rr[ok], rc[ok]]
    asym = 1 - inside.mean()
    border = props.perimeter ** 2 / (4 * np.pi * area)
    return [size, cv, contrast, asym, border]


class Imgs(Dataset):
    def __init__(self, kind, items):
        self.kind, self.items, self.h5 = kind, items, None

    def __len__(self):
        return len(self.items)

    def _load(self, i):
        it = self.items[i]
        if self.kind == "isic2024":
            import h5py
            if self.h5 is None:
                self.h5 = h5py.File(os.path.join(V.DATA, "data_ISIC2024/train-image.hdf5"), "r")
            return Image.open(io.BytesIO(self.h5[it][()])).convert("RGB")
        return Image.open(it).convert("RGB")

    def __getitem__(self, i):
        im = self._load(i).resize((SIZE, SIZE), Image.BILINEAR)
        rgb = np.asarray(im)
        try:
            cc = image_concepts(rgb)
        except Exception:
            cc = [np.nan] * 5
        x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1)), torch.tensor(cc, dtype=torch.float32)


def items_for(kind):
    if kind == "isic2024":
        meta = pd.read_csv(V.SV.DATA_CSV, usecols=["isic_id"], low_memory=False)
        ids = meta["isic_id"].tolist()
        return ids, ids
    if kind == "isic2019":
        gt = pd.read_csv(os.path.join(V.DATA, "data_ISIC_2019/ISIC_2019_Training_GroundTruth.csv"))
        ids = gt["image"].tolist()
        return ids, [os.path.join(V.DATA, "data_ISIC_2019/ISIC_2019_Training_Input", f"{i}.jpg") for i in ids]
    if kind == "padufes":
        meta = pd.read_csv(os.path.join(V.DATA, "PAD-UFES-20/metadata.csv"))
        paths = {os.path.basename(p): p for p in glob.glob(os.path.join(V.DATA, "PAD-UFES-20/imgs_part_*/**/*.png"),
                                                            recursive=True)}
        ids = meta["img_id"].tolist()
        missing = [i for i in ids if i not in paths]
        assert not missing, f"{len(missing)} PAD-UFES images missing, e.g. {missing[:3]}"
        return ids, [paths[i] for i in ids]
    raise ValueError(kind)


@torch.no_grad()
def run(kind, batch, workers, dev):
    out = os.path.join(V.EMB, f"{kind}.npz")
    if os.path.exists(out):
        V.log(f"{kind}: exists, skip"); return
    ids, items = items_for(kind)
    V.log(f"{kind}: {len(ids)} images")
    net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
                                      if os.path.exists(os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth"))
                                      else torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net = net.eval().to(dev).to(memory_format=torch.channels_last)
    dl = DataLoader(Imgs(kind, items), batch_size=batch, num_workers=workers, pin_memory=True)
    E = np.zeros((len(ids), 2048), np.float16); Cc = np.zeros((len(ids), 5), np.float32)
    k = 0
    for bi, (x, cc) in enumerate(dl):
        x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            e = net(x)
        n = len(x)
        E[k:k + n] = e.float().cpu().numpy().astype(np.float16); Cc[k:k + n] = cc.numpy(); k += n
        if bi % 100 == 0:
            V.log(f"  {kind}: {k}/{len(ids)}")
    os.makedirs(V.EMB, exist_ok=True)
    np.savez(out, emb=E, concepts=Cc, concept_names=np.array(CONCEPT_NAMES), ids=np.array(ids))
    V.log(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="padufes,isic2019,isic2024")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    torch.set_num_threads(16)
    for kind in a.datasets.split(","):
        run(kind, a.batch, a.workers, dev)


if __name__ == "__main__":
    main()
