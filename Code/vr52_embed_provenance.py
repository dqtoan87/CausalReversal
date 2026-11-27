#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR52 — Which ResNet-50 weights produced the stored image embeddings.

The stored files Result/embeddings/{isic2024,padufes}.npz carry no record of the torchvision weights. This script recomputes
the pooled features of a fixed sample of images (256 ISIC-2024 crops, 64 PAD-UFES-20 images; rng(52)) with the pipeline of
vl0_embed.py under both IMAGENET1K_V1 and IMAGENET1K_V2, and compares them with the stored half-precision features
(cosine similarity and maximum absolute difference per image). The stored concept proxies are compared as well.
Out -> Result/vr52_embed_provenance.json
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torchvision

import vl_common as V
import vr_common as R
from vl0_embed import Imgs, items_for

SAMPLE = {"isic2024": 256, "padufes": 64}


@torch.no_grad()
def features(net, x, dev):
    x = x.to(dev).to(memory_format=torch.channels_last)
    with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
        return net(x).float().cpu().numpy().astype(np.float16).astype(np.float32)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets = {}
    for name in ("IMAGENET1K_V1", "IMAGENET1K_V2"):
        net = torchvision.models.resnet50(weights=getattr(torchvision.models.ResNet50_Weights, name))
        net.fc = torch.nn.Identity()
        nets[name] = net.eval().to(dev).to(memory_format=torch.channels_last)
    rng = np.random.default_rng(52)
    out = {"torchvision": torchvision.__version__, "torch": torch.__version__, "device": str(dev), "datasets": {}}
    for kind, n in SAMPLE.items():
        z = np.load(os.path.join(V.EMB, f"{kind}.npz"))
        E, C, ids = z["emb"], z["concepts"], z["ids"]
        sel = np.sort(rng.choice(len(ids), n, replace=False))
        all_ids, items = items_for(kind)
        assert [str(i) for i in all_ids] == [str(i) for i in ids], f"{kind}: stored ids differ from the archive order"
        ds = Imgs(kind, [items[i] for i in sel])
        xs, cs = zip(*[ds[i] for i in range(len(sel))])
        x = torch.stack(xs); cc = torch.stack(cs).numpy()
        st = E[sel].astype(np.float32)
        res = {"n": int(n), "concepts_max_abs_diff": float(np.nanmax(np.abs(cc - C[sel])))}
        for name, net in nets.items():
            f = np.concatenate([features(net, x[i:i + 64], dev) for i in range(0, len(x), 64)])
            cos = (f * st).sum(1) / (np.linalg.norm(f, axis=1) * np.linalg.norm(st, axis=1))
            res[name] = {"cosine_min": float(cos.min()), "cosine_median": float(np.median(cos)),
                         "max_abs_diff": float(np.abs(f - st).max()), "median_abs_diff": float(np.median(np.abs(f - st)))}
            R.log(f"{kind} {name}: cosine min {cos.min():.6f}, max |diff| {np.abs(f - st).max():.4f}")
        res["match"] = max(("IMAGENET1K_V1", "IMAGENET1K_V2"), key=lambda k: res[k]["cosine_min"])
        out["datasets"][kind] = res
    R.save_json(out, "vr52_embed_provenance.json")


def compare(path, note):
    """Compare a complete rebuild of isic2024.npz (for example from a clean copy of the release) with the stored file.
    Adds key "full_rebuild" to the output."""
    import json
    a = np.load(os.path.join(V.EMB, "isic2024.npz")); b = np.load(path)
    assert [str(i) for i in a["ids"]] == [str(i) for i in b["ids"]]
    cmin, dmax, n = 1.0, 0.0, len(a["ids"])
    for i in range(0, n, 50_000):
        x = a["emb"][i:i + 50_000].astype(np.float32); y = b["emb"][i:i + 50_000].astype(np.float32)
        cos = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
        cmin = min(cmin, float(cos.min())); dmax = max(dmax, float(np.abs(x - y).max()))
    cc = np.nanmax(np.abs(a["concepts"] - b["concepts"]))
    res = {"n": int(n), "weights_recorded": str(b["weights"]) if "weights" in b.files else None, "cosine_min": cmin,
           "max_abs_diff": dmax, "concepts_max_abs_diff": float(cc), "note": note}
    R.log(f"full rebuild: {res}")
    fp = os.path.join(R.RES, "vr52_embed_provenance.json"); out = json.load(open(fp)); out["full_rebuild"] = res
    R.save_json(out, "vr52_embed_provenance.json")


if __name__ == "__main__":
    import sys
    if "--compare" in sys.argv:
        compare(sys.argv[sys.argv.index("--compare") + 1], " ".join(sys.argv[sys.argv.index("--compare") + 2:]))
    else:
        main()
