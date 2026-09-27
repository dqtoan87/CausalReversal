#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR45 — Tỉ lệ tổn thương test giữ lại ở mỗi ngưỡng hỗ trợ của σ̂ chung (GBM bảng), để bài báo dựng lại được từ JSON
mà không cần vl_nuisance.npz (không phân phối kèm repo). Ngưỡng lấy từ vr19 (support_thresholds).

Out -> Result/vr45_support_retention.json
"""
from __future__ import annotations

import json
import os

import numpy as np

import vr_common as R


def main():
    nz = np.load(os.path.join(R.MISS_RES, "vl_nuisance.npz"), allow_pickle=True)
    sig, fold = nz["sigma"], nz["fold"]
    thr = json.load(open(os.path.join(R.RES, "vr19_primary_bootstrap.json")))["families"]["tabular"]["support_thresholds"]
    te = fold == "test"
    R.save_json({"thresholds": thr, "retained": {q: float((sig[te] >= v).mean()) for q, v in thr.items()}},
                "vr45_support_retention.json")


if __name__ == "__main__":
    main()
