#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vr_common — module dùng chung cho các script vr*.py.

Dùng các module cùng thư mục: nạp dữ liệu và biến giai đoạn (sv_common), chia bệnh nhân đóng băng, định lý và các
độ đo (vl_common), learner (vl_models). Đầu vào dùng chung (Result/vl_split.json, vl_nuisance*.npz, Result/embeddings)
chỉ được đọc; mọi đầu ra ghi vào Result/.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
RES = os.path.join(BASE, "Result")
sys.path.insert(0, HERE)

import sv_common as SV          # noqa: E402
import vl_common as VL          # noqa: E402

MISS_RES = VL.RES               # Result/: đầu vào dùng chung (chia tập đóng băng, nuisance, embedding)
EMB = VL.EMB
SEED = 42
log = SV.log
auroc, ece, coverage, worst_case_risk = VL.auroc, VL.ece, VL.coverage, VL.worst_case_risk
disease_interval, B_interval = VL.disease_interval, VL.B_interval
CONCEPTS = VL.CONCEPTS
# §13: proxy kích thước từ tile ảnh không hợp lệ (Spearman ≈ 0 với TBP), nên bị loại khỏi phân tích ảnh.
IMAGE_CONCEPTS = ["color_variegation", "lesion_skin_contrast", "asymmetry", "border_irregularity"]
S_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def save_json(obj, name):
    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, name)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=SV._np_default)
    os.replace(tmp, path)
    log(f"wrote {path}")
    return path


def load_isic():
    """ISIC-2024 với biến giai đoạn và fold đóng băng Result/vl_split.json (không tạo lại)."""
    df = SV.load()
    assert os.path.exists(os.path.join(MISS_RES, "vl_split.json")), "frozen split missing"
    fold = VL.make_split(df)
    return df, fold


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))
