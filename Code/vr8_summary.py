#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR8 — Tự sinh Result/vr_summary.md từ các JSON của vr1–vr6 (chỉ số liệu, không diễn giải).
Phần diễn giải theo RQ1–RQ4 và quyết định định vị (§23) được viết riêng sau khi đọc các số này.
"""
from __future__ import annotations

import json
import os

import numpy as np

import vr_common as R


def J(n):
    p = os.path.join(R.RES, n)
    return json.load(open(p)) if os.path.exists(p) else None


def f(x, d=3):
    return "n/a" if x is None else (f"{x:.{d}f}" if isinstance(x, (float, int, np.floating)) else str(x))


def main():
    L = ["# VILRR: tóm tắt số liệu tự động", "", "Sinh bởi `Code/vr8_summary.py` từ các JSON trong thư mục này.", ""]
    t = J("vr1_reversal_theory.json")
    if t:
        a, b = t["theorem4"], t["theorem5_population"]["summary"]
        L += ["## Định lý 4 và 5 (vr1)", "",
              f"- T4 trên {a['n_models']} mô hình ngẫu nhiên: sai số đồng nhất thức tối đa {a['max_abs_identity_error']:.1e}, "
              f"khớp phân loại {a['category_agreement']:.3f}.",
              f"- T5, LR log-tuyến tính: sai số hệ số M2 tối đa {b['loglinear_max_abs_err_M2']:.1e}, khớp đảo dấu "
              f"{b['loglinear_reversal_agreement']:.3f}. Chọn lọc có sàn s ≥ 0.5: sai số tối đa {b['floored_max_abs_err_M2']:.3f}, "
              f"khớp {b['floored_reversal_agreement']:.3f}. Chênh tối đa giữa M0 − M2 và độ dốc V: {b['M0_minus_M2_vs_slopeV_max_abs_diff']:.3f}.",
              "", "| Concept (ISIC-2024) | độ dốc M0 | độ dốc M2 | độ dốc V | phần dư M0 − M2 − V (SE) |", "| --- | ---: | ---: | ---: | ---: |"]
        for c, v in t["isic2024_slopes"]["single"].items():
            L.append(f"| {c} | {v['slope_M0']:+.3f} | {v['slope_M2']:+.3f} | {v['slope_V']:+.3f} | "
                     f"{v['identity_residual']:+.3f} ({v['residual_se']:.3f}) |")
        L.append("")
    p = J("vr2_phase_diagram.json")
    if p:
        s = p["summary"]
        L += ["## Phase diagram (vr2)", "",
              f"- {s['n_points']} điểm, loại: {s['category_counts']}; s ác tính nhỏ nhất {s['min_s_malignant']:.3f}.",
              f"- Đảo dấu hệ số M2 khớp Định lý 4 mức tertile: {s['M2_coef_vs_tertile_theorem4_agreement']:.3f}; "
              f"dự đoán chỉ từ quan sát (β_M0 − β_V) khớp dấu M2: {s['observable_prediction_agreement']:.3f}.",
              f"- MLP hữu hạn mẫu (24 điểm): dấu M2 khớp Định lý 4: {f(s.get('mlp_M2_sign_agrees_with_theorem4'))}.",
              f"- Tỉ lệ đảo dấu theo b': {s['reversal_frac_by_bprime']}; theo β: {s['reversal_frac_by_beta']}.", ""]
    d = J("vr3_icdl.json")
    if d and "sim_rows" in d:
        L += ["## ICDL trên lưới s_true × s_assumed (vr3, icdl_kl)", "",
              "| s_true | s_assumed | \\|log err\\| | p thật ∈ 𝓘 (q̂) | p thật ∈ 𝓘 (oracle) | p̂ ∈ 𝓘 oracle | AUROC_D | sai dấu |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for r in d["sim_rows"]:
            if r["method"] == "icdl_kl":
                L.append(f"| {r['s_true']} | {r['s_assumed']} | {r['abs_log_err']:.3f} | {r['true_p_in_interval']:.3f} | "
                         f"{r['true_p_in_oracle_interval']:.3f} | {r['pred_in_oracle_interval']:.3f} | {r['auroc_D']:.3f} | {r['n_sign_wrong']} |")
        L += ["", "| s_true | phương pháp | \\|log err\\| | AUROC_D | sai dấu |", "| ---: | --- | ---: | ---: | ---: |"]
        for r in d["sim_rows"]:
            if r["method"] != "icdl_kl" and (r["s_assumed"] is None or r["s_assumed"] == 0.5):
                L.append(f"| {r['s_true']} | {r['method']}{'' if r['s_assumed'] is None else ' @0.5'} | "
                         f"{r['abs_log_err']:.3f} | {r['auroc_D']:.3f} | {r['n_sign_wrong']} |")
        L += ["", "| s_true | dải theo giả định phủ p thật | tỉ số trung vị max/min | τ=0.01: quyết định bất biến | đúng oracle (bất biến) | lỗi @ s đúng | lỗi ERM-Y | lỗi M2 |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for b in d["sim_bands"]:
            q = b["decisions"]["0.01"]
            L.append(f"| {b['s_true']} | {b['band_covers_true_p']:.3f} | {b['median_band_ratio']:.2f} | {q['frac_assumption_free']:.3f} | "
                     f"{f(q['assumption_free_agree_oracle'])} | {q['error_at_true_s']:.4f} | {q['error_erm_y']:.4f} | {q['error_erm_verified']:.4f} |")
        cov = {}
        for r in d["sim_structured"]:
            if "concept" in r:
                for h in ("rect", "strata", "shared"):
                    cov.setdefault((h, r["s_assumed"] == r["s_true"]), []).append(r[f"{h}_covers"])
        L += ["", "ℋ có cấu trúc, độ phủ RR_D thật (s_assumed = s_true / s_assumed = 0.5): " +
              "; ".join(f"{h}: {np.mean(cov.get((h, True), [np.nan])):.2f} / {np.mean(cov.get((h, False), [np.nan])):.2f}"
                        for h in ("rect", "strata", "shared")), ""]
    if d and "isic" in d:
        for kind, v in d["isic"].items():
            L += [f"## ICDL trên ISIC-2024 ({kind})", "", "| s | AUROC_Y | AUROC_D đã xác minh | p̂ ∈ 𝓘 | WCR | p̂ trung bình |",
                  "| ---: | ---: | ---: | ---: | ---: | ---: |"]
            for s, r in v["per_s"].items():
                L.append(f"| {s} | {r['auroc_Y']:.3f} | {f(r['auroc_D_verified'])} | {r['pred_in_interval']:.3f} | "
                         f"{r['wcr_rect']:.4f} | {r['mean_pred']:.5f} |")
            dd = v["decisions"]["0.01"]
            L += ["", f"- Dải theo giả định: tỉ số trung vị max/min {v['band']['median_ratio']:.2f}; đơn điệu giảm theo s ở "
                      f"{v['band']['monotone_decreasing_in_s']:.3f} tổn thương.",
                  f"- τ = 0.01: quyết định bất biến theo giả định {dd['frac_assumption_free_model']:.4f}; ác tính bền vững "
                  f"{dd['frac_robust_malignant']:.4f}; tỉ lệ s* > s: {dd['frac_s_star_above']}; phân vị s*: {dd['s_star_quantiles']}.",
                  "", "| Concept | RR_D, ℋ chữ nhật | ℋ 5 tầng | ℋ chung (A = 1) |", "| --- | --- | --- | ---: |"]
            for c, b in v["structured_contrast_s0.5"].items():
                L.append(f"| {c} | [{b['rect'][0]:.2f}, {b['rect'][1]:.2f}] | [{b['strata'][0]:.2f}, {b['strata'][1]:.2f}] | {b['shared'][0]:.2f} |")
            L += ["", f"- WCR của ICDL (s = 0.5) dưới ℋ: {v['wcr_structured_s0.5']}", ""]
    r5 = J("vr5_representation.json")
    if r5:
        L += ["## Fine-tune và representation (vr4, vr5)", "", f"- Phán quyết §23: **{r5['verdict']}**", f"- CKA: {r5['cka']}", ""]
        for mode, sm in r5["summary"].items():
            L += [f"### {mode}", "", "| Đo | Concept | M0 (các seed) | M2 (các seed) | trái dấu ở mọi seed |", "| --- | --- | --- | --- | --- |"]
            for k in ("decision", "layer3", "alignment", "readout_Y", "readout_S"):
                for c, v in sm.get(k, {}).items():
                    L.append(f"| {k} | {c} | {', '.join(f'{x:+.3f}' for x in v['M0'])} | {', '.join(f'{x:+.3f}' for x in v['M2'])} | "
                             f"{v['opposite_sign_all_seeds']} |")
            L += ["", "| Concept | R² probe M0 | R² probe M2 |", "| --- | --- | --- |"]
            for c, v in sm["probe_r2"].items():
                L.append(f"| {c} | {', '.join(f'{x:.3f}' for x in v['M0'])} | {', '.join(f'{x:.3f}' for x in v['M2'])} |")
            L.append("")
        ft = J("vr4_finetune.json")
        if ft:
            L += ["| Run | val AUROC tốt nhất | AUROC_Y test | AUROC_D đã xác minh | phút |", "| --- | ---: | ---: | ---: | ---: |"]
            for r in ft["runs"]:
                L.append(f"| {r['mode']}_{r['regime']}_s{r['seed']} | {f(r['best_val_auroc'])} | {f(r['test_auroc_Y_uniform'])} | "
                         f"{f(r['test_auroc_D_verified'])} | {r['minutes']:.1f} |")
            L.append("")
    pad = J("vr6_pad_boundary.json")
    if pad:
        L += ["## PAD-UFES, trường hợp biên (vr6, A = 1)", "", "| Đặc trưng | log OR_D | log B | log OR_D\\|S | loại | P(đảo dấu) bootstrap |",
              "| --- | ---: | ---: | ---: | --- | ---: |"]
        for c, v in pad["features"].items():
            L.append(f"| {c} | {v['logOR_D']:+.2f} | {v['logB']:+.2f} | {v['logOR_DS']:+.2f} | {v['category']} | {v['bootstrap_prob_reversal']:.2f} |")
        L.append("")
    path = os.path.join(R.RES, "vr_summary.md")
    open(path, "w").write("\n".join(L))
    R.log(f"wrote {path}")


if __name__ == "__main__":
    main()
