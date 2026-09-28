#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build supplementary.md from supplementary_src.md. Every {{TABLE:name}} is replaced by a markdown table generated
directly from the locked JSON results, so no number in these tables is typed by hand."""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "Result")
CL = {"color_variegation": "Color variegation", "size": "Size", "lesion_skin_contrast": "Lesion-skin contrast",
      "asymmetry": "Asymmetry", "border_irregularity": "Border irregularity"}
DEFS = ["tertile", "quartile", "quintile", "median", "slope", "within_slope", "rank_slope"]


def J(n, base=RES):
    return json.load(open(os.path.join(base, n)))


def s(x, d=2):
    return f"{x:+.{d}f}".replace("-", "−")


def u(x, d=2):
    return f"{x:.{d}f}".replace("-", "−")


def sci(x):
    """1.8e-15 -> 1.8 × 10⁻¹⁵ (ký hiệu toán, không dùng dạng e của máy tính)."""
    m, e = f"{x:.1e}".split("e")
    sup = str(int(e)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m} × 10{sup}"


def ci(a, d=2):
    return f"[{u(a[0], d)}, {u(a[1], d)}]"


def t_theory():
    t, p = J("vr1_reversal_theory.json"), J("vr2_phase_diagram.json")
    a, b, ps = t["theorem4"], t["theorem5_population"]["summary"], p["summary"]
    sym = J("vr10_symbolic.json")
    nsym = sum(1 for k, v in sym.items() if isinstance(v, bool) and k != "all_passed")
    joint = t["isic2024_slopes"]["joint"]
    rj = [v["identity_residual"] for v in joint.values()]
    rows = [("Proposition 1, learner gap and general offset", "5,000 random models; symbolic", "exact to machine precision; verified symbolically"),
            ("Lemma 1, identity", f"{a['n_models']:,} random models",
             f"max error {sci(a['max_abs_identity_error'])}; region agreement {a['category_agreement']:.3f}"),
            ("Proposition 1, Lemmas 1 and 2, symbolic", "sympy, positive domain", f"all {nsym} statements verified"),
            ("Lemma 2, log-linear likelihood ratio", "48 population learners",
             f"max coefficient error {sci(b['loglinear_max_abs_err_M2'])}; reversal agreement {b['loglinear_reversal_agreement']:.3f}"),
            ("Lemma 2, verification floor 0.5", "48 population learners",
             f"max coefficient error {b['floored_max_abs_err_M2']:.3f}; reversal agreement {b['floored_reversal_agreement']:.3f}"),
            ("Phase diagram, population learner", f"{ps['n_points']} populations",
             f"{ps['category_counts']['reversal']} reversals; agreement {ps['M2_coef_vs_tertile_theorem4_agreement']:.3f}"),
            ("Phase diagram, prediction from observables", f"{ps['n_points']} populations",
             f"sign agreement {ps['observable_prediction_agreement']:.3f}"),
            ("Phase diagram, finite-sample perceptron", "24 populations, 300,000 lesions each",
             f"sign agreement {ps['mlp_M2_sign_agrees_with_theorem4']:.3f}"),
            ("ISIC-2024 slope identity, joint model", "five concepts", f"residual from {s(min(rj))} to {s(max(rj))}")]
    out = ["| Check | Setting | Result |", "| --- | --- | --- |"]
    out += [f"| {a_} | {b_} | {c_} |" for a_, b_, c_ in rows]
    return "\n".join(out)


def t_definitions():
    d = J("vr9_closing.json")["families"]
    fams = [("head_tabular", "Tabular"), ("head_image", "Image"), ("lp", "Linear probe"), ("ft", "Fine-tuned")]
    out = ["| Family | Concept | tertile | quartile | quintile | median | slope | within-patient slope | rank slope |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for f, lab in fams:
        for c in ("color_variegation", "size"):
            x = d[f]["concepts"][c]
            out.append(f"| {lab} | {CL[c]} | " + " | ".join(f"{s(x[h]['M0'])} / {s(x[h]['M2'])}" for h in DEFS) + " |")
    return "\n".join(out)


def t_size_matched():
    a = J("vr12_claim_validation.json")["size_matched"]["results"]
    out = ["| Features | Concept | Benign from all lesions | Benign from flagged, unverified | Benign from verified (M2) |",
           "| --- | --- | --- | --- | --- |"]
    for f in ("tabular", "image"):
        for c in CL:
            cells = []
            for arm in ("random", "flagged", "verified"):
                x = a[f][arm][c]
                cells.append(f"{s(x['mean'])} [{u(x['q025'])}, {u(x['q975'])}], {int(round(20 * x['frac_positive']))} of 20 positive")
            out.append(f"| {f} | {CL[c]} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def t_quant():
    q = J("vr12_quantitative.json")
    fams = [("head_tabular", "Tabular"), ("head_image", "Image"), ("lp", "Linear probe"), ("ft", "Fine-tuned")]
    vm = q["families"]["head_tabular"]["marginal"]["V_slope_single"]
    out = ["| Concept | Benign verification slope | " + " | ".join(l for _, l in fams) + " |",
           "| --- | ---: | " + " | ".join("---:" for _ in fams) + " |"]
    for k, c in enumerate(q["concepts"]):
        out.append(f"| {CL[c]} | {s(vm[k])} | " + " | ".join(s(q["families"][f]["marginal"]["gap_slopes"][k]) for f, _ in fams) + " |")
    out.append("| Calibration slope [95% CI] | | " + " | ".join(
        f"{u(q['families'][f]['marginal']['calibration_slope'])} {ci(q['families'][f]['marginal']['calibration_slope_ci'])}" for f, _ in fams) + " |")
    out.append("| Correlation [95% CI] | | " + " | ".join(
        f"{u(q['families'][f]['marginal']['pearson_r'])} {ci(q['families'][f]['marginal']['pearson_r_ci'])}" for f, _ in fams) + " |")
    out.append("| Joint model: correlation, signs agreeing | | " + " | ".join(
        f"{u(q['families'][f]['pearson_r'])}, {q['families'][f]['sign_agreement']} of 5" for f, _ in fams) + " |")
    return "\n".join(out)


def t_pad():
    p = J("vr12_claim_validation.json")["pad_learner"]
    lab = {"color_var": "Image color variegation", "contrast": "Image lesion-skin contrast", "asym": "Image asymmetry",
           "border": "Image border irregularity", "itch": "Itch", "grew": "Grew", "hurt": "Hurt", "changed": "Changed",
           "bleed": "Bleeding", "elevation": "Elevation"}
    out = ["| Feature | Δ_M0 [95% CI] | Δ_M2 [95% CI] | Category | P(signs differ) |", "| --- | --- | --- | --- | ---: |"]
    for fam in ("clinical", "image"):
        for k, x in p["families"][fam].items():
            cat = x["category"] if abs(x["M0"]) > 0.1 else "unresolved (Δ_M0 near 0)"
            out.append(f"| {lab[k]} | {s(x['M0'])} {ci(x['M0_ci'])} | {s(x['M2'])} {ci(x['M2_ci'])} | {cat} | {u(x['p_sign_differ'])} |")
    return "\n".join(out)


def headswap_effects():
    """Tách hiệu ứng: head = TB(head M2) − TB(head M0) qua hai encoder; encoder = TB(encoder M2) − TB(encoder M0)."""
    runs = J("vr13_headswap.json")["runs"]
    seeds = sorted({r["seed"] for r in runs})
    out = {}
    for c in CL:
        he, ee = [], []
        for sd in seeds:
            x = {(r["encoder"], r["head"]): r["delta"][c] for r in runs if r["seed"] == sd}
            he.append(((x["M0", "M2"] + x["M2", "M2"]) - (x["M0", "M0"] + x["M2", "M0"])) / 2)
            ee.append(((x["M2", "M0"] + x["M2", "M2"]) - (x["M0", "M0"] + x["M0", "M2"])) / 2)
        out[c] = {"head": he, "enc": ee}
    return out, len(seeds)


def t_headswap():
    h = J("vr13_headswap.json")["summary"]
    eff, n = headswap_effects()
    out = ["| Concept | Encoder M0, head M0 | Encoder M0, head M2 | Encoder M2, head M0 | Encoder M2, head M2 | Head effect [seed range] | Encoder effect [seed range] | Seeds where sign follows the head |",
           "| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |"]
    for c in CL:
        x = h[c]; e = eff[c]
        he, ee = e["head"], e["enc"]
        out.append(f"| {CL[c]} | " + " | ".join(s(x[k]["mean"]) for k in ("encM0_headM0", "encM0_headM2", "encM2_headM0", "encM2_headM2"))
                   + f" | {s(sum(he) / n)} [{u(min(he))}, {u(max(he))}] | {s(sum(ee) / n)} [{u(min(ee))}, {u(max(ee))}]"
                   + f" | {x['seeds_sign_follows_head']} of {x['n_seeds']} |")
    return "\n".join(out)


def t_icdl():
    d = J("vr3_icdl.json")["sim_rows"]
    keep = {(0.5, 0.3), (0.5, 0.5), (0.5, 0.9), (0.7, 0.5), (0.7, 0.7), (0.7, 0.9), (0.9, 0.5), (0.9, 0.9)}
    out = ["| True floor | Assumed floor | Oracle coverage of true *p* | Plug-in coverage of true *p* | Mean \\|log p̂ − log *p*\\| | AUROC for *D* | Concept sign errors |",
           "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in d:
        if r["method"] == "icdl_kl" and (r["s_true"], r["s_assumed"]) in keep:
            out.append(f"| {r['s_true']} | {r['s_assumed']} | {r['true_p_in_oracle_interval']:.3f} | {r['true_p_in_interval']:.3f} | "
                       f"{r['abs_log_err']:.3f} | {r['auroc_D']:.3f} | {r['n_sign_wrong']} |")
    return "\n".join(out)


def t_repr():
    sm = J("vr5_representation.json")["summary"]["ft"]
    out = ["| Diagnostic | Concept | M0 | M2 | Seeds with opposite sign |", "| --- | --- | ---: | ---: | ---: |"]
    for k, lab in (("readout_Y", "Common readout, recorded-label risk"), ("readout_S", "Common readout, verified-lesion risk"),
                   ("layer3", "Layer-3 shift, output change")):
        for c in ("color_variegation", "lesion_skin_contrast", "asymmetry", "border_irregularity"):
            v = sm[k][c]
            out.append(f"| {lab} | {CL[c]} | {s(sum(v['M0']) / len(v['M0']), 3)} | {s(sum(v['M2']) / len(v['M2']), 3)} | "
                       f"{v['n_seeds_opposite']} of {len(v['M0'])} |")
    return "\n".join(out)


def t_dose():
    d = J("vr15_dose_response.json")
    out = ["| Features | Selected concept | Δ at η = −1 / 0 / 3 | Slope, observed [replicate range] | Slope, predicted | Zero crossing, observed / predicted | Other concepts moving as predicted |",
           "| --- | --- | --- | --- | ---: | --- | ---: |"]
    for key, r in d["results"].items():
        fam, k = key.split("|")
        m = r["mean_delta_by_eta"][k]; e = d["etas"]
        cr = u(r["eta_star_observed"]) if r["eta_star_observed"] is not None else "none"
        out.append(f"| {fam} | {CL[k]} | {s(m[e.index(-1.0)])} / {s(m[e.index(0.0)])} / {s(m[e.index(3.0)])} | "
                   f"{s(r['observed_slope'][k])} [{u(r['target_slope_replicate_range'][0])}, {u(r['target_slope_replicate_range'][1])}] | "
                   f"{s(r['predicted_slope'][k])} | {cr} / {u(r['eta_star_predicted'])} | {r['spillover_sign_agreement']} of 5 |")
    return "\n".join(out)


def t_bridge():
    b = J("vr17_joint_bootstrap.json")["calibrated_bridge"]
    out = ["| Concept | Tabular: predicted | Tabular: observed, calibrated | Image: predicted | Image: observed, calibrated | Image: observed, raw |",
           "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for j, c in enumerate(b["tabular"]["concepts"]):
        out.append(f"| {CL[c]} | {s(b['tabular']['predicted'][j])} | {s(b['tabular']['gap_calibrated'][j])} | "
                   f"{s(b['image']['predicted'][j])} | {s(b['image']['gap_calibrated'][j])} | {s(b['image']['gap_raw'][j])} |")
    out.append(f"| Calibration slope, calibrated / raw | | {u(b['tabular']['calibration_slope_calibrated'])} / {u(b['tabular']['calibration_slope_raw'])} | "
               f"| {u(b['image']['calibration_slope_calibrated'])} / {u(b['image']['calibration_slope_raw'])} | |")
    return "\n".join(out)


def t_within():
    m = J("vr16_mechanism.json")["within_between"]
    out = ["| Concept | Within patient (SE) | Between patients (SE) |", "| --- | --- | --- |"]
    for c, v in m.items():
        out.append(f"| {CL[c]} | {s(v['within'], 3)} ({u(v['within_se'], 3)}) | {s(v['between'], 3)} ({u(v['between_se'], 3)}) |")
    return "\n".join(out)


def _site_order():
    """Mọi cơ sở theo thứ tự chữ cái; cùng một số hiệu ở Table S16 và S18."""
    names = set(J("vr16_mechanism.json")["by_site"]["color_variegation"]["sites"])
    names |= set(J("vr20_audit.json").get("leave_one_site_out", {}).get("tabular", {}))
    return sorted(names)


SITE_VB = {"ViDIR Group, Department of Dermatology, ": "7 in all tertiles"}     # số tổn thương lành đã xác minh; dưới ngưỡng 10 nên vr16 bỏ qua


def t_sites():
    m = J("vr16_mechanism.json")["by_site"]
    out = ["| Site | Verified benign lesions | " + " | ".join(CL[c] for c in PRIMARY_SUPP) + " |",
           "| --- | ---: | " + " | ".join("---" for _ in PRIMARY_SUPP) + " |"]
    for i, st in enumerate(_site_order()):
        if st not in m["color_variegation"]["sites"]:
            out.append(f"| Site {i + 1} | {SITE_VB.get(st, 'fewer than 10')} | " + " | ".join("fewer than ten verified benign" for _ in PRIMARY_SUPP) + " |")
            continue
        n = m["color_variegation"]["sites"][st]["n_verified"]
        cells = []
        for c in PRIMARY_SUPP:
            v = m[c]["sites"].get(st)
            cells.append("n/a" if v is None else (f"{s(v['log_BV'], 3)} ({u(v['se'], 3)})" if v["log_BV"] < 8 else "separated"))
        out.append(f"| Site {i + 1} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def t_overlap():
    o = J("vr16_mechanism.json")["overlap"]
    lab = {"all_test": "all test lesions", "supported": "within support", "flagged": "flagged lesions"}
    out = ["| Features | Population (share of test) | Concept | Δ_M0 [95% CI] | Δ_M2 [95% CI] | P(signs differ) |",
           "| --- | --- | --- | --- | --- | ---: |"]
    for fam, f in o["families"].items():
        for sub, r in f.items():
            for c, v in r.items():
                out.append(f"| {fam} | {lab[sub]} ({u(100 * o['fraction_retained'][sub], 0)} percent) | {CL[c]} | "
                           f"{s(v['M0'])} {ci(v['M0_ci'])} | {s(v['M2'])} {ci(v['M2_ci'])} | {v['p_sign_differ']:.3f} |")
    return "\n".join(out)


def t_joint():
    jb = J("vr17_joint_bootstrap.json")["joint_bootstrap"]["families"]
    out = ["| Features | Concept | Δ_M0, 95% / Bonferroni | Δ_M2, 95% / Bonferroni | Robust, 95% / Bonferroni |",
           "| --- | --- | --- | --- | --- |"]
    for fam in ("tabular", "image"):
        for c in CL:
            b = jb[fam][c]
            out.append(f"| {fam} | {CL[c]} | {ci(b['M0_ci'])} / {ci(b['M0_ci_bonferroni'])} | {ci(b['M2_ci'])} / {ci(b['M2_ci_bonferroni'])} | "
                       f"{'yes' if b['robust_reversal_95'] else 'no'} / {'yes' if b['robust_reversal_bonferroni'] else 'no'} |")
    return "\n".join(out)


FAMS2 = ("tabular", "image")


def t_joint19():
    d = J("vr19_primary_bootstrap.json")
    out = ["| Features | Concept | Δ_M0: estimate, 95% / Bonferroni | Δ_M2: estimate, 95% / Bonferroni | Calibrated Δ_M0 / Δ_M2, 95% | Robust, 95% / Bonferroni |",
           "| --- | --- | --- | --- | --- | --- |"]
    for f in FAMS2:
        for c in CL:
            e = d["families"][f]["concepts"][c]
            out.append(f"| {f} | {CL[c]} | {s(e['d0_orig'])}, {ci(e['d0_ci95'])} / {ci(e['d0_ci_bonf'])} | {s(e['d2_orig'])}, {ci(e['d2_ci95'])} / {ci(e['d2_ci_bonf'])} | "
                       f"{ci(e['d0c_ci95'])} / {ci(e['d2c_ci95'])} | {'yes' if e['robust_95'] else 'no'} / {'yes' if e['robust_bonf'] else 'no'} |")
    return "\n".join(out)


def t_psi():
    d = J("vr19_primary_bootstrap.json")
    out = ["| Features | Concept | ψ bounds at *s*_min = 0.5 | ψ bounds at *s*_min = 0.7 | ψ bounds at *s*_min = 0.9 | Tipping floor s∗, original / 95th percentile |",
           "| --- | --- | --- | --- | --- | --- |"]
    pc = lambda v: "none" if v is None else u(v)
    for f in FAMS2:
        for c in PRIMARY_SUPP:
            e = d["families"][f]["concepts"][c]
            b = lambda sv: f"[{u(e['psiL_orig'][sv])}, {u(e['psiU_orig'][sv])}]"
            sp = J("vr24_full_bootstrap.json")["families"][f]["concepts"][c]["s_star_platt"]["orig_mean_curve"]
            sp = None if sp != sp else sp
            out.append(f"| {f} | {CL[c]} | {b('0.5')} | {b('0.7')} | {b('0.9')} | {pc(sp)} / {pc(e['s_pos_q95'])} |")
    return "\n".join(out)


def t_support19():
    d = J("vr19_primary_bootstrap.json")
    thr = d["families"]["tabular"]["support_thresholds"]
    out = ["| Features | Concept | Threshold percentile | Δ_M0 [95% CI] | Δ_M2 [95% CI] | Robust at 95% |", "| --- | --- | ---: | --- | --- | --- |"]
    for f in FAMS2:
        for c in PRIMARY_SUPP:
            for q in sorted(thr, key=float):
                v = d["families"][f]["concepts"][c]["support"][q]
                out.append(f"| {f} | {CL[c]} | {round(100 * float(q))} | {s(v['d0_orig'])} {ci(v['d0_ci95'])} | {s(v['d2_orig'])} {ci(v['d2_ci95'])} | "
                           f"{'yes' if v['robust_95'] else 'no'} |")
    return "\n".join(out)


def t_dose21():
    d = J("vr21_dose_calibrated.json")
    out = ["| Features | Selected concept | Calibrated slope, η in [−1, 1] [replicate range] | Raw slope | Predicted from input | Predicted if z_k observed | Crossing, observed / predicted | Concepts moving as predicted |",
           "| --- | --- | --- | ---: | ---: | ---: | --- | ---: |"]
    for f in FAMS2:
        for k in PRIMARY_SUPP:
            r = d["families"][f]["dose"][k]
            cr = u(r["cal"]["crossing"]) if r["cal"]["crossing"] is not None else "none"
            cp = u(r["crossing_predicted_input"]) if r["crossing_predicted_input"] is not None else "none"
            out.append(f"| {f} | {CL[k]} | {s(r['cal']['slope_mean'])} {ci(r['cal']['slope_range95'])} | {s(r['raw']['slope_mean'])} | "
                       f"{s(r['predicted_slope_input'])} | {s(r['predicted_slope_zk'])} | {cr} / {cp} | {r['spillover_sign_agreement']} of 5 |")
    return "\n".join(out)


def t_ess21():
    d = J("vr21_dose_calibrated.json")["support"]["color_variegation"]
    out = ["| η | Effective sample size, share of pool | Share of selection weight in lower tertile | Share in upper tertile |", "| ---: | ---: | ---: | ---: |"]
    for r in d:
        out.append(f"| {u(r['eta'], 1)} | {u(r['ess_frac'], 3)} | {u(r['share_lower'], 3)} | {u(r['share_upper'], 3)} |")
    return "\n".join(out)


def t_arms21():
    d = J("vr21_dose_calibrated.json")
    out = ["| Features | Arm | Concept | Observed shift from random arm | Predicted shift |", "| --- | --- | --- | ---: | ---: |"]
    for f in FAMS2:
        for a, lab in (("ver", "verified"), ("flag", "flagged, unverified")):
            for c in CL:
                v = d["families"][f]["arms"][a][c]
                out.append(f"| {f} | {lab} | {CL[c]} | {s(v['observed_shift'])} | {s(v['predicted_shift'])} |")
    return "\n".join(out)


def t_pointwise22():
    d = J("vr22_pointwise_bridge.json")
    out = ["| Features | Concept | Observed gap [95% CI] | Predicted gap [95% CI] | Same sign in resamples |", "| --- | --- | --- | --- | ---: |"]
    for f in FAMS2:
        fam = d["families"][f]
        for c in CL:
            v = fam["concepts"][c]
            out.append(f"| {f} | {CL[c]} | {s(v['observed'])} {ci(v['observed_ci'])} | {s(v['predicted'])} {ci(v['predicted_ci'])} | {u(v['sign_agreement_frac'], 3)} |")
        out.append(f"| {f} | lesion-level slope, median R² | {u(fam['pointwise_slope'])} {ci(fam['pointwise_slope_ci'])} | R² {u(fam['pointwise_r2'])} | |")
    return "\n".join(out)


def t_sitehet():
    a = J("vr20_audit.json")
    out = ["| Concept | Estimable sites | Q (df) | I² | τ² | Pooled log *B*_V [95% CI] | Site range |", "| --- | ---: | --- | ---: | ---: | --- | --- |"]
    for c in PRIMARY_SUPP:
        h = a["site_heterogeneity"][c]
        out.append(f"| {CL[c]} | {h['k']} | {u(h['Q'], 1)} ({h['df']}) | {u(h['I2'])} | {u(h['tau2'], 3)} | {u(h['pooled_DL'])} {ci(h['pooled_ci'])} | {ci(h['range'])} |")
    return "\n".join(out)


def t_loso():
    a = J("vr20_audit.json").get("leave_one_site_out", {})
    sites = _site_order()
    out = ["| Features | Site left out | Color Δ_M0 / Δ_M2 | Size Δ_M0 / Δ_M2 | Contrast Δ_M0 / Δ_M2 |", "| --- | --- | --- | --- | --- |"]
    for f in FAMS2:
        for st, v in a.get(f, {}).items():
            idx = sites.index(st) + 1
            out.append(f"| {f} | Site {idx} | " + " | ".join(f"{s(v[c][0])} / {s(v[c][1])}" for c in PRIMARY_SUPP) + " |")
    return "\n".join(out)


def t_psisim():
    d = J("vr23_psi_sim.json")["summary"]
    out = ["| Setting | Oracle set contains ψ | Plug-in set contains ψ | Sign certified, oracle / plug-in | Wrong sign certified |", "| --- | ---: | ---: | --- | ---: |"]
    for k, lab in (("valid_floor", "assumed floor at most the true one"), ("valid_floor_correlated", "assumed floor at most the true one, correlated concepts"),
                   ("overstated_floor", "assumed floor above the true one")):
        v = d[k]
        out.append(f"| {lab} | {u(v['oracle_covers'])} | {u(v['plugin_covers'])} | {u(v['oracle_certifies'])} / {u(v['plugin_certifies'])} | "
                   f"{u(max(v['oracle_wrong_sign'], v['plugin_wrong_sign']))} |")
    return "\n".join(out)


def t_full24():
    d = J("vr24_full_bootstrap.json"); d19 = J("vr19_primary_bootstrap.json")
    fl = lambda x: "none" if x is None or x != x else u(x)
    out = ["| Features | Concept | Robust, train and test / all three splits resampled | Plug-in s∗: point, 95th percentile, one-sided bound | Isotonic s∗: point, 95th percentile | s∗ within support at the 1st / 5th / 10th percentile | ψ_L > 0 and Δ_M2 < 0 at s = 0.7 / 0.8 / 0.9 |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    rb = lambda e: "Bonferroni" if e["robust_bonf"] else ("95 percent only" if e["robust_95"] else "no")
    for f in FAMS2:
        for c in PRIMARY_SUPP:
            e = d["families"][f]["concepts"][c]; e19 = d19["families"][f]["concepts"][c]
            p, i = e["s_star_platt"], e["s_star_iso"]
            sup = " / ".join(fl(e["s_star_support"][q]["orig"]) for q in ("0.01", "0.05", "0.1"))
            jt = " / ".join(u(p["joint_psiL_pos_and_d2_neg"][s]) for s in ("0.7", "0.8", "0.9"))
            out.append(f"| {f} | {CL[c]} | {rb(e19)} / {rb(e)} | {fl(p['orig_mean_curve'])}, {fl(p['q95'])}, {fl(p['lcb_floor'])} | "
                       f"{fl(i['orig_mean_curve'])}, {fl(i['q95'])} | {sup} | {jt} |")
    return "\n".join(out)


EST30 = [("mlp_platt", "perceptron, Platt"), ("mlp_iso", "perceptron, isotonic"), ("gbm_platt", "gradient boosting, Platt")]


def _f(x):
    return "none" if x is None or x != x else u(x)


def t_full30():
    d = J("vr30_q_bootstrap.json"); d19 = J("vr19_primary_bootstrap.json")
    rb = lambda e: "Bonferroni" if e["robust_bonf"] else ("95 percent only" if e["robust_95"] else "no")
    out = ["| Features | Concept | Robust, one seed per replicate: train and test / all three splits | Estimate of *q* | s∗: point, 95th percentile | Bootstrap stability threshold | Within support, point s∗ at the 1st / 5th / 10th percentile | Joint count at s = 0.7 / 0.8 / 0.9 |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for f in FAMS2:
        for c in PRIMARY_SUPP:
            e = d["families"][f]["concepts"][c]; e19 = d19["families"][f]["concepts"][c]
            for k, lab in EST30:
                v = e[k]
                out.append(f"| {f} | {CL[c]} | {rb(e19)} / {rb(e)} | {lab} | {_f(v['point'])}, {_f(v['q95'])} | {_f(v['lcb_floor'])} | "
                           + " / ".join(_f(v["support"][q]["point"]) for q in ("0.01", "0.05", "0.1")) + " | "
                           + " / ".join(str(v["joint_counts"][s_]) for s_ in ("0.7", "0.8", "0.9")) + f" of {d['B'][f]} |")
    return "\n".join(out)


def t_qdiag30():
    d = J("vr30_q_bootstrap.json")
    out = ["| Features | Estimate of *q* | Log-loss | Brier | Predicted risk below 0.0005: lesions; expected / observed malignant | 0.0005 to 0.002 | 0.002 to 0.01 | Above 0.01 |",
           "| --- | --- | ---: | ---: | --- | --- | --- | --- |"]
    labs = EST30 + [("mlp_raw", "perceptron, uncalibrated")]
    for f in FAMS2:
        for k, lab in labs:
            g = d["families"][f]["diagnostics"][k]
            cells = [f"{round(b['n']):,}; {b['n'] * b['mean_q']:.1f} / {b['n_pos']:.1f}" if b["mean_q"] is not None else f"{round(b['n']):,}; none / {b['n_pos']:.1f}" for b in g["bins"]]
            out.append(f"| {f} | {lab} | {g['logloss']:.5f} | {g['brier']:.6f} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def t_marginal35():
    d = J("vr35_marginal_rr.json")["populations"]
    B = J("vr35_marginal_rr.json")["B"]
    jf = lambda jc: next((k for k, v in sorted(jc.items(), key=lambda z: float(z[0])) if v >= 0.95 * B), None)
    out = ["| Population | Concept | Malignant, lower / upper tertile | RR_Y [95% CI] | Floor 1/RR_Y | Floor 1/q₀.₀₅ | Verified-only log OR [95% CI] | 95% resampling stability threshold | Largest *A* at 95% | *B*_V [95% CI] |",
           "| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- |"]
    for pop, lab in (("cohort", "whole cohort"), ("test", "test population")):
        for c in CL:
            e = d[pop][c]
            out.append(f"| {lab} | {CL[c]} | {e['n_pos'][0]} / {e['n_pos'][1]} | {u(e['rr_y'])} {ci(e['rr_y_ci95'])} | {_f(e['tipping'])} | {_f(e['tipping_one_sided'])} | "
                       f"{s(e['log_or_verified'])} {ci(e['log_or_verified_ci95'])} | {jf(e['joint_counts']) or 'none'} | "
                       f"{_f(e['A_max95'])} | {u(e['B_V'])} {ci(e['B_V_ci95'])} |")
    return "\n".join(out)


def t_sub36():
    d = J("vr36_subsample.json")["families"]
    out = ["| Features | Concept | Δ_M0, Bonferroni-scaled interval | Δ_M2, Bonferroni-scaled interval | Sign separation of subsampling intervals, 95% / Bonferroni-scaled | Same sign in subsamples |", "| --- | --- | --- | --- | --- | ---: |"]
    for f in FAMS2:
        for c in PRIMARY_SUPP:
            e = d[f][c]
            out.append(f"| {f} | {CL[c]} | {ci(e['d0_ci_bonf'])} | {ci(e['d2_ci_bonf'])} | {'yes' if e['robust_95'] else 'no'} / {'yes' if e['robust_bonf'] else 'no'} | {u(e['same_sign_frac'])} |")
    return "\n".join(out)


def t_dose46():
    d = J("vr46_dose_fixed_val.json")["families"]
    out = ["| Features | Selected concept | Recorded negatives | Observed slope [range] | Predicted slope | Observed over predicted [95% CI] | τ̂(1, −1) [95% CI] | Mean contrast crosses zero | Replicates changing sign | Calibrated τ̂(1, −1) |",
           "| --- | --- | ---: | --- | ---: | --- | --- | --- | ---: | ---: |"]
    for f in FAMS2:
        for k in PRIMARY_SUPP:
            for n in ("320", "1000", "3000"):
                c = d.get(f, {}).get(f"{k}|{n}")
                if c is None:
                    continue
                r, p = c["raw"], c["pred"]
                out.append(f"| {f} | {CL[k]} | {int(n):,} | {s(r['obs_slope'])} {ci(r['obs_range95'])} | {s(p['slope'])} | "
                           f"{u(p['ratio_raw'])} {ci(p['ratio_raw_ci95'])} | {s(r['tau_hat'])} {ci(r['tau_ci95'])} | "
                           f"{'yes' if r['crosses_zero'] else 'no'} | {r['n_sign_change']} of {c['n_rep']} | {s(c['cal']['tau_hat'])} |")
    return "\n".join(out)


def t_spill46():
    d = J("vr46_dose_fixed_val.json")["families"]
    out = ["| Features | Selected concept | " + " | ".join(CL[j] for j in CL) + " |", "| --- | --- | " + " | ".join("---" for _ in CL) + " |"]
    for f in FAMS2:
        for k in PRIMARY_SUPP:
            c = d[f][f"{k}|320"]
            out.append(f"| {f} | {CL[k]} | " + " | ".join(f"{s(c['raw']['spill_obs'][j])} / {s(c['pred']['spill_pred'][j])}" for j in CL) + " |")
    return "\n".join(out)


def t_dose32():
    d = J("vr32_dose_nuisance.json")["families"]
    out = ["| Features | Selected concept | Recorded negatives | Observed slope [range] | Prediction | Predicted slope [range] | Observed over predicted [95% CI] |",
           "| --- | --- | ---: | --- | --- | --- | --- |"]
    lab = {"exact": "Bayes-optimal shift, concept in input", "ridge": "ridge conditional mean", "mlp": "perceptron conditional mean"}
    for f in FAMS2:
        for key, c in d.get(f, {}).items():
            k, n = key.split("|")
            for sp, v in c["pred"].items():
                out.append(f"| {f} | {CL[k]} | {int(n):,} | {s(c['obs_slope'])} {ci(c['obs_range95'])} | {lab[sp]} | {s(v['slope'])} {ci(v['range95'])} | {u(v['ratio'])} {ci(v['ratio_ci95'])} |")
    return "\n".join(out)


def t_spill32():
    d = J("vr32_dose_nuisance.json")["families"]
    out = ["| Features | Selected concept | Prediction | " + " | ".join(CL[j] for j in CL) + " |", "| --- | --- | --- | " + " | ".join("---" for _ in CL) + " |"]
    for f in FAMS2:
        for key, c in d.get(f, {}).items():
            k, n = key.split("|")
            if n != "320":
                continue
            for sp, v in c["pred"].items():
                out.append(f"| {f} | {CL[k]} | {dict(exact='Bayes-optimal shift').get(sp, sp)} | " + " | ".join(f"{s(v['spill'][j][0])} / {s(v['spill'][j][1])}" for j in CL) + " |")
    return "\n".join(out)


def t_bridge33():
    d = J("vr33_bridge_alt.json")
    out = ["| Features | Model for ĝ | Test population | Slope [95% CI] | Median R² | Concepts with sign agreement ≥ 97.5% | Concept calibration slope [95% CI] |", "| --- | --- | --- | --- | ---: | ---: | --- |"]
    for f in FAMS2:
        for key, e in d["families"][f].items():
            m, reg = key.split("|")
            out.append(f"| {f} | {'perceptron' if m == 'mlp' else 'gradient boosting'} | {'full' if reg == 'all' else 'supported, 1st percentile'} | {u(e['slope'])} {ci(e['slope_ci'])} | "
                       f"{u(e['r2'])} | {e['n_concepts_975']} of 5 | {u(e['concept_calibration'])} {ci(e['concept_calibration_ci'])} |")
    return "\n".join(out)


def t_sigma37():
    d = J("vr37_overlap_sigma.json")
    lab = {"nuisance": "gradient boosting, tabular (common)", "mlp_tab": "perceptron, tabular", "mlp_img": "perceptron, image"}
    out = ["| Features | Estimate of σ | Threshold | Test retained | Changed membership | Color Δ_M0 [95% CI] | Color Δ_M2 [95% CI] | Robust at 95%: color / size / contrast |",
           "| --- | --- | ---: | ---: | ---: | --- | --- | --- |"]
    for f in FAMS2:
        for nm in d["estimators"]:
            for q in ("0.01", "0.05", "0.1"):
                c = d["contrasts"][f"{f}|{nm}|{q}|color_variegation"]
                rb = " / ".join("yes" if d["contrasts"][f"{f}|{nm}|{q}|{k}"]["robust_95"] else "no" for k in PRIMARY_SUPP)
                out.append(f"| {f} | {lab[nm]} | {round(100 * float(q))}{'st' if q == '0.01' else 'th'} | {u(d['thresholds'][f'{nm}|{q}']['retained'])} | "
                           f"{u(d['membership'][f'{nm}|{q}']['switched'])} | {s(c['d0'])} {ci(c['d0_ci'])} | {s(c['d2'])} {ci(c['d2_ci'])} | {rb} |")
    return "\n".join(out)


def t_tail38():
    d = J("vr38_logit_tail.json")["families"]
    lab = {"mlp_platt": "perceptron, Platt", "mlp_iso": "perceptron, isotonic", "gbm_platt": "gradient boosting, Platt"}
    out = ["| Features | Estimate of *q* | Tertile | Lesions | Observed malignant | Expected (sum of *q*) | Mean *q* | Mean logit *q* | Share with *q* < 0.0005 |",
           "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for f in FAMS2:
        for k, l in lab.items():
            for tt, tl in (("0", "lower"), ("1", "upper")):
                e = d[f][k][tt]
                out.append(f"| {f} | {l} | {tl} | {e['n']:,} | {e['observed']} | {e['sum_q']:.1f} | {e['mean_q']:.5f} | {u(e['mean_logit_q'])} | {u(e['frac_below_5e-4'])} |")
    return "\n".join(out)


def t_seed39():
    d = J("vr39_seed_variance.json")["families"]; mc = J("vr43_three_seed_primary.json")["families"]
    out = ["| Features | Concept | Contrast | SD between seeds, same resample | SD of single-seed replicates | Share of variance from seeds | Bonferroni interval, single seed | Bonferroni interval, three-seed average | Label, single / three-seed |",
           "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |"]
    lab_ = lambda x: "Bonferroni" if x["robust_bonf"] else ("95 percent only" if x["robust_95"] else "no")
    for f in FAMS2:
        for c in PRIMARY_SUPP:
            ec = d[f]["concepts"][c]
            for k, lab in (("d0", "Δ_M0"), ("d2", "Δ_M2")):
                e = ec[k]
                out.append(f"| {f} | {CL[c]} | {lab} | {u(e['sd_within_seed'])} | {u(e['sd_single_seed'])} | {u(e['seed_share'])} | "
                           f"{ci(ec['single'][k + '_ci_bonf'])} | {ci(ec['avg3'][k + '_ci_bonf'])} | {lab_(ec['single'])} / {lab_(ec['avg3'])}{' (unresolved)' if mc[f]['concepts'][c]['label_stability'] < 0.95 else ''} |")
    return "\n".join(out)


def t_local34():
    d = J("vr34_semisynth_local.json")["families"]
    out = ["| Features | Construction | Containment rate | Sign certified | Wrong sign certified |", "| --- | --- | ---: | ---: | ---: |"]
    lab = [("oracle", "oracle, true nuisances")] + [(k, l) for k, l in EST30]
    for f in FAMS2:
        sm, rows = d[f]["summary"], d[f]["rows"]
        for k, l in lab:
            v = sm[k]
            cert = v["certifies"] if k != "oracle" else sum(r["oracle_L"] > 0 or r["oracle_U"] < 0 for r in rows) / len(rows)
            out.append(f"| {f} | {l} | {u(v['covers'])} | {u(cert)} | {u(v['wrong'])} |")
    return "\n".join(out)


def t_comp40():
    d = J("vr40_support_composition.json")
    lab = {"nuisance": "gradient boosting, tabular", "mlp_tab": "perceptron, tabular", "mlp_img": "perceptron, image"}
    out = ["| Features | Estimate of σ | Threshold | Share of test | M0: raw / reweighted [95% CI] | M2: raw / reweighted [95% CI] |",
           "| --- | --- | ---: | ---: | --- | --- |"]
    c = "color_variegation"
    for f in FAMS2:
        out.append(f"| {f} | full population | none | 1.00 | {s(d['full'][f'{f}|M0|{c}']['est'])} {ci(d['full'][f'{f}|M0|{c}']['ci'])} | "
                   f"{s(d['full'][f'{f}|M2|{c}']['est'])} {ci(d['full'][f'{f}|M2|{c}']['ci'])} |")
        for nm in ("mlp_tab", "mlp_img", "nuisance"):
            for q in ("0.01", "0.05", "0.1"):
                e0 = d["support"][f"{f}|M0|{nm}|{q}|{c}"]; e2 = d["support"][f"{f}|M2|{nm}|{q}|{c}"]
                out.append(f"| {f} | {lab[nm]} | {round(100 * float(q))}{'st' if q == '0.01' else 'th'} | {u(e0['share'])} | "
                           f"{s(e0['raw'])} / {s(e0['transported'])} {ci(e0['transported_ci'])} | {s(e2['raw'])} / {s(e2['transported'])} {ci(e2['transported_ci'])} |")
    return "\n".join(out)


def t_partial40():
    d = J("vr40_support_composition.json")
    out = ["| Features | Concept | Δ_M0: marginal / balanced on other concepts [95% CI] | Δ_M2: marginal / balanced on other concepts [95% CI] |",
           "| --- | --- | --- | --- |"]
    for f in FAMS2:
        for c in PRIMARY_SUPP:
            a0, p0 = d["full"][f"{f}|M0|{c}"], d["partial_full"][f"{f}|M0|{c}"]
            a2, p2 = d["full"][f"{f}|M2|{c}"], d["partial_full"][f"{f}|M2|{c}"]
            out.append(f"| {f} | {CL[c]} | {s(a0['est'])} / {s(p0['est'])} {ci(p0['ci'])} | {s(a2['est'])} / {s(p2['est'])} {ci(p2['ci'])} |")
    return "\n".join(out)


def t_site41():
    d = J("vr41_marginal_site.json")
    order = _site_order()
    out = ["| Concept | Site | Malignant, lower / upper tertile | Verified benign, lower / upper | RR_Y | Floor 1/RR_Y | Verified-only log OR | 95% resampling stability threshold without this site |",
           "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for c in ("color_variegation", "size"):
        e = d["concepts"][c]
        for nm, v in e["by_site"].items():
            sid = next((f"Site {i}" for i, full in enumerate(order, 1) if full[:40] == nm[:40]), nm[:20])
            lo = e["loso"][nm]["joint_floor"]
            out.append(f"| {CL[c]} | {sid} | {v['n_pos'][0]} / {v['n_pos'][1]} | {v['n_vb'][0]} / {v['n_vb'][1]} | "
                       f"{_f(v['rr_y'])} | {_f(v['floor'])} | {s(v['log_or_verified']) if v['log_or_verified'] is not None else 'not estimable'} | {_f(lo)} |")
        st = e["standardized"]
        out.append(f"| {CL[c]} | standardized across sites | | | {u(st['rr_y'])} {ci(st['rr_y_ci95'])} | | "
                   f"{s(st['log_or_mh'])} {ci(st['log_or_mh_ci95'])} | {_f(st['joint_floor'])} (all sites) |")
    return "\n".join(out)


def t_theory42():
    d = J("vr42_sharp_weighted.json")["theory"]
    out = ["| Noise of the concept given the input | Floor | True ψ | Stratum formula [ψ_L, ψ_U] | Sharp set [ψ_L, ψ_U] | Lower endpoint with the weight sign reversed |",
           "| ---: | ---: | ---: | --- | --- | ---: |"]
    for e in d:
        out.append(f"| {u(e['overlap'], 1)} | {u(e['s_min'], 1)} | {s(e['psi'])} | {ci(e['naive'])} | {ci(e['sharp'])} | {s(e['sign_flipped_L'])} |")
    return "\n".join(out)


def t_sharp42():
    d = J("vr42_sharp_weighted.json")["isic"]
    fl = lambda x: "none" if x is None else u(x)
    out = ["| Features | Concept | Agreement of estimated weight sign with observed stratum | Stratum formula (outer set): point floor / stability threshold | Estimated-weight approximation: point floor / stability threshold |",
           "| --- | --- | ---: | --- | --- |"]
    for f in FAMS2:
        for c in PRIMARY_SUPP:
            e = d[f"{f}|{c}"]
            out.append(f"| {f} | {CL[c]} | {u(e['sign_agreement'])} | {fl(e['tertile_formula']['s_star'])} / {fl(e['tertile_formula']['stability'])} | "
                       f"{fl(e['sharp_plugin']['s_star'])} / {fl(e['sharp_plugin']['stability'])} |")
    return "\n".join(out)


def t_semi34():
    out = ["| Disease risk built from | Features | Estimate of *q* | Scenario | Mean ψ_L minus oracle | Containment rate | Sign certified | Wrong sign certified |",
           "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for fn, dl in (("vr34_semisynth.json", "perceptron"), ("vr34_semisynth_gbm.json", "gradient boosting")):
        d = J(fn)["families"]
        for f in FAMS2:
            sm = d[f]["summary"]
            for k, lab in EST30:
                for sc, sl in (("aligned", "positive target"), ("null", "target near zero"), ("reversed", "negative target")):
                    v = sm[k][sc]
                    out.append(f"| {dl} | {f} | {lab} | {sl} | {s(v['bias_L'])} | {u(v['covers'])} | {u(v['certifies'])} | {u(v['wrong'])} |")
    return "\n".join(out)


def t_semibridge34():
    out = ["| Disease risk built from | Features | Slope on log ĝ, perceptron: median [range] | Gradient boosting: median [range] | Slope on true log g: median |",
           "| --- | --- | --- | --- | ---: |"]
    for fn, dl in (("vr34_semisynth.json", "perceptron"), ("vr34_semisynth_gbm.json", "gradient boosting")):
        d = J(fn)["families"]
        for f in FAMS2:
            sm = d[f]["summary"]
            out.append(f"| {dl} | {f} | {u(sm['bridge_slope']['median'])} {ci(sm['bridge_slope']['range'])} | "
                       f"{u(sm['bridge_slope_gbm']['median'])} {ci(sm['bridge_slope_gbm']['range'])} | {u(sm['bridge_slope']['true_g_median'])} |")
    return "\n".join(out)


def t_psistress():
    d = J("vr28_psi_stress.json")["summary"]
    lab = {"base": "positive disease effects", "adversarial": "adversarial"}
    out = ["| Design | Floor | Settings | ψ ≤ 0 | Contains ψ: oracle / GBM / perceptron | Certifies a sign: oracle / GBM / perceptron | Wrong sign: oracle / GBM / perceptron |",
           "| --- | --- | ---: | ---: | --- | --- | --- |"]
    for k, v in d.items():
        sc, fl = k.split("|")
        f3 = lambda m: " / ".join(u(v[f"{nm}_{m}"]) for nm in ("oracle", "gbm", "mlp_platt"))
        out.append(f"| {lab[sc]} | {'valid' if fl == 'valid' else 'overstated'} | {v['n']} | {v['n_psi_nonpositive']} | {f3('covers')} | {f3('certifies')} | {f3('wrong')} |")
    return "\n".join(out)


def t_dose26():
    d = J("vr26_dose_poisson.json")
    out = ["| Features | Selected concept | Observed slope [range] | Predicted, first order [range] | Predicted, second order | Observed over predicted | Crossing η | Concepts moving as predicted |",
           "| --- | --- | --- | --- | --- | ---: | ---: | ---: |"]
    for f in FAMS2:
        for k in PRIMARY_SUPP:
            r = d["families"][f]["dose"][k]; c = r["concepts"][k]
            out.append(f"| {f} | {CL[k]} | {s(c['obs_slope'])} {ci(c['obs_range95'])} | {s(c['pred_slope'])} {ci(c['pred_range95'])} | "
                       f"{s(c['pred2_slope'])} | {u(c['ratio'])} | {u(r['crossing']) if r['crossing'] is not None else 'none'} | {r['spillover_sign_agreement']} of 5 |")
    return "\n".join(out)


def t_lc26():
    d = J("vr26_dose_poisson.json")
    out = ["| Features | Recorded negatives selected | Observed slope [range] | Predicted slope | Observed over predicted |", "| --- | ---: | --- | --- | ---: |"]
    for f in FAMS2:
        for n in ("320", "1000", "3000"):
            c = d["families"][f]["learning_curve"][n]["concepts"]["color_variegation"]
            out.append(f"| {f} | {int(n):,} | {s(c['obs_slope'])} {ci(c['obs_range95'])} | {s(c['pred_slope'])} | {u(c['ratio'])} |")
    return "\n".join(out)


def t_gumbel26():
    d = J("vr26_dose_poisson.json")["gumbel_check"]
    out = ["| Concept | η | Expected upper-tertile draws, nominal / actual / Monte Carlo | Predicted contrast change, nominal / actual |", "| --- | ---: | --- | --- |"]
    for k in PRIMARY_SUPP:
        for e in ("-1.0", "0.5", "1.0"):
            v = d[k][e]
            out.append(f"| {CL[k]} | {s(float(e), 1)} | {u(v['expected_upper_nominal'], 1)} / {u(v['expected_upper_true'], 1)} / {u(v['mc_upper'], 1)} | "
                       f"{s(v['pred_contrast_nominal'], 3)} / {s(v['pred_contrast_true'], 3)} |")
    return "\n".join(out)


def t_eiv27():
    d = J("vr27_bridge_eiv.json")
    out = ["| Features | Least-squares slope on log ĝ_A [95% CI] | Instrumented slope [95% CI] | Reliability of log ĝ [95% CI] |", "| --- | --- | --- | --- |"]
    for f in FAMS2:
        e = d["families"][f]
        out.append(f"| {f} | {u(e['ols']['median'])} {ci(e['ols']['ci95'])} | {u(e['iv']['median'])} {ci(e['iv']['ci95'])} | {u(e['reliability']['median'])} {ci(e['reliability']['ci95'])} |")
    return "\n".join(out)


def t_tau44():
    d = J("vr44_dose_estimand.json")["families"]
    out = ["| Features | Concept | Mean Δ at η = −1 / η = 1 | τ̂(1, −1) [95% CI] | Predicted | Observed / predicted | Replicates changing sign |",
           "| --- | --- | --- | --- | ---: | ---: | ---: |"]
    for f in FAMS2:
        for c in PRIMARY_SUPP:
            e = d[f][c]
            out.append(f"| {f} | {CL[c]} | {s(e['delta_lo_mean'])} / {s(e['delta_hi_mean'])} | {s(e['tau_hat'])} {ci(e['tau_ci95'])} | "
                       f"{s(e['tau_pred'])} | {u(e['tau_hat'] / e['tau_pred'])} | {e['n_sign_change']} of 20 |")
    return "\n".join(out)


def t_mc43():
    """Kiểm toán Monte Carlo của bootstrap ba seed chính (vr43): sai số MC của đầu mút Bonferroni, tỉ lệ tái lập nhãn."""
    d = J("vr43_three_seed_primary.json")
    out = ["| Features | Concept | Replicates | Bonferroni sign-separated | Label reproduced | Monte Carlo SE of endpoints, Δ_M0 lower / upper, Δ_M2 lower / upper | Resolved at the Monte Carlo resolution |",
           "| --- | --- | ---: | --- | ---: | --- | --- |"]
    for f in FAMS2:
        for c in PRIMARY_SUPP:
            e = d["families"][f]["concepts"][c]; m = e["mc_se_endpoints"]
            out.append(f"| {f} | {CL[c]} | {d['B'][f]:,} | {'yes' if e['robust_bonf'] else 'no'} | {u(e['label_stability'])} | "
                       f"{u(m[0], 3)} / {u(m[1], 3)}, {u(m[2], 3)} / {u(m[3], 3)} | {'yes' if e['label_stability'] >= 0.95 else 'no'} |")
    return "\n".join(out)


def t_mc29():
    d = J("vr29_bootstrap_audit.json")["families"]
    out = ["| Features | Concept | Robust | Label reproduced | Monte Carlo SE of endpoints, Δ_M0 lower / upper, Δ_M2 lower / upper | s∗ quantiles 5 / 25 / 50 / 75 / 95 | ψ_L(s) > 0 and Δ_M2 < 0 at s = 0.5 / 0.7 / 0.9 |",
           "| --- | --- | --- | ---: | --- | --- | --- |"]
    fl = lambda x: "none" if x is None else u(x)
    for f in FAMS2:
        for c in PRIMARY_SUPP:
            e = d[f][c]; m = e["mc_se_endpoints"]
            out.append(f"| {f} | {CL[c]} | {'yes' if e['robust_label'] else 'no'} | {u(e['label_stability'])} | "
                       f"{u(m['d0_lower'], 3)} / {u(m['d0_upper'], 3)}, {u(m['d2_lower'], 3)} / {u(m['d2_upper'], 3)} | "
                       + " / ".join(fl(e["sstar_quantiles"][q]) for q in ("5", "25", "50", "75", "95")) + " | "
                       + " / ".join(u(e["joint_psiL_pos_d2_neg"][s_]) for s_ in ("0.5", "0.7", "0.9")) + " |")
    return "\n".join(out)


def t_psisens25():
    d = J("vr25_psi_sensitivity.json")["families"]
    lab = [("raw", "uncalibrated M0"), ("platt", "Platt"), ("iso", "isotonic"), ("beta", "beta calibration"), ("sigma_learner", "Platt, σ from the learner's input"),
           ("winsor", "Platt, winsorized at 1 and 99 percent")]
    out = ["| Estimate of *q* | " + " | ".join(f"{f}, {CL[c].split()[0].lower().replace('lesion-skin', 'contrast')}" for f in FAMS2 for c in PRIMARY_SUPP) + " |",
           "| --- | " + " | ".join("---:" for _ in range(6)) + " |"]
    for k, l in lab:
        out.append(f"| {l} | " + " | ".join(("none" if d[f][c]["s_star"][k] != d[f][c]["s_star"][k] else u(d[f][c]["s_star"][k])) for f in FAMS2 for c in PRIMARY_SUPP) + " |")
    return "\n".join(out)


def t_twofloor25():
    d = J("vr25_psi_sensitivity.json")
    out = ["| Features | Concept | Low-support region | s_low needed at s_high = 0.5 / 0.6 / 0.7 / 0.8 / 0.9 / 1.0 |", "| --- | --- | --- | --- |"]
    for f in FAMS2:
        for c in PRIMARY_SUPP:
            for q, share in d["low_share"].items():
                b = d["families"][f][c]["two_floor"][q]["boundary_s_low_by_s_high"]
                out.append(f"| {f} | {CL[c]} | below the {round(100 * float(q))}{'st' if q == '0.01' else 'th'} percentile, {round(100 * share)} percent of test lesions | "
                           + " / ".join("none" if b[s_] is None else u(b[s_]) for s_ in ("0.5", "0.6", "0.7", "0.8", "0.9", "1.0")) + " |")
    return "\n".join(out)


PRIMARY_SUPP = ["color_variegation", "size", "lesion_skin_contrast"]


TABLES = {"theory": t_theory, "definitions": t_definitions, "size_matched": t_size_matched, "quant": t_quant,
          "pad": t_pad, "headswap": t_headswap, "icdl": t_icdl, "repr": t_repr, "dose": t_dose, "bridge": t_bridge,
          "within": t_within, "sites": t_sites, "overlap": t_overlap, "joint": t_joint, "joint19": t_joint19, "psi": t_psi,
          "support19": t_support19, "dose21": t_dose21, "ess21": t_ess21, "arms21": t_arms21, "pointwise22": t_pointwise22,
          "sitehet": t_sitehet, "loso": t_loso, "psisim": t_psisim, "psistress": t_psistress, "full24": t_full24, "full30": t_full30, "qdiag30": t_qdiag30, "marginal35": t_marginal35, "sub36": t_sub36, "dose32": t_dose32, "dose46": t_dose46, "spill46": t_spill46, "spill32": t_spill32, "bridge33": t_bridge33, "semi34": t_semi34, "seed39": t_seed39, "theory42": t_theory42, "sharp42": t_sharp42, "comp40": t_comp40, "partial40": t_partial40, "site41": t_site41, "local34": t_local34, "semibridge34": t_semibridge34, "sigma37": t_sigma37, "tail38": t_tail38, "dose26": t_dose26, "lc26": t_lc26, "gumbel26": t_gumbel26, "eiv27": t_eiv27, "mc29": t_mc29, "mc43": t_mc43, "tau44": t_tau44, "psisens25": t_psisens25, "twofloor25": t_twofloor25}


def n_platt():
    rng = {}
    for f in FAMS2:
        pl = [v["platt"] for v in J(f"vr19/{f}_-3_0.json")["reps"].values()]
        rng[f] = [(min(x[i] for x in pl), max(x[i] for x in pl)) for i in (0, 1)]
    r = lambda a: f"{u(a[0])} to {u(a[1])}"
    return (f"Platt slopes over seeds 0 to 2 were {r(rng['tabular'][0])} for tabular M0 and {r(rng['tabular'][1])} for tabular M2, "
            f"so tabular learners were close to calibrated. For image M0 they were {r(rng['image'][0])} and for image M2 "
            f"{r(rng['image'][1])}, so the image learners, and image M0 in particular, were overconfident.")


def n_bootstrap_count():
    d = J("vr19_primary_bootstrap.json")["B"]
    return f"{min(d.values()):,}"


def n_B30s():
    return f"{min(J('vr30_q_bootstrap.json')['B'].values()):,}"


def n_semi_bridge():
    return ("Table {{T:semibridge34}} reports the lesion-level slope of the calibrated gap on log ĝ in the same settings, where the "
            "population identity holds exactly. Finite trained learners can depart from slope one even then, and the model class of ĝ "
            "moves the slope. We therefore read the slopes on ISIC-2024 as agreement in direction and approximate size only.")


def n_B_primary():
    return f"{min(J('vr43_three_seed_primary.json')['B'].values()):,}"


NUMS = {"B_primary": n_B_primary, "B30s": n_B30s, "semi_bridge": n_semi_bridge, "platt": n_platt, "B19": n_bootstrap_count}


def main():
    src = open(os.path.join(HERE, "supplementary_src.md"), encoding="utf-8").read()
    out = re.sub(r"\{\{TABLE:(\w+)\}\}", lambda m: TABLES[m.group(1)](), src)
    out = re.sub(r"\{\{N:(\w+)\}\}", lambda m: NUMS[m.group(1)](), out)
    # số bảng tự động theo thứ tự chú thích "**TABLE {{T:key}}."
    order = re.findall(r"^\*\*TABLE \{\{T:(\w+)\}\}\.", out, flags=re.M)
    assert len(order) == len(set(order)), "khóa bảng trùng"
    num = {k: f"S{i + 1}" for i, k in enumerate(order)}
    out = re.sub(r"\{\{T:(\w+)\}\}", lambda m: num[m.group(1)], out)
    open(os.path.join(HERE, "supplementary.md"), "w", encoding="utf-8").write(out)
    print("built supplementary.md")


if __name__ == "__main__":
    main()
