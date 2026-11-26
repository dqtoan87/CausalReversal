#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sentences and tables of the main text that depend on results, generated from the JSON outputs so that no number
is typed by hand, and so that the wording follows the results. Used by build_paper.py through {{N:key}} and
{{TABLE:key}}."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "Result")
CL = {"color_variegation": "Color variegation", "size": "Size", "lesion_skin_contrast": "Lesion-skin contrast",
      "asymmetry": "Asymmetry", "border_irregularity": "Border irregularity"}
PRIMARY = ["color_variegation", "size", "lesion_skin_contrast"]
SECONDARY = ["asymmetry", "border_irregularity"]
FAMS = ("tabular", "image")


def J(n):
    return json.load(open(os.path.join(RES, n)))


def s(x, d=2):
    return f"{x:+.{d}f}".replace("-", "−")


def u(x, d=2):
    return f"{x:.{d}f}".replace("-", "−")


def ci(a, d=2):
    return f"[{u(a[0], d)}, {u(a[1], d)}]"


def lower(name):
    return name[0].lower() + name[1:]


def pct(x):
    return f"{round(100 * x)} percent"


def listing(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# ------------------------------------------------------------------ primary inference (vr19)
def _v19():
    return J("vr19_primary_bootstrap.json")


def _vp():
    """Suy luận chính của Table 1: bootstrap trung bình ba seed (vr43)."""
    return J("vr43_three_seed_primary.json")


def t_main_reversal():
    d = _vp()
    out = ["| Block | Concept | Features | Δ_M0 [interval] | Δ_M2 [interval] | Δ_M0 minus Δ_M2 [95% CI] | Same-sign resamples | Bonferroni sign-separated |",
           "| --- | --- | --- | --- | --- | --- | ---: | --- |"]
    for block, cs in (("Primary", PRIMARY), ("Secondary", SECONDARY)):
        for fam in FAMS:
            B = d["B"][fam]
            for c in cs:
                e = d["families"][fam]["concepts"][c]
                k0, k2 = ("d0_ci_bonf", "d2_ci_bonf") if block == "Primary" else ("d0_ci95", "d2_ci95")
                rob = e["robust_bonf"] if block == "Primary" else e["robust_95"]
                out.append(f"| {block} | {CL[c]} | {fam} | {s(e['d0_orig'])} {ci(e[k0])} | {s(e['d2_orig'])} {ci(e[k2])} | "
                           f"{s(e['d0_orig'] - e['d2_orig'])} {ci(e['diff_ci95'])} | {e['n_same_sign']:,} of {B:,} | "
                           f"{('unresolved (' + ('yes' if rob else 'no') + ' in the full set)') if block == 'Primary' and e['label_stability'] < 0.95 else ('yes' if rob else 'no')} |")
    return "\n".join(out)


def _robust_cells():
    d = _vp()
    return [(fam, c) for fam in FAMS for c in PRIMARY if d["families"][fam]["concepts"][c]["robust_bonf"]]


def B_primary():
    return f"{min(_vp()['B'].values()):,}"


def B19():
    return f"{min(_v19()['B'].values()):,}"


def mc_sentence():
    a = J("vr29_bootstrap_audit.json")["families"]
    st = [a[f][c]["label_stability"] for f in FAMS for c in PRIMARY]
    se = [v for f in FAMS for c in PRIMARY for v in a[f][c]["mc_se_endpoints"].values()]
    return (f"Resampling the replicates themselves, the Monte Carlo standard error of the Bonferroni endpoints was at most {u(max(se))}, "
            f"and each robustness label was reproduced in at least {pct(min(st))} of resamples.")


def val_sentence():
    d19, d24 = _v19(), _v24()
    lab = lambda e: (e["robust_bonf"], e["robust_95"])
    same = all(lab(d19["families"][f]["concepts"][c]) == lab(d24["families"][f]["concepts"][c]) for f in FAMS for c in CL)
    return ("With validation patients also resampled, every robustness label in Table 1 was unchanged." if same else
            "With validation patients also resampled, some robustness labels changed (Supplementary Section S6).")


def abs_robust():
    rc = _robust_cells()
    col = [f for f, c in rc if c == "color_variegation"]
    if len(col) == 2:
        return ", robustly after multiplicity adjustment in both primary learner families"
    if len(col) == 1:
        return f", robustly after multiplicity adjustment with {col[0]} features"
    return ", although the reversal was not robust after multiplicity adjustment"


def _unresolved():
    d = _vp()
    return [(f, c, d["families"][f]["concepts"][c]["label_stability"]) for f in FAMS for c in PRIMARY
            if d["families"][f]["concepts"][c]["label_stability"] < 0.95]


def abs_inference():
    """Câu tóm tắt về suy luận chính, sinh từ vr43 (nhãn chính), vr30 (lấy lại cả val) và kiểm toán Monte Carlo."""
    d30 = _v30()
    ur = {(f, c) for f, c, _ in _unresolved()}
    rc = _robust_cells()
    nm = lambda f, c: f"{f.replace('image', 'image-feature').replace('tabular', 'tabular-feature')} {lower(CL[c])}"
    every = [(f, c) for f, c in rc if (f, c) not in ur and d30["families"][f]["concepts"][c]["robust_bonf"]]
    fixed = [(f, c) for f, c in rc if (f, c) not in ur and not d30["families"][f]["concepts"][c]["robust_bonf"]]
    parts = []
    if every:
        parts.append(listing([nm(f, c) for f, c in every]) + " under every resampling scheme")
    if fixed:
        parts.append(listing([nm(f, c) for f, c in fixed]) + " with validation fixed")
    txt = ("Within a post-inspection family of six comparisons, Bonferroni-adjusted intervals separated the signs for "
           + " and for ".join(parts))
    if ur:
        txt += ". " + (lambda x: x[0].upper() + x[1:])(listing([nm(f, c) for f, c in ur])) + " was unresolved under Monte Carlo error"
    return txt + "."


def tau_sentence():
    d = _v46()
    c = {f: [d[f][f"{k}|320"]["raw"] for k in PRIMARY] for f in FAMS}
    hi_ci = max(x["tau_ci95"][1] for f in FAMS for x in c[f])
    assert hi_ci < 0
    t_ = (min(x["tau_hat"] for x in c["tabular"]), max(x["tau_hat"] for x in c["tabular"]))
    i_ = (min(x["tau_hat"] for x in c["image"]), max(x["tau_hat"] for x in c["image"]))
    return (f"The estimated training-selection effect τ̂_k(1, −1) of Section III-B was between {s(t_[0])} and {s(t_[1])} with "
            f"tabular features and between {s(i_[0])} and {s(i_[1])} with image features, and every 95 percent interval lay below "
            f"{s(hi_ci)}. These intervals are bootstrap intervals over the 20 paired replicates, conditional on the fixed cohort, split, "
            f"validation set and test population, so they describe the algorithmic randomness ω and not patient sampling "
            f"(Supplementary Section S6).")


def mc_rule():
    return ("Because the Bonferroni endpoints lie far in the tails, we resampled the stored replicates 2,000 times. We call a label "
            "resolved at the Monte Carlo resolution when at least 95 percent of these resamples reproduce it.")


def mc_text():
    d = _vp()
    se = max(v for f in FAMS for c in PRIMARY for v in d["families"][f]["concepts"][c]["mc_se_endpoints"])
    ok = [d["families"][f]["concepts"][c]["label_stability"] for f in FAMS for c in PRIMARY
          if d["families"][f]["concepts"][c]["label_stability"] >= 0.95]
    word = {5: "five", 6: "six", 4: "four"}
    return (f"The Monte Carlo standard error of the Bonferroni endpoints was at most {u(se)}, and the other {word[len(ok)]} "
            f"primary labels were reproduced in " + ("every Monte Carlo resample." if min(ok) == 1 else f"at least {pct(min(ok))} of Monte Carlo resamples."))


def table1_mc():
    d = _vp()
    se = max(v for f in FAMS for c in PRIMARY for v in d["families"][f]["concepts"][c]["mc_se_endpoints"])
    ur = _unresolved()
    ok = [d["families"][f]["concepts"][c]["label_stability"] for f in FAMS for c in PRIMARY
          if d["families"][f]["concepts"][c]["label_stability"] >= 0.95]
    txt = (f"Resampling the stored replicates, the Monte Carlo standard error of the endpoints was at most {u(se)}, and "
           f"{len(ok)} of the 6 primary labels were reproduced in " + ("every" if min(ok) == 1 else f"at least {pct(min(ok))} of the") + " Monte Carlo resample" + ("" if min(ok) == 1 else "s"))
    if ur:
        txt += ("; the label of " + listing([f"{lower(CL[c])} with {f} features ({pct(v)})" for f, c, v in ur])
                + " is marked unresolved at this Monte Carlo resolution")
    return txt + " (Supplementary Section S6)."


def jb_sentence():
    d = _vp(); d19 = _v19()
    B = min(d["B"].values())
    ur = _unresolved(); urk = {(f, c) for f, c, _ in ur}
    rc = [x for x in _robust_cells() if x not in urk]
    assert rc
    word = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    cells = (listing([lower(CL[c]) for _, c in rc]) + f", both with {rc[0][0]} features" if len(rc) == 2 and rc[0][0] == rc[1][0]
             else listing([f"{lower(CL[c])} with {f} features" for f, c in rc]))
    txt = [f"The primary rule uses {B:,} joint resamples per family of the three-seed average. Under it, {word[len(rc)]} of the six "
           f"primary cells showed a Bonferroni sign-separated learner reversal with a resolved label: {cells}."]
    nrest = 6 - len(rc) - len(ur)
    if nrest:
        txt.append(f"In {word[nrest]} other cells, whose labels were also resolved, the Bonferroni interval of Δ_M0 reached zero once "
                   f"training variability was included.")
    for f, c, v in ur:
        e = d["families"][f]["concepts"][c]
        txt.append(f"The remaining cell, {lower(CL[c])} with {f} features, sits on the boundary. Its lower Bonferroni endpoint of Δ_M0 "
                   f"was {s(e['d0_ci_bonf'][0])}, so the two intervals {'were' if e['robust_bonf'] else 'were not'} sign-separated in the "
                   f"full set of replicates. Yet this label was reproduced in only {pct(v)} of Monte Carlo resamples, and we report it as "
                   f"unresolved.")
    se = max(v for f in FAMS for c in PRIMARY for v in d["families"][f]["concepts"][c]["mc_se_endpoints"])
    ok = [d["families"][f]["concepts"][c]["label_stability"] for f in FAMS for c in PRIMARY if d["families"][f]["concepts"][c]["label_stability"] >= 0.95]
    txt.append(f"The other {word[len(ok)]} labels were reproduced in " + ("every" if min(ok) == 1 else f"at least {pct(min(ok))} of the")
               + f" Monte Carlo resample{'' if min(ok) == 1 else 's'}, and the Monte Carlo standard error of the endpoints was at most {u(se)}.")
    gaps = sum(1 for f in FAMS for c in CL
               if (lambda e: e["diff_ci95"][0] > 0 or e["diff_ci95"][1] < 0)(d["families"][f]["concepts"][c]))
    txt.append(f"The gap Δ_M0 − Δ_M2 was more stable than the full reversal: its 95 percent interval excluded zero in "
               + ("all 10" if gaps == 10 else f"{gaps} of the 10") + " primary and secondary cells.")
    return " ".join(txt)


def _v24():
    return J("vr24_full_bootstrap.json")


def _v25():
    return J("vr25_psi_sensitivity.json")


def _nn(x):
    return None if x is None or x != x else x


def abs_sstar():
    d = _v24()
    v = [_nn(d["families"][f]["concepts"]["color_variegation"]["s_star_platt"]["lcb_floor"]) for f in FAMS]
    assert all(x is not None for x in v), "sàn cận dưới không xác định; sửa câu tóm tắt"
    return u(max(v))


def psi_sentence():
    d19, d24, d25 = _v19(), _v24(), _v25()
    c = "color_variegation"
    pt = {f: d24["families"][f]["concepts"][c]["s_star_platt"]["orig_mean_curve"] for f in FAMS}
    q95 = {f: d19["families"][f]["concepts"][c]["s_pos_q95"] for f in FAMS}
    lcb = {f: d24["families"][f]["concepts"][c]["s_star_platt"]["lcb_floor"] for f in FAMS}
    jt = {f: d24["families"][f]["concepts"][c]["s_star_platt"]["joint_psiL_pos_and_d2_neg"]["0.8"] for f in FAMS}
    B = min(d19["B"].values())
    txt = [f"For color variegation, with the Platt-calibrated M0 as the estimate of *q*, the lower endpoint of the plug-in set became "
           f"positive at a floor of {u(pt['tabular'])} with tabular features and {u(pt['image'])} with image features.",
           f"Across {B:,} joint resamples of training and test patients, the 95th percentile of this plug-in tipping floor was "
           f"{u(q95['tabular'])} and {u(q95['image'])}. When validation patients were also resampled, the smallest floor at which the "
           f"one-sided 95 percent percentile bound of ψ_L was positive was " + (f"{u(lcb['tabular'])} in both families." if lcb['tabular'] == lcb['image']
           else f"{u(lcb['tabular'])} and {u(lcb['image'])}."),
           f"At a floor of 0.8, ψ_L was positive and Δ_M2 negative in the same resample, which is the disease-relative reversal, in "
           + (f"{pct(jt['tabular'])} of these resamples in both families." if round(100 * jt['tabular']) == round(100 * jt['image'])
            else f"{pct(jt['tabular'])} and {pct(jt['image'])} of these resamples.")]
    ss = {f: d25["families"][f][c]["s_star"] for f in FAMS}
    cal = {f: [ss[f][k] for k in ("platt", "iso", "beta", "sigma_learner", "winsor")] for f in FAMS}
    txt.append(f"The tipping floor depends on how *q* is estimated. Across Platt, isotonic and beta calibration, a propensity model on "
               f"the learner's input for σ and winsorized tails, it ranged from {u(min(cal['tabular']))} to {u(max(cal['tabular']))} with "
               f"tabular features and from {u(min(cal['image']))} to {u(max(cal['image']))} with image features. A gradient-boosting "
               f"estimate of *q* on the same input raised it to {u(ss['tabular']['altq'])} and {u(ss['image']['altq'])}, and the "
               f"uncalibrated image learner, which is overconfident, lowered it to {u(ss['image']['raw'])}.")
    sup = {f: [_nn(d24["families"][f]["concepts"][c]["s_star_support"][q]["orig"]) for q in ("0.01", "0.05", "0.1")] for f in FAMS}
    def fmt(v):
        got = [(q, x) for q, x in zip(("1st", "5th", "10th"), v) if x is not None]
        miss = [q for q, x in zip(("1st", "5th", "10th"), v) if x is None]
        a = listing([u(x) for _, x in got]) + " at the " + listing([q for q, _ in got]) + (" percentile threshold" if len(got) == 1 else " percentile thresholds") if got else ""
        b = "was not reached below one at the " + listing(miss) if miss else ""
        return a + (" and " + b if a and b else b)
    txt.append(f"Restricted to the region supported by verified training lesions, the tipping floor rose to {fmt(sup['tabular'])} with "
               f"tabular features, and to {fmt(sup['image'])} with image features.")
    tw = {f: d25["families"][f][c]["two_floor"]["0.01"]["boundary_s_low_by_s_high"]["0.9"] for f in FAMS}
    share = d25["low_share"]["0.01"]
    txt.append(f"The floor need not be uniform, however. If malignant verification had a floor of 0.9 elsewhere, a floor of "
               f"{u(tw['tabular'])} with tabular features and {u(tw['image'])} with image features sufficed among the {pct(share)} of test "
               f"lesions below the 1st percentile threshold.")
    txt.append("Above such floors, the negative contrast learned by M2 is a disease-relative reversal on the learner's scale. "
               "The floors are assumption-indexed and estimator-dependent thresholds, not quantities this cohort identifies.")
    return " ".join(txt)


def _ret():
    return J("vr45_support_retention.json")["retained"]


def _ord(q):
    n = round(100 * float(q))
    return f"{n}{'st' if n % 10 == 1 and n != 11 else 'th'}"


def support_sentence():
    d = _v19()
    ret = _ret()
    qs = sorted(ret, key=float)
    n_m2 = n_cells = 0
    shrink, m0zero = [], 0
    for fam in FAMS:
        for c in PRIMARY:
            e = d["families"][fam]["concepts"][c]
            for q in qs:
                v = e["support"][q]; n_cells += 1
                n_m2 += v["d2_ci95"][1] < 0
                shrink.append(v["d0_orig"] < e["d0_orig"])
                m0zero += v["d0_ci95"][0] <= 0
    return (f"Under the common tabular estimate, whose thresholds kept {listing([f'{round(100 * ret[q])}' for q in qs])} percent of "
            f"test lesions, joint intervals are available: the M2 contrast stayed below zero in {n_m2} of the {n_cells} combinations "
            f"of primary comparison and threshold, whereas the M0 contrast was smaller than on the full population in {sum(shrink)} and "
            f"its interval reached zero in {m0zero} (Table 2 and Supplementary Section S6).")


# ------------------------------------------------------------------ dose and arms (vr21)
def _v21():
    return J("vr21_dose_calibrated.json")


def _v26():
    return J("vr26_dose_poisson.json")


def dose_sentence():
    d = _v26(); d21 = _v21()
    cc = {f: d["families"][f]["dose"] for f in FAMS}
    rat = {f: [cc[f][k]["concepts"][k]["ratio"] for k in PRIMARY] for f in FAMS}
    col = {f: cc[f]["color_variegation"]["concepts"]["color_variegation"] for f in FAMS}
    txt = [f"For η from −1 to 1, the calibrated contrast of the selected concept fell at {u(min(rat['tabular']))} to "
           f"{u(max(rat['tabular']))} times the predicted rate with tabular features and at {u(min(rat['image']))} to "
           f"{u(max(rat['image']))} times with image features, over the three primary concepts.",
           f"For color variegation the falls were {u(-col['tabular']['obs_slope'])} per unit dose "
           f"{ci([-x for x in col['tabular']['obs_range95'][::-1]])} and {u(-col['image']['obs_slope'])} "
           f"{ci([-x for x in col['image']['obs_range95'][::-1]])}, against {u(-col['tabular']['pred_slope'])} and "
           f"{u(-col['image']['pred_slope'])} predicted from each learner's input."]
    lc = {f: {n: d["families"][f]["learning_curve"][n]["concepts"]["color_variegation"]["ratio"] for n in ("320", "1000", "3000")} for f in FAMS}
    mono = lc["tabular"]["320"] > lc["tabular"]["1000"] > lc["tabular"]["3000"]
    assert mono, "độ vượt của learner bảng không giảm đơn điệu; sửa câu"
    txt.append(f"With tabular features the excess shrank as more recorded negatives were selected, to a ratio of "
               f"{u(lc['tabular']['1000'])} with 1,000 and {u(lc['tabular']['3000'])} with 3,000, which points to a finite-sample effect "
               f"rather than a failure of the identity. With image features the ratio was {u(lc['image']['1000'])} and "
               f"{u(lc['image']['3000'])}, with no trend.")
    dev = col["image"]["pred_range95"][1] - col["image"]["pred_range95"][0]
    assert max(abs(cc["image"][k]["concepts"][k]["pred2_slope"] - cc["image"][k]["concepts"][k]["pred_slope"]) for k in PRIMARY) < 0.01
    txt.append(f"The image prediction rests on a regression of the concept on the image features. Refitting it on resampled "
               f"training patients moved the predicted slope within a range of {u(dev)}, and the variance term of the Gaussian "
               f"approximation changed it by less than 0.01. Treating the concept as fully visible would have predicted a fall of "
               f"{u(-d21['families']['image']['dose']['color_variegation']['predicted_slope_zk'])}, twice the observed one.")
    n_cross = sum(1 for f in FAMS for k in PRIMARY if cc[f][k]["crossing"] is not None)
    ag = sum(cc[f][k]["spillover_sign_agreement"] for f in FAMS for k in PRIMARY)
    txt.append(f"The calibrated contrast of the selected concept crossed zero in {n_cross} of 6 combinations of concept and family, and "
               f"the contrasts of all five concepts moved in the direction predicted from their association with the selected one in "
               f"{ag} of 30 cases.")
    return " ".join(txt)


def abs_dose():
    d = _v26()
    r = [d["families"][f]["dose"][k]["concepts"][k]["ratio"] for f in FAMS for k in PRIMARY]
    return f"{u(min(r))} to {u(max(r))}"


def contrib_dose():
    d = _v26()
    r = [d["families"][f]["dose"][k]["concepts"][k]["ratio"] for f in FAMS for k in PRIMARY]
    return pct(max(abs(x - 1) for x in r))


def arms_sentence():
    d = _v21()
    cells = [(f, a) for f in FAMS for a in ("ver", "flag")]
    ag = sum(d["families"][f]["arms"][a]["summary"]["sign_agreement"] for f, a in cells)
    mad = {(f, a): d["families"][f]["arms"][a]["summary"]["mean_abs_diff"] for f, a in cells}
    lab = {"ver": "verified", "flag": "flagged"}
    lo, hi = min(mad, key=mad.get), max(mad, key=mad.get)
    return (f"Proposition 1 predicted the sign of the shift from the random arm in {ag} of {5 * len(cells)} cases, with mean "
            f"absolute differences in size from {u(mad[lo])} to {u(mad[hi])}.")


# ------------------------------------------------------------------ pointwise bridge (vr22)
def _v22():
    return J("vr22_pointwise_bridge.json")


def bridge_sentence():
    d = _v22(); e = J("vr27_bridge_eiv.json")["families"]
    t_, i_ = d["families"]["tabular"], d["families"]["image"]
    sg = lambda f: sum(1 for c in CL if f["concepts"][c]["sign_agreement_frac"] >= 0.975)
    return (f"Lesion by lesion, the calibrated logit gap between M0 and M2 rose with the calibrated log propensity log ĝ, with "
            f"least-squares slope {u(t_['pointwise_slope'])} {ci(t_['pointwise_slope_ci'])} for tabular features and "
            f"{u(i_['pointwise_slope'])} {ci(i_['pointwise_slope_ci'])} for image features and median R² of {u(t_['pointwise_r2'])} "
            f"and {u(i_['pointwise_r2'])}. Because log ĝ is estimated, these slopes are attenuated. With a second propensity model "
            f"fitted on disjoint training patients as an instrument, the slope was {u(e['tabular']['iv']['median'])} "
            f"{ci(e['tabular']['iv']['ci95'])} and {u(e['image']['iv']['median'])} {ci(e['image']['iv']['ci95'])}, against 1 for "
            f"Bayes-optimal learners. The regression is thus an association diagnostic. After correction both point estimates "
            f"exceeded one, in line with the excess response of tabular learners in the dose experiment. For concept contrasts, the predicted gap had the observed sign in at least "
            f"97.5 percent of {d['B']} joint resamples for {sg(t_)} of 5 concepts with tabular features and {sg(i_)} of 5 with image "
            f"features.")


# ------------------------------------------------------------------ sites (vr20) and simulation (vr23)
def site_sentence():
    a = J("vr20_audit.json")
    h = a["site_heterogeneity"]["color_variegation"]
    assert h["range"][0] > 0, "log B_V không dương ở mọi cơ sở; sửa câu"
    txt = (f"The verified-benign contrast log *B*_V for color variegation was positive in all {h['k']} acquisition sites where it "
           f"could be estimated. Its pooled value was {u(h['pooled_DL'])} {ci(h['pooled_ci'])}, with substantial heterogeneity "
           f"(I² = {u(h['I2'])}). Two further sites were not estimable (Supplementary Section S6).")
    if "leave_one_site_out" in a:
        lo = a["leave_one_site_out"]
        mn0 = min(v["color_variegation"][0] for f in lo for v in lo[f].values())
        mx2 = max(v["color_variegation"][1] for f in lo for v in lo[f].values())
        keep = mn0 > 0 and mx2 < 0
        assert keep
        txt += " Removing any single site left both signs unchanged."
    return txt


def psi_sim_sentence():
    v = J("vr23_psi_sim.json")["summary"]["valid_floor"]
    st = J("vr28_psi_stress.json")["summary"]
    est_cov = [v["plugin_covers"]] + [st[f"{sc}|valid"][f"{m}_covers"] for sc in ("base", "adversarial") for m in ("gbm", "mlp_platt")]
    wrong_valid = max(st[f"{sc}|valid"][f"{m}_wrong"] for sc in ("base", "adversarial") for m in ("oracle", "gbm", "mlp_platt"))
    wrong_valid = max(wrong_valid, v["oracle_wrong_sign"], v["plugin_wrong_sign"])
    adv = st["adversarial|overstated"]
    w_or, w_est = adv["oracle_wrong"], max(adv["gbm_wrong"], adv["mlp_platt_wrong"])
    assert wrong_valid == 0, "có chứng nhận sai dấu dưới sàn hợp lệ; sửa câu"
    return (f"In simulation with a valid floor, the identified set of Proposition 2 contained the true disease target in "
            f"{pct(st['base|valid']['oracle_covers'])} of settings when built from true nuisances and in {round(100 * min(est_cov))} to "
            f"{pct(max(est_cov))} when built from estimated ones, including a perceptron with Platt calibration, and no setting certified "
            f"the wrong sign. With an overstated floor, in an adversarial design where malignant verification rose steeply with concepts "
            f"whose disease effect was zero or negative, the set certified the wrong sign in {pct(w_or)} of settings from true nuisances "
            f"and in up to {pct(w_est)} from estimated ones. The conclusions about disease therefore depend on the floor holding.")


# ------------------------------------------------------------------ Table 2 rows
def table2_rows():
    rows = []
    d21 = _v21()
    rows.append(table2_dose_rows())
    for fam in FAMS:
        a = d21["families"][fam]["arms"]["ver"]["summary"]
        rows.append(f"| Verified against random recorded negatives, predicted shift | {fam}; five concepts | "
                    f"sign {a['sign_agreement']} of 5; calibration slope {u(a['calibration_slope'])} |")
    d22 = _v22()
    for fam in FAMS:
        f = d22["families"][fam]
        ev = J("vr27_bridge_eiv.json")["families"][fam]["iv"]
        rows.append(f"| Proposition 1, lesion-level gap on log ĝ | {fam}; calibrated; least squares, then split-sample errors-in-variables diagnostic | "
                    f"{u(f['pointwise_slope'])} {ci(f['pointwise_slope_ci'])}; {u(ev['median'])} {ci(ev['ci95'])} |")
    d19 = _v19()
    for fam in FAMS:
        e = d19["families"][fam]["concepts"]["color_variegation"]["support"]
        cells = "; ".join(f"{_ord(q)} percentile {s(v['d0_orig'])} against {s(v['d2_orig'])}"
                          for q, v in sorted(e.items(), key=lambda z: float(z[0])))
        rows.append(f"| Overlap sensitivity, color variegation, common tabular σ̂ | {fam}; Δ_M0 against Δ_M2 | {cells} |")
    rows.append("| PAD-UFES-20 learners | six symptoms / image color | 6 of 6 attenuated, P(signs differ) at most 0.04 / +2.06 [1.47, 2.84] to +0.70 [0.22, 1.23] |")
    return "\n".join(rows)



# ------------------------------------------------------------------ round five: estimators of q (vr30), marginal (vr35), subsampling (vr36)
EST = ("mlp_platt", "mlp_iso", "gbm_platt")
EST_LAB = {"mlp_platt": "perceptron, Platt", "mlp_iso": "perceptron, isotonic", "gbm_platt": "gradient boosting, Platt"}


def _v30():
    return J("vr30_q_bootstrap.json")


def _fl(x):
    return "none" if x is None or x != x else u(x)


def B30():
    return f"{min(_v30()['B'].values()):,}"


def val_sentence():
    d19, d30 = _v19(), _v30()
    lab = lambda e: (e["robust_bonf"], e["robust_95"])
    same = all(lab(d19["families"][f]["concepts"][c]) == lab(d30["families"][f]["concepts"][c]) for f in FAMS for c in CL)
    return (f"With validation patients also resampled, in {B30()} replicates per family, every robustness label in Table 1 was unchanged." if same else
            "With validation patients also resampled, some robustness labels changed (Supplementary Section S6).")


def subsample_sentence():
    d = J("vr36_subsample.json")["families"]; d19 = _v19()
    lab = lambda e: 2 if e["robust_bonf"] else (1 if e["robust_95"] else 0)
    down = [f"{lower(CL[c])} with {f} features" for f in FAMS for c in PRIMARY if lab(d[f][c]) < lab(d19["families"][f]["concepts"][c])]
    txt = f"Subsampling half of the patients without replacement, in {d['tabular']['B']} replicates per family, "
    txt += "reversed no bootstrap-robust label." if not down else "lost the label for " + listing(down) + "."
    return txt + (" We use it only as this check, because its rescaling assumes a root-n rate that fitted networks need not follow "
                  "(Supplementary Section S6).")

def psi_primary_sentence():
    e = {f: _v30()["families"][f]["concepts"]["color_variegation"]["mlp_platt"] for f in FAMS}
    B = {f: _v30()["B"][f] for f in FAMS}
    lf = (f"{_fl(e['tabular']['lcb_floor'])} for both targets" if e['tabular']['lcb_floor'] == e['image']['lcb_floor']
          else f"{_fl(e['tabular']['lcb_floor'])} and {_fl(e['image']['lcb_floor'])}")
    return (f"With the primary plug-in, the Platt-calibrated M0, the lower endpoint for color variegation became positive at a floor of "
            f"{u(e['tabular']['point'])} for the tabular-input target and {u(e['image']['point'])} for the image-feature target. Resampling "
            f"training, validation and test patients together, the bootstrap stability threshold was {lf}, with Monte Carlo ranges of {ci(e['tabular']['lcb_floor_mc_range'])} and "
            f"{ci(e['image']['lcb_floor_mc_range'])}, for this specification of *q* only. It is not a confidence bound for the floor or for ψ. At a floor of 0.8, ψ_L was positive and Δ_M2 negative "
            f"in the same replicate in {e['tabular']['joint_counts']['0.8']:,} and {e['image']['joint_counts']['0.8']:,} of {B['tabular']:,} "
            f"replicates.")


def psi_est_sentence():
    d = _v30()["families"]; d25 = _v25()["families"]
    pt = {f: {k: d[f]["concepts"]["color_variegation"][k]["point"] for k in EST} for f in FAMS}
    lb = {f: {k: d[f]["concepts"]["color_variegation"][k]["lcb_floor"] for k in EST} for f in FAMS}
    ll = {f: {k: d[f]["diagnostics"][k]["logloss"] for k in EST} for f in FAMS}
    low = {f: {k: d[f]["diagnostics"][k]["bins"][0]["n"] for k in EST} for f in FAMS}
    npos = sum(d["tabular"]["diagnostics"]["mlp_platt"]["pos_by_tertile"]["color_variegation"])
    best = {f: min(ll[f], key=ll[f].get) for f in FAMS}
    nm = {"mlp_platt": "Platt", "mlp_iso": "isotonic", "gbm_platt": "Platt"}
    assert all(max(ll[f], key=ll[f].get) == "gbm_platt" for f in FAMS), "gradient boosting không phải tệ nhất; sửa câu"
    assert all(best[f] != "gbm_platt" for f in FAMS)
    other = {f: [u(d25[f]["color_variegation"]["s_star"][k]) for k in ("beta", "sigma_learner", "winsor")] for f in FAMS}
    return (f"The plug-in tipping floor moved with the estimator of *q* far more than with sampling (Table 2). The point floor for the tabular-input "
            f"target was {u(pt['tabular']['mlp_iso'])} with isotonic calibration and {u(pt['tabular']['gbm_platt'])} with gradient boosting, "
            f"against {u(pt['tabular']['mlp_platt'])} with Platt calibration, and {u(pt['image']['mlp_iso'])} and {u(pt['image']['gbm_platt'])} "
            f"against {u(pt['image']['mlp_platt'])} for the image-feature target. With only {npos} malignant test lesions in the outer tertiles of color variegation, held-out fit cannot settle between the "
            f"two calibrations of the perceptron in the low-risk region that dominates ψ_L (Supplementary Section S7).")

def psi_support_sentence():
    d = _v30()["families"]; tw = _v25()
    sup = {f: [_nn(d[f]["concepts"]["color_variegation"]["mlp_platt"]["support"][q]["point"]) for q in ("0.01", "0.05", "0.1")] for f in FAMS}
    def fmt(v):
        got = [(q, x) for q, x in zip(("1st", "5th", "10th"), v) if x is not None]
        miss = [q for q, x in zip(("1st", "5th", "10th"), v) if x is None]
        a = (listing([u(x) for _, x in got]) + " at the " + listing([q for q, _ in got]) + (" percentile threshold" if len(got) == 1 else " percentile thresholds")) if got else ""
        b = "was not reached below one at the " + listing(miss) if miss else ""
        return a + (" and " + b if a and b else b)
    w = {f: tw["families"][f]["color_variegation"]["two_floor"]["0.01"]["boundary_s_low_by_s_high"]["0.9"] for f in FAMS}
    return (f"Within the region supported by verified training lesions the primary point floor rose to {fmt(sup['tabular'])} for the tabular-input "
            f"target, and to {fmt(sup['image'])} for the image-feature target. The floor need not be uniform. If malignant verification had a "
            f"floor of 0.9 elsewhere, a floor of {u(w['tabular'])} for the tabular-input target and {u(w['image'])} for the image-feature target sufficed among the {pct(tw['low_share']['0.01'])} of test "
            f"lesions below the 1st percentile threshold.")


def rr_sentence():
    d = J("vr35_marginal_rr.json")["populations"]
    c, tst = d["cohort"]["color_variegation"], d["test"]["color_variegation"]
    return (f"A companion estimand, the marginal risk ratio of disease between the outer tertiles of color variegation, needs only a floor on "
            f"the average malignant verification within each tertile and no model of *q*. From counts on the whole cohort, the recorded-label "
            f"ratio was {u(c['rr_y'])} {ci(c['rr_y_ci95'])}, so a positive sign is identified above a floor of {u(c['tipping'])}, or "
            f"{_fl(c['tipping_one_sided'])} with the 5th percentile of the ratio over patient-cluster resamples. On the test population alone the "
            f"interval was wider, {ci(tst['rr_y_ci95'])}. Among verified lesions the recorded-label log odds ratio was "
            f"{s(c['log_or_verified'])} {ci(c['log_or_verified_ci95'])}, the verified-only reversal of Lemma 1 at the level of the association. "
            f"This estimand is on a different scale and does not validate ψ.")


def t_psi_est():
    d = _v30()
    out = ["| Features | Estimate of *q* | Plug-in point tipping floor | Bootstrap stability threshold [Monte Carlo range] | Joint count at 0.8 | Test log-loss |",
           "| --- | --- | ---: | --- | ---: | ---: |"]
    for f in FAMS:
        for k in EST:
            e = d["families"][f]["concepts"]["color_variegation"][k]
            mc = ci(e["lcb_floor_mc_range"]) if (e["lcb_floor_mc_range"] and _nn(e["lcb_floor"]) is not None) else ""
            out.append(f"| {f} | {EST_LAB[k]} | {_fl(e['point'])} | {(_fl(e['lcb_floor']) + ' ' + mc).strip()} | {e['joint_counts']['0.8']:,} of {d['B'][f]:,} | "
                       f"{d['families'][f]['diagnostics'][k]['logloss']:.4f} |")
    return "\n".join(out)




# ------------------------------------------------------------------ dose with two nuisance specifications (vr32)
def _v32():
    return J("vr32_dose_nuisance.json")["families"]


def _v46():
    return J("vr46_dose_fixed_val.json")["families"]


def _lc_tab():
    d = _v46()["tabular"]
    return {n: d[f"color_variegation|{n}"]["pred"] for n in ("320", "1000", "3000")}


def _tab_converges():
    return False


def abs_dose2():
    return ", at a rate approaching the prediction for tabular features as samples grew" if _tab_converges() else ""


def contrib_dose2():
    return "; with tabular features the rate approaches the predicted one as the training sample grows" if _tab_converges() else ""


def overshoot_detail():
    lc = _lc_tab(); d = _v32()["tabular"]
    r320 = [d[f"{k}|320"]["pred"]["exact"]["ratio"] for k in PRIMARY]
    return (f" by {round(100 * (min(r320) - 1))} to {round(100 * (max(r320) - 1))} percent, and the excess for color variegation fell to "
            f"{round(100 * (lc['3000']['ratio'] - 1))} percent with 3,000")


def _cross(curve):
    e = np.array([-1.0, -0.5, 0.0, 0.5, 1.0]); m = np.asarray(curve)
    return any(m[i] * m[i + 1] <= 0 for i in range(len(m) - 1))


def dose_sentence():
    d = _v46()
    c = {f: {k: d[f][f"{k}|320"] for k in PRIMARY} for f in FAMS}
    tau = {f: [c[f][k]["raw"]["tau_hat"] for k in PRIMARY] for f in FAMS}
    hi_ci = max(c[f][k]["raw"]["tau_ci95"][1] for f in FAMS for k in PRIMARY)
    assert hi_ci < 0
    ncross = sum(c[f][k]["raw"]["crosses_zero"] for f in FAMS for k in PRIMARY)
    nsc = sum(c[f][k]["raw"]["n_sign_change"] for f in FAMS for k in PRIMARY)
    ag = sum(c[f][k]["pred"]["spill_sign_agreement_raw"] for f in FAMS for k in PRIMARY)
    rat = [c[f][k]["pred"]["ratio_raw"] for f in FAMS for k in PRIMARY]
    lc = {f: {n: d[f][f"color_variegation|{n}"] for n in ("1000", "3000")} for f in FAMS}
    big = lc["image"]["3000"]["raw"]
    assert not big["crosses_zero"] and big["tau_ci95"][1] < 0
    lcr = [lc[f][n]["pred"]["ratio_raw"] for f in FAMS for n in ("1000", "3000")]
    return (f"The main result is the training-selection effect τ of Section III-B. With 320 selected recorded negatives, close to "
            f"the training size of the biopsy-only regime, τ̂_k(1, −1) of the selected concept ranged from {s(min(tau['tabular']))} to "
            f"{s(max(tau['tabular']))} with tabular features. With image features it ranged from {s(min(tau['image']))} to "
            f"{s(max(tau['image']))}, and every 95 percent interval lay below {s(hi_ci)}. These are bootstrap intervals over the 20 paired replicates, "
            f"conditional on the cohort, patient split and test population. They average over the algorithmic randomness ω, including "
            f"the validation-set draw used for early stopping, and do not reflect patient sampling. At this size, the mean contrast of the "
            f"selected concept crossed zero in {ncross} of 6 combinations of concept and family. Within the same replicate, it changed "
            f"sign in {nsc} of {20 * 6} replicates. The contrasts of the five concepts moved in the predicted direction in {ag} of "
            f"the 30 combinations of family, selected concept and responding concept.\n\n"
            f"Whether a contrast crosses zero also depends on its baseline value and on the dose range. With 3,000 selected recorded "
            f"negatives, color variegation with image features had τ̂ = {s(big['tau_hat'])} {ci(big['tau_ci95'])}, yet its mean contrast "
            f"did not cross zero and changed sign in only {big['n_sign_change']} of 20 replicates. The raw-logit slopes pointed in the "
            f"direction of the Bayes-optimal training-distribution shift. Their sizes were also close to it in this implementation, at "
            f"{u(min(rat))} to {u(max(rat))} times the prediction with 320 recorded negatives and {u(min(lcr))} to {u(max(lcr))} times "
            f"with 1,000 or 3,000. Raw finite-network logits need not equal the Bayes-optimal log-odds of the selected training "
            f"distribution, so these ratios are descriptive and do not test the magnitude predicted by Proposition 1. In short, the paired "
            f"experiment identifies the finite-pipeline effect τ, and Proposition 1 separately predicts the direction of the matching "
            f"Bayes-optimal shift. Their numerical agreement is a diagnostic, not an identification result (Supplementary Section S6).")



def table2_dose_rows():
    d = _v46(); rows = []
    for f in FAMS:
        rat = [d[f][f"{k}|320"]["pred"]["ratio_raw"] for k in PRIMARY]
        tau = [d[f][f"{k}|320"]["raw"] for k in PRIMARY]
        nsc = sum(x["n_sign_change"] for x in tau)
        rows.append(f"| Selection dose on training only; replicate-specific validation shared across doses | {f}; 320 selected recorded negatives; three primary concepts"
                    f"{'' if f == 'tabular' else ', ridge conditional mean'} | τ̂(1, −1) {s(min(x['tau_hat'] for x in tau))} to "
                    f"{s(max(x['tau_hat'] for x in tau))}; sign change in {nsc} of 60 replicates; descriptive slope ratio {u(min(rat))} to {u(max(rat))} |")
    for f in FAMS:
        cells = []
        for n in ("1000", "3000"):
            c = d[f][f"color_variegation|{n}"]
            cells.append(f"{int(n):,}: τ̂ {s(c['raw']['tau_hat'])} {ci(c['raw']['tau_ci95'])}, sign change in {c['raw']['n_sign_change']} of 20, "
                         f"mean {'crosses' if c['raw']['crosses_zero'] else 'does not cross'} zero, ratio {u(c['pred']['ratio_raw'])}")
        rows.append(f"| Selection dose, learning curve | {f}; color variegation | " + "; ".join(cells) + " |")
    return "\n".join(rows)


def bridge_sentence():
    d = _v22(); a = J("vr33_bridge_alt.json")["families"]; sm = J("vr34_semisynth.json")["families"]
    t_, i_ = d["families"]["tabular"], d["families"]["image"]
    sg = {k: sum(a[f][k]["n_concepts_975"] for f in FAMS) for k in ("mlp|all", "gbm|all")}
    cal = {k: [a[f][k]["concept_calibration"] for f in FAMS] for k in ("mlp|all", "gbm|all")}
    bs = {f: sm[f]["summary"]["bridge_slope"]["median"] for f in FAMS}; bg = {f: sm[f]["summary"]["bridge_slope_gbm"]["median"] for f in FAMS}
    return (f"Proposition 1 is exact only for Bayes-optimal learners, so for trained networks we examine direction first. With a perceptron "
            f"for ĝ, the predicted concept-level gap had the observed sign in at least 97.5 percent of joint resamples in {sg['mlp|all']} "
            f"of 10 combinations of concept and family. With gradient boosting, it did so in {sg['gbm|all']} of 10. Magnitude depended on the "
            f"propensity model, with calibration slopes of observed on predicted gap of {u(min(cal['mlp|all']))} to {u(max(cal['mlp|all']))} "
            f"and {u(min(cal['gbm|all']))} to {u(max(cal['gbm|all']))}. Lesion by lesion, the calibrated logit gap rose with log ĝ with slope "
            f"{u(t_['pointwise_slope'])} {ci(t_['pointwise_slope_ci'])} and {u(i_['pointwise_slope'])} {ci(i_['pointwise_slope_ci'])}. Even where the "
            f"identity holds exactly, in a semi-synthetic design, these slopes ranged from {u(min(list(bs.values()) + list(bg.values())))} "
            f"to {u(max(list(bs.values()) + list(bg.values())))} across propensity models (Supplementary Section S2). Direction thus agreed across propensity models, while size agreed only for some.")

def psi_sim_sentence():
    v = J("vr23_psi_sim.json")["summary"]["valid_floor"]
    st = J("vr28_psi_stress.json")["summary"]
    sms = [J("vr34_semisynth.json")["families"], J("vr34_semisynth_gbm.json")["families"]]
    wrong_valid = max([st[f"{sc}|valid"][f"{m}_wrong"] for sc in ("base", "adversarial") for m in ("oracle", "gbm", "mlp_platt")] + [v["oracle_wrong_sign"], v["plugin_wrong_sign"]])
    assert wrong_valid == 0, "có chứng nhận sai dấu dưới sàn hợp lệ; sửa câu"
    adv = st["adversarial|overstated"]
    ks = ("mlp_platt", "mlp_iso", "gbm_platt")
    cov = [sm[f]["summary"][k]["covers"] for sm in sms for f in FAMS for k in ks]
    wr = max(sm[f]["summary"][k]["wrong"] for sm in sms for f in FAMS for k in ks)
    gcov = [sms[1][f]["summary"]["gbm_platt"]["covers"] for f in FAMS]
    bias = [sm[f]["summary"][k]["bias_L"] for sm in sms for f in FAMS for k in ks]
    return (f"In synthetic populations no construction of the set certified the wrong sign under a valid floor, whereas an overstated floor "
            f"did in {pct(adv['oracle_wrong'])} of adversarial settings even with true nuisances. In semi-synthetic designs on the ISIC-2024 "
            f"inputs, with disease risk built from the perceptron or from gradient boosting, the share of settings in which the plug-in set "
            f"at the true floor contained the true target was {round(100 * min(cov))} to "
            f"{pct(max(cov))} across the three estimators of *q*, and {'none' if wr == 0 else f'up to {pct(wr)}'} certified the wrong sign. "
            f"Gradient boosting had the lowest containment, {round(100 * min(gcov))} to {pct(max(gcov))}, even when disease risk was built "
            f"from gradient boosting, and the plug-in ψ_L mostly lay below its oracle value, with mean differences from {s(min(bias))} to "
            f"{s(max(bias))} (Supplementary Section S2).")

def val_sentence():
    d19, d30 = _v19(), _v30()
    B = min(d30["B"].values())
    lab = lambda e: "Bonferroni" if e["robust_bonf"] else ("95" if e["robust_95"] else "no")
    ch = [(f, c, lab(d19["families"][f]["concepts"][c]), lab(d30["families"][f]["concepts"][c])) for f in FAMS for c in CL
          if lab(d19["families"][f]["concepts"][c]) != lab(d30["families"][f]["concepts"][c])]
    if not ch:
        return f"With validation patients also resampled, in {B:,} replicates per family, every robustness label in Table 1 was unchanged."
    desc = []
    for f, c, a, b in ch:
        e = d30["families"][f]["concepts"][c]
        desc.append(f"{lower(CL[c])} with {f} features" + (" was robust at the 95 percent level only" if (a, b) == ("Bonferroni", "95") else f" changed from {a} to {b}"))
    return (f"With validation patients also resampled, in {B:,} replicates per family, " + listing(desc) +
            ", because the Bonferroni interval of Δ_M0 then reached zero. Every other label was unchanged.")


def abs_robust():
    d19, d30 = _v19(), _v30(); d36 = J("vr36_subsample.json")["families"]
    c = "color_variegation"
    r95 = all(d19["families"][f]["concepts"][c]["robust_95"] and d30["families"][f]["concepts"][c]["robust_95"] and d36[f][c]["robust_95"] for f in FAMS)
    return (". Their intervals lay on opposite sides of zero at the 95 percent level in both feature families under every resampling scheme"
            if r95 else "")

def _joint_floor(jc, B):
    ok = [float(k) for k, v in sorted(jc.items(), key=lambda z: float(z[0])) if v >= 0.95 * B]
    return ok[0] if ok else None


def rr_sentence():
    d = J("vr35_marginal_rr.json"); B = d["B"]; d = d["populations"]
    c, tst = d["cohort"]["color_variegation"], d["test"]["color_variegation"]
    fc, ft = _joint_floor(c["joint_counts"], B), _joint_floor(tst["joint_counts"], B)
    k = f"{fc:.2f}"
    return (f"A second disease-relative result lives on the scale of the marginal association. It needs no model of *q* and only a floor "
            f"on the average malignant verification within each tertile, and it uses counts alone, so we compute it on the whole cohort. It "
            f"is a companion estimand on a different scale and population from ψ, not a check of ψ or Δ_M2. For color variegation the "
            f"recorded-label risk ratio between the outer tertiles was {u(c['rr_y'])} {ci(c['rr_y_ci95'])}, whereas among verified lesions the "
            f"recorded-label log odds ratio was {s(c['log_or_verified'])} {ci(c['log_or_verified_ci95'])}. The disease association was "
            f"identified as positive while the verified-only association was negative in {c['joint_counts'][k]:,} of {B:,} patient-cluster "
            f"resamples at a floor of {k}, the smallest floor reaching 95 percent. On the test population alone the ratio was "
            f"{u(tst['rr_y'])} {ci(tst['rr_y_ci95'])} and that floor {_fl(ft)}.")

def tail_sentence():
    d = J("vr38_logit_tail.json")["families"]["tabular"]
    pl, iso, gb = d["mlp_platt"], d["mlp_iso"], d["gbm_platt"]
    return (f"The dependence has a mechanism. Calibration constrains the expected number of events, not the mean of logit *q* in the "
            f"steep low-risk tail. In the lower tertile of color variegation with tabular features, Platt and isotonic calibration expected "
            f"{pl['0']['sum_q']:.1f} and {iso['0']['sum_q']:.1f} events against {pl['0']['observed']} observed, yet their mean logit *q* differed "
            f"by {abs(pl['0']['mean_logit_q'] - iso['0']['mean_logit_q']):.2f}. Gradient boosting missed even the tertile-level contrast, "
            f"expecting {gb['0']['sum_q']:.1f} and {gb['1']['sum_q']:.1f} events against {gb['0']['observed']} and {gb['1']['observed']}. The "
            f"logit-scale target is therefore weakly estimable in the rare-event tail.")

def abs_primary():
    d19, d30 = _v19(), _v30()
    both = [c for c in PRIMARY for f in ("image",) if d19["families"][f]["concepts"][c]["robust_bonf"] and d30["families"][f]["concepts"][c]["robust_bonf"]]
    fixed = [c for c in PRIMARY for f in ("image",) if d19["families"][f]["concepts"][c]["robust_bonf"] and not d30["families"][f]["concepts"][c]["robust_bonf"]]
    assert both == ["size"] and fixed == ["color_variegation"], "nhãn Bonferroni đổi; sửa câu tóm tắt"
    assert not any(d19["families"]["tabular"]["concepts"][c]["robust_bonf"] for c in PRIMARY)
    return ("After multiplicity adjustment the tertile-based reversal was robust for image-feature size under all resampling schemes, "
            "and for image-feature color variegation only with the validation set fixed")


def sigma_sentence():
    d = J("vr37_overlap_sigma.json")
    sw = [d["membership"][f"{nm}|{q}"]["switched"] for nm in ("mlp_tab", "mlp_img") for q in ("0.01", "0.05", "0.1")]
    assert all(v["m2_below"] for v in d["contrasts"].values())
    rob = lambda kind, nm: sum(d["contrasts"][f"{kind}|{nm}|{q}|color_variegation"]["robust_95"] for q in ("0.01", "0.05", "0.1"))
    rr = [rob(k, nm) for k in FAMS for nm in d["estimators"]]
    return (f"Support is itself estimated. With perceptrons trained on the tabular or on the image features as other estimates of σ, "
            f"{round(100 * min(sw))} to {pct(max(sw))} of test lesions changed membership. Under one test-patient bootstrap conditional on the "
            f"fitted learners, M2 stayed below zero in every combination, whereas the reversal of color variegation stayed robust at "
            f"{min(rr)} to {max(rr)} of the three thresholds depending on the estimate and family (Supplementary Section S6).")

def resample_sentence():
    """Nhãn của các phép lấy lại khác so với nhãn chính của Table 1; kiểm tra một seed (vr19), lấy lại cả val (vr30), lấy mẫu con (vr36)."""
    dp, d30, d19 = _vp(), _v30(), _v19(); d36 = J("vr36_subsample.json")["families"]
    lab = lambda e: "Bonferroni" if e["robust_bonf"] else ("95" if e["robust_95"] else "no")
    rank = {"no": 0, "95": 1, "Bonferroni": 2}
    ch = [(f, c) for f in FAMS for c in CL if lab(dp["families"][f]["concepts"][c]) != lab(d30["families"][f]["concepts"][c])]
    assert all(rank[lab(d30["families"][f]["concepts"][c])] < rank[lab(dp["families"][f]["concepts"][c])] for f, c in ch)
    down = [1 for f in FAMS for c in PRIMARY if (d36[f][c]["robust_bonf"], d36[f][c]["robust_95"]) < (dp["families"][f]["concepts"][c]["robust_bonf"], dp["families"][f]["concepts"][c]["robust_95"])]
    assert not down
    diff = [(f, c) for f in FAMS for c in CL
            if (dp["families"][f]["concepts"][c]["robust_bonf"], dp["families"][f]["concepts"][c]["robust_95"])
            != (d19["families"][f]["concepts"][c]["robust_bonf"], d19["families"][f]["concepts"][c]["robust_95"])]
    a = J("vr29_bootstrap_audit.json")["families"]
    se19 = max(v for f in FAMS for c in PRIMARY for v in a[f][c]["mc_se_endpoints"].values())
    txt = [f"Three checks vary the resampling. The {B19()}-replicate one-seed analysis targets the randomized one-seed training "
           f"procedure and had Monte Carlo standard errors of at most {u(se19)}."]
    if diff:
        txt.append("It agreed on every label except " + listing([f"{lower(CL[c])} with {f} features" for f, c in diff])
                   + ", whose one-seed Bonferroni interval of Δ_M0 was "
                   + listing([ci(d19['families'][f]['concepts'][c]['d0_ci_bonf']) for f, c in diff]) + ".")
    else:
        txt.append("It agreed on every label.")
    txt.append(f"Resampling validation patients as well, in {min(d30['B'].values()):,} one-seed replicates per family, "
               + ("changed no label." if not ch else "changed only the labels of "
                  + (f"{lower(CL[ch[0][1]])} with " + listing([f for f, _ in ch]) + " features" if len({c for _, c in ch}) == 1
                     else listing([f"{lower(CL[c])} with {f} features" for f, c in ch]))
                  + ": their Bonferroni intervals of Δ_M0 then reached zero."))
    txt.append("Half-sampling patients without replacement reversed no label, and we use it only as this check (Supplementary Section S6).")
    return " ".join(txt)


def abs_floor():
    d = J("vr30_q_bootstrap.json")["families"]
    v = [d[f]["concepts"]["color_variegation"][k]["point"] for f in FAMS for k in ("mlp_platt", "mlp_iso", "gbm_platt")]
    v = [x for x in v if x == x]
    return u(min(v)), u(max(v))


# ------------------------------------------------------------------ vòng 7
def _floor_rng(f):
    d = J("vr30_q_bootstrap.json")["families"][f]["concepts"]["color_variegation"]
    v = [d[k]["point"] for k in ("mlp_platt", "mlp_iso", "gbm_platt") if d[k]["point"] == d[k]["point"]]
    return f"{u(min(v))} to {u(max(v))}"


def abs_boot():
    d = J("vr30_q_bootstrap.json")["families"]
    v = [d[f]["concepts"]["color_variegation"][k]["lcb_floor"] for f in FAMS for k in ("mlp_platt", "mlp_iso", "gbm_platt")]
    fin = [x for x in v if x is not None and x == x]
    assert len(fin) < len(v), "mọi ước lượng đều có sàn bootstrap; sửa câu"
    return u(min(fin))


def _marg():
    d = J("vr35_marginal_rr.json"); B = d["B"]; c = d["populations"]["cohort"]["color_variegation"]
    return c, B, _joint_floor(c["joint_counts"], B)


def abs_marg_floor():
    c, B, fc = _marg()
    assert c["B_V"] > 2 * c["A_max95"], "B_V không lớn hơn nhiều so với A tối đa; sửa câu tóm tắt"
    return pct(fc)


def size_defs_sentence():
    d = J("vr9_closing.json")["families"]
    img = d["head_image"]["concepts"]["size"]
    col_ok = all(d[f]["concepts"]["color_variegation"][h]["M0"] > 0 > d[f]["concepts"]["color_variegation"][h]["M2"]
                 for f in ("head_tabular", "head_image", "lp", "ft")
                 for h in ("tertile", "quartile", "quintile", "median", "slope", "within_slope", "rank_slope"))
    assert col_ok and img["within_slope"]["M2"] > 0
    return (f"The two concepts are robust along different axes. Size with image features is the strongest multiplicity-adjusted result "
            f"but rests on the tertile definition: with slope definitions its M2 contrast was {s(img['slope']['M2'])} and, within "
            f"patients, {s(img['within_slope']['M2'])}. Color variegation reversed under all seven concept definitions in all four "
            f"families (Supplementary Section S4).")


def sigma_sentence():
    d = J("vr37_overlap_sigma.json"); qs = ("0.01", "0.05", "0.1")
    own = {"tabular": "mlp_tab", "image": "mlp_img"}
    ret = [d["thresholds"][f"{own[f]}|{q}"]["retained"] for f in FAMS for q in qs]
    assert all(v["m2_below"] for v in d["contrasts"].values())
    c = {f: [d["contrasts"][f"{f}|{own[f]}|{q}|color_variegation"] for q in qs] for f in FAMS}
    assert all(c[f][0]["robust_95"] and c[f][0]["d0"] > 0 and all(x["d0"] < 0 for x in c[f][1:]) for f in FAMS)
    assert all(d["contrasts"][f"image|{nm}|{q}|size"]["robust_95"] for nm in d["estimators"] for q in qs)
    neg = [x["d0"] for f in FAMS for x in c[f][1:]]
    return (f"With the input-matched estimate, which kept {round(100 * min(ret))} to {pct(max(ret))} of test lesions, M2 stayed below "
            f"zero throughout in test-patient intervals conditional on the fitted learners, whereas the M0 contrast of color variegation "
            f"was positive at the 1st percentile threshold but between {s(min(neg))} and {s(max(neg))} at the 5th and 10th. Size with image "
            f"features kept 95 percent intervals on opposite sides of zero at every threshold under every estimate of σ.")


def tail_sentence():
    d = J("vr38_logit_tail.json")["families"]["tabular"]
    pl, iso = d["mlp_platt"], d["mlp_iso"]
    return (f"Calibration approximately matches aggregate event risk, a sum of *q*. It does not match the mean of logit *q*, which is "
            f"steep near zero. In the lower tertile of color variegation with tabular features, Platt and isotonic calibration expected "
            f"{pl['0']['sum_q']:.1f} and {iso['0']['sum_q']:.1f} events against {pl['0']['observed']} observed. Yet their mean logit *q* "
            f"differed by {abs(pl['0']['mean_logit_q'] - iso['0']['mean_logit_q']):.2f} (Supplementary Section S7). In the rare-event "
            f"tail, the logit-scale target is therefore empirically unstable to the estimator of *q*.")


def psi_est_sentence():
    d = _v30()["families"]
    pt = {f: {k: d[f]["concepts"]["color_variegation"][k]["point"] for k in EST} for f in FAMS}
    lb = {f: {k: d[f]["concepts"]["color_variegation"][k]["lcb_floor"] for k in EST} for f in FAMS}
    ll = {f: {k: d[f]["diagnostics"][k]["logloss"] for k in EST} for f in FAMS}
    npos = sum(d["tabular"]["diagnostics"]["mlp_platt"]["pos_by_tertile"]["color_variegation"])
    assert all(max(ll[f], key=ll[f].get) == "gbm_platt" for f in FAMS), "gradient boosting không phải tệ nhất; sửa câu"
    assert lb["image"]["gbm_platt"] is None or lb["image"]["gbm_platt"] != lb["image"]["gbm_platt"]
    return (f"The plug-in tipping floor moved with the estimator of *q* far more than with sampling (Table 2). For the tabular-input "
            f"target, the point floor was {u(pt['tabular']['mlp_iso'])} with isotonic calibration and {u(pt['tabular']['gbm_platt'])} with "
            f"gradient boosting, against {u(pt['tabular']['mlp_platt'])} with Platt calibration. For the image-feature target, it was "
            f"{u(pt['image']['mlp_iso'])} and {u(pt['image']['gbm_platt'])} against {u(pt['image']['mlp_platt'])}. With "
            f"gradient boosting for the image-feature target, no floor up to one made "
            f"ψ_L positive in 95 percent of replicates, so nuisance uncertainty cannot be summarized by one threshold. Gradient boosting had the poorest calibration of "
            f"tertile-level event counts among the three and the most conservative tipping behavior (Supplementary Section S7). We "
            f"report it as a sensitivity specification rather than privileging any nuisance model.")


def psi_support_sentence():
    d = _v30()["families"]; tw = _v25()
    sup = {f: [_nn(d[f]["concepts"]["color_variegation"]["mlp_platt"]["support"][q]["point"]) for q in ("0.01", "0.05", "0.1")] for f in FAMS}
    def fmt(v):
        got = [(q, x) for q, x in zip(("1st", "5th", "10th"), v) if x is not None]
        miss = [q for q, x in zip(("1st", "5th", "10th"), v) if x is None]
        a = (listing([u(x) for _, x in got]) + " at the " + listing([q for q, _ in got]) + (" percentile threshold" if len(got) == 1 else " percentile thresholds")) if got else ""
        b = "was not reached below one at the " + listing(miss) if miss else ""
        return a + (" and " + b if a and b else b)
    w = {f: tw["families"][f]["color_variegation"]["two_floor"]["0.01"]["boundary_s_low_by_s_high"]["0.9"] for f in FAMS}
    return (f"Within the region supported by verified training lesions the primary point floor rose to {fmt(sup['tabular'])} for the tabular-input "
            f"target, and to {fmt(sup['image'])} for the image-feature target. An exploratory analysis with two floors is in Supplementary Section S7.")


def psi_sim_sentence():
    v = J("vr23_psi_sim.json")["summary"]["valid_floor"]
    st = J("vr28_psi_stress.json")["summary"]
    sms = [J("vr34_semisynth.json")["families"], J("vr34_semisynth_gbm.json")["families"]]
    lo = J("vr34_semisynth_local.json")["families"]
    wrong_valid = max([st[f"{sc}|valid"][f"{m}_wrong"] for sc in ("base", "adversarial") for m in ("oracle", "gbm", "mlp_platt")] + [v["oracle_wrong_sign"], v["plugin_wrong_sign"]])
    assert wrong_valid == 0, "có chứng nhận sai dấu dưới sàn hợp lệ; sửa câu"
    adv = st["adversarial|overstated"]
    ks = ("mlp_platt", "mlp_iso", "gbm_platt")
    cov = [sm[f]["summary"][k]["covers"] for sm in sms for f in FAMS for k in ks]
    wr = max(sm[f]["summary"][k]["wrong"] for sm in sms for f in FAMS for k in ks)
    bias = [sm[f]["summary"][k]["bias_L"] for sm in sms for f in FAMS for k in ks]
    assert wr == 0 and all("summary" in lo[f] for f in FAMS)
    lc_or = [lo[f]["summary"]["oracle"]["covers"] for f in FAMS]
    lw = max([lo[f]["summary"]["oracle"]["wrong"] for f in FAMS] + [lo[f]["summary"][k]["wrong"] for f in FAMS for k in ks])
    assert lw == 0
    return (f"Proposition 2 identifies a population set, but its plug-in estimate need not be reliable. An overstated floor "
            f"certified the wrong sign in {pct(adv['oracle_wrong'])} of adversarial synthetic settings even with true nuisances. "
            f"In semi-synthetic designs on the ISIC-2024 inputs, disease risk was built from the perceptron or from gradient boosting. "
            f"No false-sign certification occurred at the true floor. However, the plug-in sets contained the true target in only "
            f"{round(100 * min(cov))} to {pct(max(cov))} of settings. Their lower endpoints were mostly below the oracle ones (mean "
            f"differences {s(min(bias))} to {s(max(bias))}), so they have no demonstrated nominal coverage. In one design, the floor "
            f"failed only among the 30 percent of lesions with the lowest verification propensity. There, even the oracle set contained the true target in only "
            f"{round(100 * min(lc_or))} to {pct(max(lc_or))} of settings (Supplementary Section S2).")


def rr_sentence():
    c, B, fc = _marg()
    tst = J("vr35_marginal_rr.json")["populations"]["test"]["color_variegation"]
    ft = _joint_floor(tst["joint_counts"], B)
    k = f"{fc:.2f}"
    assert c["verified_negative_count"] == B
    return (f"This analysis is neither an estimator nor a check of ψ, and it does not validate Proposition 2. It uses the whole "
            f"cohort rather than the learner's evaluation population. It asks a different question: whether the disease "
            f"association between the outer tertiles has the sign opposite to the verified-only association. Because RR_Y = RR_D·*A* "
            f"exactly (Supplementary Section S1), the disease risk ratio exceeds one when *A* < RR_Y. A floor π₀¹ ≥ *s*_min on the "
            f"average malignant verification in the lower tertile alone gives *A* ≤ 1/*s*_min. The analysis uses counts only. For "
            f"color variegation RR_Y was {u(c['rr_y'])} {ci(c['rr_y_ci95'])}, whereas the verified-only log odds ratio was "
            f"{s(c['log_or_verified'])} {ci(c['log_or_verified_ci95'])}. The marginal association reversal held in "
            f"{c['joint_counts'][k]:,} of {B:,} patient-cluster resamples at a lower-tertile floor of {k}. This is the 95 percent resampling "
            f"stability threshold on a grid of step 0.05. On the continuous *A* scale, it corresponds to about {u(c['A_max95'])}. "
            f"This threshold is not a confidence bound for the floor, and nothing controls error for the choice of concept and analysis, "
            f"both made after the primary results had been seen. On the test population alone the threshold was {_fl(ft)}.")


def seed_sentence():
    d = J("vr39_seed_variance.json")["families"]; d19 = _v19()["families"]
    n = min(d[f]["n_reps"] for f in FAMS)
    same = all(d[f]["concepts"][c]["avg3"]["robust_bonf"] == d[f]["concepts"][c]["single"]["robust_bonf"] == d19[f]["concepts"][c]["robust_bonf"]
               and d[f]["concepts"][c]["avg3"]["robust_95"] == d[f]["concepts"][c]["single"]["robust_95"] == d19[f]["concepts"][c]["robust_95"]
               for f in FAMS for c in PRIMARY)
    assert same, "nhãn trung bình ba seed khác; sửa câu"
    return (f"Refitting the first {n:,} replicates per family with three seeds and "
            f"bootstrapping the three-seed average, as the point estimate is computed, gave the same labels as the primary analysis in all six primary cells, so the conclusions do not depend on this approximation "
            f"(Supplementary Section S6).")


def composition_sentence():
    d = J("vr40_support_composition.json"); qs = ("0.01", "0.05", "0.1"); ests = ("nuisance", "mlp_tab", "mlp_img")
    c = "color_variegation"
    full = {f: d["full"][f"{f}|M0|{c}"]["est"] for f in FAMS}
    tr0 = {f: [d["support"][f"{f}|M0|{nm}|{q}|{c}"]["transported"] for nm in ests for q in qs] for f in FAMS}
    tr2 = [d["support"][f"{f}|M2|{nm}|{q}|{c}"]["transported"] for f in FAMS for nm in ests for q in qs]
    assert all(max(tr0[f]) < full[f] for f in FAMS) and max(tr2) < 0
    return (f"Restriction also changes the composition of each tertile. Reweighting the supported lesions of each tertile to the "
            f"full-population distribution of the five measured concepts did not restore the M0 contrast of color variegation in any of the nine "
            f"settings: it ranged from {s(min(tr0['tabular']))} to "
            f"{s(max(tr0['tabular']))} with tabular and from {s(min(tr0['image']))} to {s(max(tr0['image']))} with image features, "
            f"against {s(full['tabular'])} and {s(full['image'])} on the full population, whereas the reweighted M2 contrast stayed "
            f"between {s(min(tr2))} and {s(max(tr2))} (Supplementary Section S6). The loss of the color learner reversal in "
            f"better-supported regions coincided with a smaller M0 contrast among lesions that are more likely to be verified.")


def partial_sentence():
    d = J("vr40_support_composition.json")["partial_full"]
    v = {f: d[f"{f}|M2|color_variegation"]["est"] for f in FAMS}
    return (f"The tertile contrast is marginal over the other concepts; balancing the tertiles on the other four concepts moved the M2 "
            f"contrast of color variegation to {s(v['tabular'])} with tabular and {s(v['image'])} with image features, so the contrast "
            f"is not the effect of one concept with the others held fixed (Supplementary Section S6).")


def local_sentence():
    lo = J("vr34_semisynth_local.json")["families"]
    lc = [lo[f]["summary"]["oracle"]["covers"] for f in FAMS]
    assert max(lc) < 0.8
    return ("A floor that fails only where verification is rarest lowered even the oracle containment (Section IV-F). The "
            "two-floor analysis of Supplementary Section S7 parameterizes this assumption more flexibly without testing it.")


def site_sentence_marg():
    d = J("vr41_marginal_site.json")["concepts"]["color_variegation"]
    st = d["standardized"]; lo = [v["joint_floor"] for v in d["loso"].values()]
    est = [v for v in d["by_site"].values() if v["rr_y"] is not None]
    npos_or = sum(1 for v in est if v["log_or_verified"] is not None and v["log_or_verified"] > 0)
    nmiss = len(d["by_site"]) - len(est)
    big = sorted(d["by_site"].values(), key=lambda v: -sum(v["n_pos"]))[:2]
    word = {1: "one", 2: "two", 3: "three"}
    w6 = {5: "five", 6: "six", 7: "seven"}
    return (f"Under a common lower-tertile floor assumed to hold within every observed site, the site-standardized whole-cohort "
            f"threshold was {u(st['joint_floor'])}. Leaving out one site at a time, it ranged from {u(min(lo))} to {u(max(lo))}. "
            f"This is a mixture estimand for the observed sites, not evidence of a reversal that holds in each site. Site-specific RR_Y "
            f"ranged from {u(min(v['rr_y'] for v in est))} to {u(max(v['rr_y'] for v in est))} across the {w6[len(est)]} estimable sites. "
            f"The two sites with the most malignant lesions had {u(big[0]['rr_y'])} and {u(big[1]['rr_y'])}, implying point floors of "
            f"{u(big[0]['floor'])} and {u(big[1]['floor'])}. {word[nmiss].capitalize()} site could not be estimated, and in {word[npos_or]} "
            f"site the verified-only log odds ratio was positive (Supplementary Section S7).")


def benchmark_sentence():
    d = J("vr41_marginal_site.json")["concepts"]["color_variegation"]["logit_benchmark"]
    c = J("vr35_marginal_rr.json")["populations"]["cohort"]["color_variegation"]
    return (f"Whether such a floor is plausible depends on how appearance dependence is transported from benign to malignant "
            f"verification, which the data cannot identify. The crude whole-cohort benign ratio *B*_V was {u(c['B_V'])} {ci(c['B_V_ci95'])}. On a "
            f"ratio scale this would exceed every *A* compatible with π₀¹ above {u(d['ratio_ceiling_p0'])}. With the same logit gradient, "
            f"log *B*_V, *A* stays below {u(d['A_max95'])} once π₀¹ ≥ {u(d['p0_threshold'])} (Supplementary Fig. S3). Both are "
            f"benchmarks, not estimates of *A*.")



def balanced_sentence():
    d = J("vr40_support_composition.json")["partial_full"]
    g = lambda f, m, c: d[f"{f}|{m}|{c}"]
    sep = lambda f, c: (g(f, "M0", c)["ci"][0] > 0 and g(f, "M2", c)["ci"][1] < 0)
    assert sep("tabular", "color_variegation") and not sep("image", "size") and not sep("image", "color_variegation")
    return (f"Balancing the tertiles on the other four measured concepts changes the estimand. This removed the reversal of size "
            f"with image features (M0 {s(g('image', 'M0', 'size')['est'])}, M2 {s(g('image', 'M2', 'size')['est'])}) and nearly that of "
            f"color variegation with image features (M2 {s(g('image', 'M2', 'color_variegation')['est'])}). Color variegation "
            f"with tabular features kept separated signs. These intervals condition on the fitted learners and are not part of the "
            f"primary joint-bootstrap multiplicity analysis (Supplementary Section S6).")


def sharp_sentence():
    d = J("vr42_sharp_weighted.json")["isic"]
    c = "color_variegation"
    tb, im = d[f"tabular|{c}"], d[f"image|{c}"]
    assert tb["sharp_plugin"]["s_star"] == tb["tertile_formula"]["s_star"] and im["sharp_plugin"]["s_star"] < im["tertile_formula"]["s_star"]
    return ("For the image-feature target, these floors come from the outer stratum formula. An exploratory estimated-weight "
            "approximation to the sharp set gave narrower bounds. Its finite-sample validity was not established, so it is not "
            "used for certification (Supplementary Section S7).")


def overlap_paragraph():
    d37 = J("vr37_overlap_sigma.json"); qs = ("0.01", "0.05", "0.1")
    own = {"tabular": "mlp_tab", "image": "mlp_img"}
    ret = [d37["thresholds"][f"{own[f]}|{q}"]["retained"] for f in FAMS for q in qs]
    assert all(v["m2_below"] for v in d37["contrasts"].values())
    c = {f: [d37["contrasts"][f"{f}|{own[f]}|{q}|color_variegation"] for q in qs] for f in FAMS}
    assert all(c[f][0]["d0"] > 0 and all(x["d0"] < 0 for x in c[f][1:]) for f in FAMS)
    assert all(d37["contrasts"][f"image|{nm}|{q}|size"]["robust_95"] for nm in d37["estimators"] for q in qs)
    neg = [x["d0"] for f in FAMS for x in c[f][1:]]
    d = _v19(); rt = _ret(); qq = sorted(rt, key=float)
    n_m2 = n_cells = m0zero = 0
    for fam in FAMS:
        for k in PRIMARY:
            for q in qq:
                v = d["families"][fam]["concepts"][k]["support"][q]; n_cells += 1
                n_m2 += v["d2_ci95"][1] < 0; m0zero += v["d0_ci95"][0] <= 0
    d40 = J("vr40_support_composition.json"); ests = ("nuisance", "mlp_tab", "mlp_img"); cc = "color_variegation"
    full = {f: d40["full"][f"{f}|M0|{cc}"]["est"] for f in FAMS}
    tr0 = [d40["support"][f"{f}|M0|{nm}|{q}|{cc}"]["transported"] for f in FAMS for nm in ests for q in qs]
    tr2 = [d40["support"][f"{f}|M2|{nm}|{q}|{cc}"]["transported"] for f in FAMS for nm in ests for q in qs]
    assert max(tr2) < 0 and all(max(d40["support"][f"{f}|M0|{nm}|{q}|{cc}"]["transported"] for nm in ests for q in qs) < full[f] for f in FAMS)
    return (f"M2 learns from verified lesions but is evaluated on the whole test population, which could force extrapolation. We "
            f"therefore restricted the test population to lesions whose estimated verification propensity σ̂ reached the 1st, 5th "
            f"or 10th percentile among verified training lesions. The theory-matched σ̂ is fitted on each learner's own input, as in "
            f"Proposition 2, and kept {round(100 * min(ret))} to {pct(max(ret))} of test lesions. A second σ̂ on the tabular features "
            f"gives a common support that admits joint intervals. The M2 contrast stayed below zero throughout. Its joint intervals "
            f"were below zero in {n_m2} of {n_cells} combinations of primary comparison and threshold. Size with image features kept "
            f"95 percent intervals on opposite sides of zero at every threshold under every estimate of σ. The M0 contrast of color "
            f"variegation fell instead. With the input-matched σ̂, it was between {s(min(neg))} and {s(max(neg))} at the 5th and 10th "
            f"thresholds. Under the common support, the joint interval of M0 reached zero in {m0zero} of {n_cells} combinations. "
            f"Standardizing the supported lesions on the five measured concepts did not restore it. It ranged from {s(min(tr0))} to "
            f"{s(max(tr0))}, against {s(full['tabular'])} and {s(full['image'])} on the full population, while the standardized M2 "
            f"contrast stayed between {s(min(tr2))} and {s(max(tr2))}. In better-supported regions, the loss of the color learner "
            f"reversal thus coincided with a smaller M0 contrast among lesions more likely to be verified. The restriction changes the "
            f"evaluation population and does not identify why. The full-population learner reversal describes the held-out test "
            f"population, and the thresholds are a sensitivity analysis, not a proof of positivity (Supplementary Section S6).")

NUM = {"overlap_paragraph": overlap_paragraph, "sharp_sentence": sharp_sentence, "balanced_sentence": balanced_sentence, "composition_sentence": composition_sentence, "partial_sentence": partial_sentence, "local_sentence": local_sentence, "site_sentence_marg": site_sentence_marg, "benchmark_sentence": benchmark_sentence, "floor_tab": lambda: _floor_rng("tabular"), "floor_img": lambda: _floor_rng("image"), "abs_boot": abs_boot, "size_defs_sentence": size_defs_sentence, "seed_sentence": seed_sentence, "resample_sentence": resample_sentence, "sigma_sentence": sigma_sentence, "abs_primary": lambda: abs_primary(), "abs_marg_floor": abs_marg_floor, "tail_sentence": tail_sentence, "concl_dose": lambda: ", and with tabular features at a rate that approached the predicted one as the training sample grew" if _tab_converges() else "", "abs_dose2": abs_dose2, "contrib_dose2": contrib_dose2, "overshoot_detail": overshoot_detail, "abs_floor_lo": lambda: abs_floor()[0], "abs_floor_hi": lambda: abs_floor()[1], "B30": B30, "subsample_sentence": subsample_sentence, "psi_primary_sentence": psi_primary_sentence, "psi_est_sentence": psi_est_sentence, "psi_support_sentence": psi_support_sentence, "val_sentence": val_sentence, "rr_sentence": rr_sentence, "B_primary": B_primary, "B19": B19, "mc_sentence": mc_sentence, "abs_dose": abs_dose, "contrib_dose": contrib_dose, "abs_robust": abs_robust, "abs_sstar": abs_sstar, "jb_sentence": jb_sentence, "mc_rule": mc_rule, "abs_inference": abs_inference, "tau_sentence": tau_sentence, "table1_mc": table1_mc, "mc_text": mc_text, "psi_sentence": psi_sentence,
       "support_sentence": support_sentence, "dose_sentence": dose_sentence, "arms_sentence": arms_sentence,
       "bridge_sentence": bridge_sentence, "site_sentence": site_sentence, "psi_sim_sentence": psi_sim_sentence,
       "table2_rows": table2_rows}
TABLE = {"main_reversal": t_main_reversal, "psi_est": t_psi_est}


# ---------------------------------------------------------------------------------------------------------------------
# Revision: exact bounds, nuisance inputs, expected dose counts, site weights and Monte Carlo resolution (vr47, vr48).
# Each function below replaces an earlier one of the same name or adds a new placeholder.
# ---------------------------------------------------------------------------------------------------------------------
import glob as _glob


def nuis_sentence():
    d = J("vr48_nuisance_audit.json")["families"]
    assert all(d[f]["s_star"]["platt"] == d[f]["s_star"]["platt_sigma_input"] for f in FAMS)
    mx = max(v["share_lower_tertile_second_branch"] for f in FAMS for v in d[f]["branch"].values())
    assert mx < 0.0001 + 1e-12
    return ("For the image-feature target this σ̂ is not on the learner's input. It enters only through the second branch of *p̄*, "
            "which was active for at most 0.01 percent of lower-tertile test lesions at the point tipping floor. An input-matched σ̂ "
            "left the tipping floors unchanged (Supplementary Section S7).")


def _labels(e):
    return "Bonferroni" if e["robust_bonf"] else ("95" if e["robust_95"] else "no")


def resample_sentence():
    dp, d30, d19 = _vp(), _v30(), _v19(); d47 = J("vr47_counts_audit.json")["allsplits_mc"]
    a = J("vr29_bootstrap_audit.json")["families"]
    se19 = max(v for f in FAMS for c in PRIMARY for v in a[f][c]["mc_se_endpoints"].values())
    diff19 = [(f, c) for f in FAMS for c in CL if _labels(dp["families"][f]["concepts"][c]) != _labels(d19["families"][f]["concepts"][c])]
    assert diff19 == [("tabular", "color_variegation")]
    for f in FAMS:
        for c in PRIMARY:
            assert {"Bonferroni": "Bonferroni", "95": "95", "no": "no"}[d47[f][c]["label"]] == _labels(d30["families"][f]["concepts"][c])
    ch = [(f, c) for f in FAMS for c in PRIMARY if _labels(d30["families"][f]["concepts"][c]) != _labels(d19["families"][f]["concepts"][c])]
    assert ch == [("image", "color_variegation")]
    e = d47["image"]["color_variegation"]
    assert e["reproduced"] < 0.95 and all(d47[f][c]["reproduced"] >= 0.95 for f in FAMS for c in PRIMARY if (f, c) not in ch)
    d36 = J("vr36_subsample.json")["families"]
    sep = [(f, c) for f in FAMS for c in PRIMARY if dp["families"][f]["concepts"][c]["robust_bonf"] or _labels(d19["families"][f]["concepts"][c]) == "Bonferroni"]
    assert all(d36[f][c]["robust_bonf"] for f, c in sep)
    return (f"Three checks vary the resampling. The {B19()}-replicate one-seed analysis targets the randomized one-seed training "
            f"procedure and had Monte Carlo standard errors of at most {u(se19)}. It agreed on every label except color variegation "
            f"with tabular features, whose one-seed Bonferroni interval of Δ_M0 was "
            f"{ci(d19['families']['tabular']['concepts']['color_variegation']['d0_ci_bonf'])}. A second check also resampled "
            f"validation patients, independently within each split, in {min(d30['B'].values()):,} one-seed replicates per family. "
            f"Against the one-seed check, it changed only the label of color variegation with image features, from Bonferroni to "
            f"95 percent only. This change is not resolved. The lower Bonferroni endpoint of Δ_M0 was {s(e['d0_bonf'][0])}, each "
            f"Bonferroni tail held about four replicates, and the new label was reproduced in only {pct(e['reproduced'])} of "
            f"Monte Carlo resamples. Half-sampling patients without replacement kept every bootstrap sign-separated cell separated, "
            f"and we use it only as this check (Supplementary Section S6).")


def size_defs_sentence():
    d = J("vr9_closing.json")["families"]
    img = d["head_image"]["concepts"]["size"]
    H = ("tertile", "quartile", "quintile", "median", "slope", "within_slope", "rank_slope")
    ok = lambda c: all(d[f]["concepts"][c][h]["M0"] > 0 > d[f]["concepts"][c][h]["M2"] for f in ("head_tabular", "head_image", "lp", "ft") for h in H)
    assert ok("color_variegation") and ok("lesion_skin_contrast") and not ok("size") and img["within_slope"]["M2"] > 0
    return (f"The two concepts are robust along different axes. Size with image features is the strongest multiplicity-adjusted result "
            f"but rests on the tertile definition: with slope definitions its M2 contrast was {s(img['slope']['M2'])} and, within "
            f"patients, {s(img['within_slope']['M2'])}. Color variegation and lesion-skin contrast kept opposite point-estimate signs "
            f"under all seven concept definitions in all four families. These are point estimates conditional on the fitted models "
            f"(Supplementary Section S4).")


def arms_sentence():
    d = _v21()
    cells = [(f, a) for f in FAMS for a in ("ver", "flag")]
    ag = sum(d["families"][f]["arms"][a]["summary"]["sign_agreement"] for f, a in cells)
    mad = [d["families"][f]["arms"][a]["summary"]["mean_abs_diff"] for f, a in cells]
    return (f"Proposition 1 predicted the sign of the shift from the random arm in {ag} of {5 * len(cells)} combinations of arm, family "
            f"and concept. Across the four combinations of arm and family, the mean absolute difference between observed and "
            f"predicted shifts ranged from {u(min(mad))} to {u(max(mad))}.")


def _dose_reps():
    out = {}
    for p in _glob.glob(os.path.join(os.path.dirname(HERE), "Result", "vr46", "*_320.json")):
        x = json.load(open(p)); k = x["concept"]
        M = np.array([[r[k] for r in x["reps"][i]["raw"]] for i in x["reps"]])
        n = np.array([x["reps"][i]["n_neg"] for i in x["reps"]])
        out[(x["family"], k)] = (M, n)
    assert len(out) == 6
    return out


_dose_sentence_old = dose_sentence


def dose_sentence():
    txt = _dose_sentence_old()
    R_ = _dose_reps()
    nmin = min(int(n.min()) for _, n in R_.values()); nmax = max(int(n.max()) for _, n in R_.values())
    mono_mean = all(np.all(np.diff(M.mean(0)) < 0) for M, _ in R_.values())
    sh = [float(np.mean(np.all(np.diff(M, axis=1) < 0, axis=1))) for M, _ in R_.values()]
    assert mono_mean
    a = ("With 320 selected recorded negatives, close to the training size of the biopsy-only regime, τ̂_k(1, −1) of the selected "
         "concept ranged from")
    assert a in txt
    txt = txt.replace(a, f"The design fixes 320 recorded negatives in expectation; realized counts ranged from {nmin} to {nmax} per fit. "
                         "At this size, close to the training size of the biopsy-only regime, τ̂_k(1, −1), the paired difference between "
                         "the two extreme doses, of the selected concept ranged from")
    b = ", and every 95 percent interval lay below "
    assert b in txt
    txt = txt.replace(b, ", and every upper 95 percent endpoint was at most ")
    c = "At this size, the mean contrast of the selected concept crossed zero in"
    assert c in txt
    txt = txt.replace(c, f"The mean contrast of the selected concept decreased at each of the four steps between doses in 6 of 6 "
                         f"combinations, and within single replicates it did so in {round(100 * min(sh))} to {pct(max(sh))} of replicates. "
                         f"The mean contrast crossed zero in")
    txt = txt.replace("With 3,000 selected recorded negatives", "With 3,000 expected recorded negatives")
    txt = txt.replace("times the prediction with 320 recorded negatives", "times the prediction with 320 expected recorded negatives")
    return txt


_table2_dose_rows_old = table2_dose_rows


def table2_dose_rows():
    return _table2_dose_rows_old().replace("320 selected recorded negatives", "320 expected recorded negatives")


_table2_rows_old = table2_rows


def table2_rows():
    t = _table2_rows_old()
    old = "6 of 6 attenuated, P(signs differ) at most 0.04"
    assert old in t
    return t.replace(old, "6 of 6 attenuated, opposite signs in at most 4 percent of resamples")


def bridge_sentence():
    d = _v22(); a = J("vr33_bridge_alt.json")["families"]
    sms = [J("vr34_semisynth.json")["families"], J("vr34_semisynth_gbm.json")["families"]]
    t_, i_ = d["families"]["tabular"], d["families"]["image"]
    sg = {k: sum(a[f][k]["n_concepts_975"] for f in FAMS) for k in ("mlp|all", "gbm|all")}
    cal = {k: [a[f][k]["concept_calibration"] for f in FAMS] for k in ("mlp|all", "gbm|all")}
    est = [sm[f]["summary"][k]["median"] for sm in sms for f in FAMS for k in ("bridge_slope", "bridge_slope_gbm")]
    tru = [sm[f]["summary"]["bridge_slope"]["true_g_median"] for sm in sms for f in FAMS]
    return (f"Proposition 1 is exact only for Bayes-optimal learners, so for trained networks we examine direction first. With a perceptron "
            f"for ĝ, the predicted concept-level gap had the observed sign in at least 97.5 percent of joint resamples in {sg['mlp|all']} "
            f"of 10 combinations of concept and family. With gradient boosting, it did so in {sg['gbm|all']} of 10. Magnitude depended on the "
            f"propensity model, with calibration slopes of observed on predicted gap of {u(min(cal['mlp|all']))} to {u(max(cal['mlp|all']))} "
            f"and {u(min(cal['gbm|all']))} to {u(max(cal['gbm|all']))}. Lesion by lesion, the calibrated logit gap rose with log ĝ with slope "
            f"{u(t_['pointwise_slope'])} {ci(t_['pointwise_slope_ci'])} and {u(i_['pointwise_slope'])} {ci(i_['pointwise_slope_ci'])}. In "
            f"semi-synthetic designs where the identity holds exactly, the median slopes ranged from {u(min(est))} to {u(max(est))} across "
            f"the two designs and two propensity models, and from {u(min(tru))} to {u(max(tru))} on the true log *g* (Supplementary "
            f"Section S2). Direction thus agreed across propensity models, while size agreed only for some. These results hold for the "
            f"five concepts and two learner families examined.")


def site_sentence():
    a = J("vr20_audit.json")
    h = a["site_heterogeneity"]["color_variegation"]
    assert h["range"][0] > 0
    lo = a["leave_one_site_out"]
    assert min(v["color_variegation"][0] for f in lo for v in lo[f].values()) > 0 > max(v["color_variegation"][1] for f in lo for v in lo[f].values())
    return (f"Between the outer tertiles of color variegation, the adjusted log odds ratio of verification among recorded negatives "
            f"was positive in all {h['k']} acquisition sites where it could be estimated. It approximates log *B*_V when verification is "
            f"rare. Its pooled value was {u(h['pooled_DL'])} {ci(h['pooled_ci'])}, with substantial heterogeneity (I² = {u(h['I2'])}). "
            f"Two further sites were not estimable. Leaving out one site at a time and retraining kept the signs of both color "
            f"variegation contrasts in both families (Supplementary Section S6).")


_overlap_old = overlap_paragraph


def overlap_paragraph():
    t = _overlap_old()
    d37 = J("vr37_overlap_sigma.json")["contrasts"]; d40 = J("vr40_support_composition.json")["support"]
    qs = ("0.01", "0.05", "0.1")
    raw = [d37[f"tabular|mlp_img|{q}|color_variegation"]["d0"] for q in qs]
    tr = [d40[f"tabular|M0|mlp_img|{q}|color_variegation"]["transported"] for q in qs]
    assert min(raw) > 0 and min(tr) > 0
    old = ("In better-supported regions, the loss of the color learner reversal thus coincided with a smaller M0 contrast among "
           "lesions more likely to be verified.")
    assert old in t
    return t.replace(old, f"Under the common and input-matched σ̂, the loss of the color learner reversal in better-supported regions "
                          f"thus coincided with a smaller M0 contrast among lesions more likely to be verified. This is not universal. "
                          f"With σ̂ fitted on image features, the tabular-feature M0 contrast stayed between {s(min(raw))} and "
                          f"{s(max(raw))}, and between {s(min(tr))} and {s(max(tr))} after standardization.")


def sharp_sentence():
    d = J("vr42_sharp_weighted.json")["isic"]
    c = "color_variegation"
    tb, im = d[f"tabular|{c}"], d[f"image|{c}"]
    assert tb["sharp_plugin"]["s_star"] == tb["tertile_formula"]["s_star"] and im["sharp_plugin"]["s_star"] < im["tertile_formula"]["s_star"]
    return ("For the image-feature target, these floors come from the outer stratum formula. An exploratory estimated-weight "
            "approximation to the sharp set gave narrower bounds. Its finite-sample validity was not established, so we do not use it "
            "to report a sign (Supplementary Section S7).")


def _dq():
    d = J("vr38_logit_tail.json")["families"]
    return {f: {k: d[f][k]["1"]["mean_logit_q"] - d[f][k]["0"]["mean_logit_q"] for k in EST} for f in FAMS}


def psi_est_sentence():
    d = _v30()["families"]
    pt = {f: {k: d[f]["concepts"]["color_variegation"][k]["point"] for k in EST} for f in FAMS}
    ll = {f: {k: d[f]["diagnostics"][k]["logloss"] for k in EST} for f in FAMS}
    assert all(max(ll[f], key=ll[f].get) == "gbm_platt" for f in FAMS)
    dq = _dq()
    assert all(abs(np.exp(-dq[f][k]) - pt[f][k]) < 0.03 for f in FAMS for k in EST)
    return (f"The plug-in tipping floor also moved with the estimator of *q* (Table 2). For the tabular-input target, the point floor "
            f"was {u(pt['tabular']['mlp_iso'])} with isotonic calibration and {u(pt['tabular']['gbm_platt'])} with gradient boosting, "
            f"against {u(pt['tabular']['mlp_platt'])} with Platt calibration. For the image-feature target, it was "
            f"{u(pt['image']['mlp_iso'])} and {u(pt['image']['gbm_platt'])} against {u(pt['image']['mlp_platt'])}. Each point floor lies "
            f"within 0.03 of exp(−Δ_q), computed from the mean logit *q* in the two strata. This approximation explains the point "
            f"estimates, and the floors themselves come from the exact bound. With gradient boosting for the image-feature target, no "
            f"floor up to one made ψ_L positive in 95 percent of replicates, so nuisance uncertainty cannot be summarized by one "
            f"threshold. Gradient boosting had the poorest calibration of tertile-level event counts among the three and the most "
            f"conservative tipping behavior (Supplementary Section S7). We report it as a sensitivity specification rather than "
            f"privileging any nuisance model.")


def tail_sentence():
    d = J("vr38_logit_tail.json")["families"]["tabular"]
    pl, iso = d["mlp_platt"], d["mlp_iso"]
    a = J("vr48_nuisance_audit.json")["families"]
    t = a["tabular"]
    assert a["image"]["stats_mean_over_seeds"]["iso_at_lower"] == 0
    return (f"Calibration approximately matches aggregate event risk, a sum of *q*. It does not match the mean of logit *q*, which is "
            f"steep near zero. In the lower tertile of color variegation with tabular features, Platt and isotonic calibration expected "
            f"{pl['0']['sum_q']:.1f} and {iso['0']['sum_q']:.1f} events against {pl['0']['observed']} observed. Yet their mean logit *q* "
            f"differed by {abs(pl['0']['mean_logit_q'] - iso['0']['mean_logit_q']):.2f}. Isotonic calibration put {pct(t['stats_mean_over_seeds']['iso_at_lower'])} "
            f"of tabular test lesions at its lower bound of 10⁻⁶. Moving that bound to 10⁻⁵ or 10⁻⁷ moved the point floor from "
            f"{u(t['s_star']['iso_1e-6'])} to {u(t['s_star']['iso_1e-5'])} or {u(t['s_star']['iso_1e-7'])} (Supplementary Section S7). "
            f"In the rare-event tail, the logit-scale target is therefore empirically unstable to the estimator of *q*.")


def t_psi_est():
    d = _v30(); dq = _dq()
    out = ["| Features | Estimate of *q* | Δ_q | exp(−Δ_q) | Plug-in point tipping floor | Bootstrap stability threshold [Monte Carlo range] | Joint count at 0.8 | Test log-loss |",
           "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |"]
    for f in FAMS:
        for k in EST:
            e = d["families"][f]["concepts"]["color_variegation"][k]
            mc = ci(e["lcb_floor_mc_range"]) if (e["lcb_floor_mc_range"] and _nn(e["lcb_floor"]) is not None) else ""
            out.append(f"| {f} | {EST_LAB[k]} | {u(dq[f][k])} | {u(np.exp(-dq[f][k]))} | {_fl(e['point'])} | {(_fl(e['lcb_floor']) + ' ' + mc).strip()} | "
                       f"{e['joint_counts']['0.8']:,} of {d['B'][f]:,} | {d['families'][f]['diagnostics'][k]['logloss']:.4f} |")
    return "\n".join(out)


def psi_sim_sentence():
    st = J("vr28_psi_stress.json")["summary"]
    sms = [J("vr34_semisynth.json")["families"], J("vr34_semisynth_gbm.json")["families"]]
    lo = J("vr34_semisynth_local.json")["families"]
    ks = ("mlp_platt", "mlp_iso", "gbm_platt")
    adv = st["adversarial|overstated"]
    cov = [sm[f]["summary"][k]["covers"] for sm in sms for f in FAMS for k in ks]
    assert max(sm[f]["summary"][k]["wrong"] for sm in sms for f in FAMS for k in ks) == 0
    bias = [sm[f]["summary"][k]["bias_L"] for sm in sms for f in FAMS for k in ks]
    lc_or = [lo[f]["summary"]["oracle"]["covers"] for f in FAMS]
    return (f"Proposition 2 identifies a population set, but its plug-in estimate need not be reliable. With an overstated floor, "
            f"even the oracle set excluded zero on the wrong side in {pct(adv['oracle_wrong'])} of adversarial synthetic settings. "
            f"In semi-synthetic designs on the ISIC-2024 inputs, disease risk was built from the perceptron or from gradient boosting. "
            f"At the true floor, no plug-in reported a sign opposite to a true target beyond ±0.05. Pooling the 24 settings of each "
            f"combination of design, family and estimator of *q*, the plug-in sets contained the true target in {round(100 * min(cov))} "
            f"to {pct(max(cov))} of settings; single scenarios ranged from 50 to 100 percent. Where the true target was positive, the "
            f"plug-in lower endpoint became positive in 38 to 100 percent of settings. Lower endpoints were mostly below the oracle ones "
            f"(mean differences {s(min(bias))} to {s(max(bias))}), so the plug-in has no demonstrated nominal coverage. In one design, the "
            f"floor failed only among the 30 percent of lesions with the lowest verification propensity. There, even the oracle set "
            f"contained the true target in only {round(100 * min(lc_or))} to {pct(max(lc_or))} of settings (Supplementary Section S2).")


_rr_old = rr_sentence


def rr_sentence():
    t = _rr_old()
    v = J("vr47_counts_audit.json")["fig4_paths"]
    sz = J("vr35_marginal_rr.json")["populations"]["cohort"]["size"]
    c, B, fc = _marg()
    assert _joint_floor(sz["joint_counts"], B) == fc
    br = max(v[k]["floor_branch_active_above"] for k in v)
    assert br < 0.01
    t = t.replace(" The analysis uses counts only.", "")
    t = t.replace(" On the test population alone", " Size had the same whole-cohort threshold. On the test population alone")
    return t + (f" The floor branch of the bound was active at every floor above {br:.3f}, so "
                f"[*s*_min·RR_Y, RR_Y/*s*_min] is the sharp interval for RR_D here (Supplementary Section S1).")


def site_sentence_marg():
    d = J("vr41_marginal_site.json")["concepts"]["color_variegation"]
    w = J("vr47_counts_audit.json")["site_common_weights"]["color_variegation"]
    lo = {k: v["joint_floor"] for k, v in d["loso"].items()}
    est = [v for v in d["by_site"].values() if v["rr_y"] is not None]
    allsites = list(d["by_site"].values())
    tot = sum(sum(v["n_pos"]) for v in allsites)
    big = sorted(allsites, key=lambda v: -sum(v["n_pos"]))[:2]
    share = sum(sum(v["n_pos"]) for v in big) / tot
    npos_or = sum(1 for v in est if v["log_or_verified"] is not None and v["log_or_verified"] > 0)
    assert npos_or == 1 and len(w["excluded_sites"]) == 1 and w["log_rr_verified_std_ci95"][1] < 0
    return (f"With the same site weights on both sides, over the {len(w['supported_sites'])} sites with verified lesions in both "
            f"tertiles, the standardized RR_Y was {u(w['rr_y_std'])} {ci(w['rr_y_std_ci95'])}. The standardized verified-only log risk "
            f"ratio was {s(w['log_rr_verified_std'])} {ci(w['log_rr_verified_std_ci95'])}, and the threshold was {u(w['joint_floor'])}. "
            f"This is a mixture estimand for the observed sites, not evidence of a reversal in each site. The two sites with the most "
            f"malignant lesions held {pct(share)} of those in the outer tertiles. Their own RR_Y were {u(big[0]['rr_y'])} and "
            f"{u(big[1]['rr_y'])}, implying point floors of {u(big[0]['floor'])} and {u(big[1]['floor'])}, against "
            f"{u(1 / J('vr35_marginal_rr.json')['populations']['cohort']['color_variegation']['rr_y'])} for the whole cohort. With the "
            f"verified-only side pooled by Mantel-Haenszel, leaving out one site at a time moved the threshold in both directions, "
            f"from {u(min(lo.values()))} to {u(max(lo.values()))}. In one site the verified-only log odds ratio was positive (Supplementary Section S7).")


NUM.update({"nuis_sentence": nuis_sentence, "resample_sentence": resample_sentence, "size_defs_sentence": size_defs_sentence,
            "arms_sentence": arms_sentence, "dose_sentence": dose_sentence, "table2_rows": table2_rows,
            "bridge_sentence": bridge_sentence, "site_sentence": site_sentence, "overlap_paragraph": overlap_paragraph,
            "sharp_sentence": sharp_sentence, "psi_est_sentence": psi_est_sentence, "tail_sentence": tail_sentence,
            "psi_sim_sentence": psi_sim_sentence, "rr_sentence": rr_sentence, "site_sentence_marg": site_sentence_marg})
TABLE.update({"psi_est": t_psi_est})


# Shorter versions of long Results sentences (same numbers, fewer repeated caveats).
_dose_sentence_rev = dose_sentence


def dose_sentence():
    t = _dose_sentence_rev()
    a = t.index(" Raw finite-network logits need not equal")
    b = t.index("(Supplementary Section S6).", a) + len("(Supplementary Section S6).")
    return t[:a] + " These ratios are descriptive, as Section III-C explains (Supplementary Section S6)." + t[b:]


_rr_rev = rr_sentence


def rr_sentence():
    t = _rr_rev()
    old = ("This analysis is neither an estimator nor a check of ψ, and it does not validate Proposition 2. It uses the whole "
           "cohort rather than the learner's evaluation population. It asks a different question: whether the disease "
           "association between the outer tertiles has the sign opposite to the verified-only association.")
    assert old in t
    return t.replace(old, "This analysis is separate from ψ and does not validate Proposition 2. It uses counts from the whole cohort "
                          "and asks whether the disease association between the outer tertiles has the sign opposite to the "
                          "verified-only association.")


_overlap_rev = overlap_paragraph


def overlap_paragraph():
    t = _overlap_rev()
    old = ("The restriction changes the evaluation population and does not identify why. The full-population learner reversal "
           "describes the held-out test population, and the thresholds are a sensitivity analysis, not a proof of positivity")
    assert old in t
    return t.replace(old, "The restriction changes the evaluation population, so it is a sensitivity analysis and does not identify why")


_psi_est_rev = psi_est_sentence


def psi_est_sentence():
    t = _psi_est_rev()
    old = ("Gradient boosting had the poorest calibration of tertile-level event counts among the three and the most conservative "
           "tipping behavior (Supplementary Section S7). We report it as a sensitivity specification rather than privileging any "
           "nuisance model.")
    assert old in t
    return t.replace(old, "Gradient boosting fitted worst of the three, and we report it as a sensitivity specification rather than "
                          "privileging any nuisance model (Supplementary Section S7).")


NUM.update({"dose_sentence": dose_sentence, "rr_sentence": rr_sentence, "overlap_paragraph": overlap_paragraph,
            "psi_est_sentence": psi_est_sentence})


# Second revision: indeterminate histopathology, fixed site mixture, dose endpoints, calibrated and raw gap scales.

def nuis_sentence():
    d = J("vr48_nuisance_audit.json")["families"]
    assert all(d[f]["s_star"]["platt"] == d[f]["s_star"]["platt_sigma_input"] for f in FAMS)
    mx = max(v["share_lower_tertile_second_branch"] for f in FAMS for v in d[f]["branch"].values())
    assert mx < 0.0001 + 1e-12
    return ("For the image-feature target this σ̂ is not on the learner's input. It enters only through the second branch of *p̄*, "
            "which was active for at most 0.01 percent of lower-tertile test lesions at the point tipping floor. For color "
            "variegation, an input-matched σ̂ left the point floors unchanged; we did not repeat this check in the bootstrap "
            "(Supplementary Section S7).")


def size_defs_sentence():
    d = J("vr9_closing.json")["families"]
    img = d["head_image"]["concepts"]["size"]
    H = ("tertile", "quartile", "quintile", "median", "slope", "within_slope", "rank_slope")
    F4 = ("head_tabular", "head_image", "lp", "ft")
    ok = lambda c: all(d[f]["concepts"][c][h]["M0"] > 0 > d[f]["concepts"][c][h]["M2"] for f in F4 for h in H)
    near = [d[f]["concepts"]["lesion_skin_contrast"][h]["M0"] for f in F4 for h in H if abs(d[f]["concepts"]["lesion_skin_contrast"][h]["M0"]) < 0.02]
    near_c = [d[f]["concepts"]["color_variegation"][h]["M0"] for f in F4 for h in H if abs(d[f]["concepts"]["color_variegation"][h]["M0"]) < 0.02]
    assert ok("color_variegation") and ok("lesion_skin_contrast") and len(near) == 2 and not near_c and img["within_slope"]["M2"] > 0
    return (f"The two concepts are robust along different axes. Size with image features is the strongest multiplicity-adjusted result "
            f"but rests on the tertile definition: with slope definitions its M2 contrast was {s(img['slope']['M2'])} and, within "
            f"patients, {s(img['within_slope']['M2'])}. Color variegation kept opposite point-estimate signs under all seven concept "
            f"definitions in all four families. Lesion-skin contrast did too, but two of its M0 estimates, both with median splits, "
            f"were {s(min(near), 3)} and {s(max(near), 3)}, with intervals that included zero. These are point estimates conditional "
            f"on the fitted models (Supplementary Section S4).")


_dose_sentence_rev2 = dose_sentence


def dose_sentence():
    t = _dose_sentence_rev2()
    d = _v46()
    mb = {(f, k): d[f][f"{k}|320"]["raw"]["mean_by_eta"] for f in FAMS for k in PRIMARY}
    lo = [v[0] for v in mb.values()]; hi = [v[-1] for v in mb.values()]
    assert min(lo) > 0 and max(hi) < 0
    small = min(hi, key=abs)
    old = "Within the same replicate, it changed sign in 120 of 120 replicates."
    assert old in t
    nsc = [d[f][f"{k}|320"]["raw"]["n_sign_change"] for f in FAMS for k in PRIMARY]
    assert all(n == 20 for n in nsc)
    return t.replace(old, f"Mean raw contrasts ranged from {s(min(lo))} to {s(max(lo))} at η = −1 and from {s(min(hi))} to "
                          f"{s(max(hi))} at η = +1, the closest to zero being {s(small)} for size with image features. Within a "
                          f"replicate, the contrast changed sign between the two extreme doses in 20 of 20 replicates in each of the "
                          f"six combinations.")


def bridge_sentence():
    d = _v22(); a = J("vr33_bridge_alt.json")["families"]; dp = _vp()
    sms = [J("vr34_semisynth.json")["families"], J("vr34_semisynth_gbm.json")["families"]]
    t_, i_ = d["families"]["tabular"], d["families"]["image"]
    sg = {k: sum(a[f][k]["n_concepts_975"] for f in FAMS) for k in ("mlp|all", "gbm|all")}
    cal = {k: [a[f][k]["concept_calibration"] for f in FAMS] for k in ("mlp|all", "gbm|all")}
    est = [sm[f]["summary"][k]["median"] for sm in sms for f in FAMS for k in ("bridge_slope", "bridge_slope_gbm")]
    tru = [sm[f]["summary"]["bridge_slope"]["true_g_median"] for sm in sms for f in FAMS]
    raw_ok = 0
    for f in FAMS:
        for c in CL:
            pr = d["families"][f]["concepts"][c]["predicted"]; e = dp["families"][f]["concepts"][c]["diff_ci95"]
            raw_ok += (pr > 0 and e[0] > 0) or (pr < 0 and e[1] < 0)
    return (f"Proposition 1 is exact only for Bayes-optimal learners, so for trained networks we compare predicted and observed "
            f"directions. Here the gap is computed from Platt-calibrated logits on the full test population. With a perceptron for ĝ, "
            f"the predicted concept-level gap had the observed sign in at least 97.5 percent of joint resamples in {sg['mlp|all']} of "
            f"10 combinations of concept and family; with gradient boosting, in {sg['gbm|all']} of 10. On the raw scale of Table 1, "
            f"the 95 percent interval of the gap excluded zero with the predicted sign in {raw_ok} of 10 cells. The ten cells share "
            f"patients and include correlated concepts, so these counts describe agreement and are not independent tests. Magnitude "
            f"depended on the propensity model, with calibration slopes of observed on predicted gap of {u(min(cal['mlp|all']))} to "
            f"{u(max(cal['mlp|all']))} and {u(min(cal['gbm|all']))} to {u(max(cal['gbm|all']))}. Lesion by lesion, the calibrated logit "
            f"gap rose with log ĝ with slope {u(t_['pointwise_slope'])} {ci(t_['pointwise_slope_ci'])} and {u(i_['pointwise_slope'])} "
            f"{ci(i_['pointwise_slope_ci'])}. In semi-synthetic designs where the identity holds exactly, the median slopes ranged from "
            f"{u(min(est))} to {u(max(est))}, and from {u(min(tru))} to {u(max(tru))} on the true log *g* (Supplementary Section S2).")


def site_sentence():
    a = J("vr20_audit.json")
    h = a["site_heterogeneity"]["color_variegation"]
    assert h["range"][0] > 0
    lo = a["leave_one_site_out"]
    assert min(v["color_variegation"][0] for f in lo for v in lo[f].values()) > 0 > max(v["color_variegation"][1] for f in lo for v in lo[f].values())
    return (f"Between the outer tertiles of color variegation, the log odds ratio of verification among recorded negatives, adjusted "
            f"for sex, body site and skin tone, was positive in all {h['k']} acquisition sites where it could be estimated. Its "
            f"site-pooled value was {u(h['pooled_DL'])} {ci(h['pooled_ci'])}, with substantial heterogeneity (I² = {u(h['I2'])}). This "
            f"is a conditional log odds ratio, not the crude ratio *B*_V. Leaving out one site at a time and retraining kept the signs "
            f"of both color variegation contrasts in both families (Supplementary Section S6).")


def marg_paragraph():
    c, B, fc = _marg()
    v35 = J("vr35_marginal_rr.json")["populations"]
    tst = _joint_floor(v35["test"]["color_variegation"]["joint_counts"], B)
    sz = _joint_floor(v35["cohort"]["size"]["joint_counts"], B)
    d49 = J("vr49_indeterminate.json")["codings"]
    alt = [d49[f"cohort|{k}|{cc}"] for k in ("exclude", "malign") for cc in ("color_variegation", "size")]
    assert all(e["log_or_verified_ci95"][1] < 0 and e["rr"] > 1 for e in alt)
    fl = [e["joint_floor"] for e in alt]
    w = J("vr47_counts_audit.json")["site_common_weights"]
    wc, ws = w["color_variegation"], w["size"]
    assert wc["joint_floor_fixed"] is None and sz == fc and len(wc["supported_sites"]) == 6
    big = sorted(J("vr41_marginal_site.json")["concepts"]["color_variegation"]["by_site"].values(), key=lambda v: -sum(v["n_pos"]))[:2]
    return (f"This post-inspection analysis concerns the marginal disease association between the outer tertiles and uses "
            f"whole-cohort counts. Because RR_Y = RR_D·*A* exactly, RR_D > 1 when *A* < RR_Y. A floor π₀¹ ≥ *s*_min on malignant "
            f"verification in the lower tertile gives *A* ≤ 1/*s*_min and RR_D ≥ *s*_min·RR_Y; a two-sided interval needs the floor "
            f"in both tertiles (Supplementary Section S1). For color variegation RR_Y was {u(c['rr_y'])} {ci(c['rr_y_ci95'])}, and the "
            f"verified-only log odds ratio was {s(c['log_or_verified'])} {ci(c['log_or_verified_ci95'])}. Both signs held together in "
            f"at least 95 percent of {B:,} patient-cluster resamples once the lower-tertile floor reached {fc:.2f}, and for size at the "
            f"same floor. This is a resampling threshold, not a confidence bound, and the concept and analysis were chosen after the "
            f"primary results had been seen. On the test population alone the threshold was {_fl(tst)}. Coding indeterminate "
            f"histopathology as unverified or as malignant kept both signs, with whole-cohort thresholds from {min(fl):.2f} to "
            f"{max(fl):.2f}. Sites weaken the result. The two sites with the most malignant lesions would need floors of "
            f"{u(big[0]['floor'])} and {u(big[1]['floor'])} on their own. A standardized mixture of the six sites "
            f"reached 95 percent at {u(wc['joint_floor'])} only when a resample dropped sites without verified lesions in a tertile; "
            f"with the six-site mixture fixed, color variegation reached at most {pct(max(wc['joint_counts_fixed'].values()) / B)} and "
            f"size reached 95 percent at {u(ws['joint_floor_fixed'])} (Supplementary Section S7). The crude benign ratio *B*_V was "
            f"{u(c['B_V'])} {ci(c['B_V_ci95'])}, a benchmark for *A* rather than an estimate of it (Supplementary Fig. S3).")


def psi_sim_sentence():
    st = J("vr28_psi_stress.json")["summary"]
    sms = [J("vr34_semisynth.json")["families"], J("vr34_semisynth_gbm.json")["families"]]
    lo = J("vr34_semisynth_local.json")["families"]
    ks = ("mlp_platt", "mlp_iso", "gbm_platt")
    cov = [sm[f]["summary"][k]["covers"] for sm in sms for f in FAMS for k in ks]
    assert max(sm[f]["summary"][k]["wrong"] for sm in sms for f in FAMS for k in ks) == 0
    lc_or = [lo[f]["summary"]["oracle"]["covers"] for f in FAMS]
    return (f"In simulation, an overstated floor made even the oracle set exclude zero on the wrong side in "
            f"{pct(st['adversarial|overstated']['oracle_wrong'])} of adversarial settings. In semi-synthetic designs on the ISIC-2024 "
            f"inputs, no wrong sign was observed at the true floor, but the plug-in sets contained the true target in {round(100 * min(cov))} "
            f"to {pct(max(cov))} of settings per design, family and estimator. Their lower endpoints were mostly below the oracle ones, "
            f"yet up to 0.23 above them when the target was negative. A floor that failed only where verification is rarest left the "
            f"oracle set containing the target in {round(100 * min(lc_or))} to {pct(max(lc_or))} of settings (Supplementary Section S2).")


def tail_sentence():
    d = J("vr38_logit_tail.json")["families"]["tabular"]
    pl, iso = d["mlp_platt"], d["mlp_iso"]
    a = J("vr48_nuisance_audit.json")
    t = a["families"]["tabular"]
    assert a["families"]["image"]["stats_mean_over_seeds"]["iso_at_lower"] == 0 and a["iso_zero"]["all"]["below"] < 0.0001
    return (f"Calibration approximately matches aggregate event risk, a sum of *q*, but not the mean of logit *q*, which is steep "
            f"near zero. In the lower tertile of color variegation with tabular features, Platt and isotonic calibration expected "
            f"{pl['0']['sum_q']:.1f} and {iso['0']['sum_q']:.1f} events against {pl['0']['observed']} observed, yet their mean logit *q* "
            f"differed by {abs(pl['0']['mean_logit_q'] - iso['0']['mean_logit_q']):.2f}. Isotonic calibration put "
            f"{pct(t['stats_mean_over_seeds']['iso_at_lower'])} of tabular test lesions at its lower bound of 10⁻⁶, almost all of them "
            f"exact zeros of the fit. Moving the bound to 10⁻⁵ or 10⁻⁷ moved the point floor from {u(t['s_star']['iso_1e-6'])} to "
            f"{u(t['s_star']['iso_1e-5'])} or {u(t['s_star']['iso_1e-7'])}, so that row of Table 2 is regularized by clipping. The "
            f"logit scale amplifies the rare-event tail, and the specifications of *q* express that tail differently.")


_t_psi_est_rev = t_psi_est


def t_psi_est():
    t = _t_psi_est_rev()
    old = "| tabular | perceptron, isotonic |"
    assert t.count(old) == 1
    return t.replace(old, "| tabular | perceptron, isotonic, clipped at 10⁻⁶ |")


NUM.update({"nuis_sentence": nuis_sentence, "size_defs_sentence": size_defs_sentence, "dose_sentence": dose_sentence,
            "bridge_sentence": bridge_sentence, "site_sentence": site_sentence, "marg_paragraph": marg_paragraph,
            "psi_sim_sentence": psi_sim_sentence, "tail_sentence": tail_sentence})
TABLE.update({"psi_est": t_psi_est})


_dose_sentence_rev3 = dose_sentence


def dose_sentence():
    t = _dose_sentence_rev3()
    old = (" These are bootstrap intervals over the 20 paired replicates, conditional on the cohort, patient split and test population. "
           "They average over the algorithmic randomness ω, including the validation-set draw used for early stopping, and do not "
           "reflect patient sampling.")
    assert old in t
    return t.replace(old, " Intervals are bootstrap intervals over the 20 paired replicates and do not reflect patient sampling.")


_bridge_rev = bridge_sentence


def bridge_sentence():
    t = _bridge_rev()
    a = t.index(" In semi-synthetic designs where the identity holds exactly")
    return t[:a] + " Even where the identity holds exactly, in semi-synthetic designs, trained learners did not attain slope one (Supplementary Section S2)."


NUM.update({"dose_sentence": dose_sentence, "bridge_sentence": bridge_sentence})


_psi_est_rev2 = psi_est_sentence


def psi_est_sentence():
    t = _psi_est_rev2()
    for old, new in (("Each point floor lies within 0.03 of exp(−Δ_q), computed from the mean logit *q* in the two strata. This approximation "
                      "explains the point estimates, and the floors themselves come from the exact bound.",
                      "Each point floor lies within 0.03 of exp(−Δ_q), which explains the differences; the floors themselves use the exact bound."),
                     ("With gradient boosting for the image-feature target, no floor up to one made ψ_L positive in 95 percent of replicates, "
                      "so nuisance uncertainty cannot be summarized by one threshold. ",
                      "With gradient boosting for the image-feature target, no floor up to one reached the 95 percent rule. ")):
        assert old in t, old[:50]
        t = t.replace(old, new)
    return t


_balanced_old = balanced_sentence


def balanced_sentence():
    t = _balanced_old()
    old = (" Color variegation with tabular features kept separated signs. These intervals condition on the fitted learners and are "
           "not part of the primary joint-bootstrap multiplicity analysis (Supplementary Section S6).")
    assert old in t
    return t.replace(old, " Color variegation with tabular features kept separated signs (Supplementary Section S6, intervals conditional on the fitted learners).")


NUM.update({"psi_est_sentence": psi_est_sentence, "balanced_sentence": balanced_sentence})


# Third revision: output measure, validation filter, marginal scope.

def clip_sentence():
    d = J("vr50_logit_clip.json")["families"]
    assert all(d[f]["M2"]["share_at_cap"] == 0 for f in FAMS) and d["tabular"]["M0"]["share_at_cap"] < 0.002
    im = d["image"]["M0"]; cc = im["contrasts"]
    assert all(cc[c]["clipped"] * cc[c]["direct"] > 0 and abs(cc[c]["direct"]) > abs(cc[c]["clipped"]) for c in cc)
    for f in FAMS:
        for c, x in d[f]["M0"]["contrasts"].items():
            assert x["clipped"] * x["direct"] > 0
    dif = [abs(cc[c]["direct"] - cc[c]["clipped"]) for c in cc]
    lo = cc["color_variegation"]["share_at_cap_lower"]; up = cc["color_variegation"]["share_at_cap_upper"]
    return (f"The logit cap of Section III-B binds only for M0 with image features, at {pct(im['share_at_cap'])} of test lesions, more often in the "
            f"lower tertile of color variegation ({pct(lo)}) than in the upper one ({pct(up)}). With the direct network logits, all "
            f"five contrasts kept their signs and moved away from zero by {u(min(dif))} to {u(max(dif))} (Supplementary Section S6).")


_marg_rev = marg_paragraph


def marg_paragraph():
    t = _marg_rev()
    v35 = J("vr35_marginal_rr.json")
    B = v35["B"]
    assert _joint_floor(v35["populations"]["test"]["size"]["joint_counts"], B) is None
    rep = [("This post-inspection analysis concerns the marginal disease association between the outer tertiles and uses "
            "whole-cohort counts.",
            "The learner-scale floor of Section IV-F concerns mean log odds on the test population. This post-inspection "
            "analysis instead concerns the marginal disease association, a ratio of mean risks between the outer tertiles, "
            "and uses crude pooled whole-cohort counts."),
           ("; a two-sided interval needs the floor in both tertiles (Supplementary Section S1).",
            ". A two-sided interval with this floor uses it in both tertiles; without an upper-tertile floor the observed law "
            "still bounds the upper-tertile risk by *q*₁ + 1 − σ₁ (Supplementary Section S1)."),
           ("On the test population alone the threshold was ", "On the test population alone, the threshold for color variegation was "),
           ("Coding indeterminate histopathology as unverified or as malignant kept both signs, with whole-cohort thresholds from",
            "There, size did not reach the 95 percent rule at any floor. Coding indeterminate histopathology as unverified or as "
            "malignant kept both signs, with whole-cohort thresholds for color variegation and size from"),
           ("The two sites with the most malignant lesions would need floors of",
            "At the two sites with the most malignant lesions, the sufficient point floors for color variegation were")]
    for a, b in rep:
        assert a in t, a[:40]
        t = t.replace(a, b)
    return t


NUM.update({"clip_sentence": clip_sentence, "marg_paragraph": marg_paragraph})


_marg_rev2 = marg_paragraph


def marg_paragraph():
    t = _marg_rev2()
    d49 = J("vr49_indeterminate.json")["codings"]
    fl = {d49[f"cohort|{k}|{c}"]["joint_floor"] for k in ("dataset", "exclude", "malign") for c in ("color_variegation", "size")}
    a = t.index("Coding indeterminate histopathology"); b = t.index("Sites weaken the result.")
    if len(fl) == 1:
        new = (f"Coding indeterminate histopathology as unverified or as malignant kept both signs and the whole-cohort threshold of "
               f"{fl.pop():.2f} for color variegation and size. ")
    else:
        new = (f"Coding indeterminate histopathology as unverified or as malignant kept both signs, with whole-cohort thresholds for "
               f"color variegation and size from {min(fl):.2f} to {max(fl):.2f}. ")
    return t[:a] + new + t[b:]


NUM.update({"marg_paragraph": marg_paragraph})
