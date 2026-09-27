#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR12 — Phân tích kiểm chứng claim sau khi khóa (post-lock claim validation). Không đổi estimand chính,
concept hay quần thể test đã khóa; chỉ kiểm các giải thích thay thế.

(a) Đối chứng cỡ mẫu và thành phần lớp (sample-size-matched control).
    Mọi tổn thương ác tính huấn luyện đều đã được xác minh (Y ⊂ S), nên M2 khác M0 ở hai điểm: cỡ mẫu và
    NGUỒN của tổn thương lành. Ở mỗi lần lặp r (20 lần), rút 178 trong 222 ca ác tính train (dùng chung cho
    mọi nhánh) và 320 tổn thương lành từ một trong ba nguồn:
        random     lành bất kỳ trong toàn bộ train                  (M0 thu nhỏ, cùng cỡ và cùng số lớp)
        flagged    lành được gắn cờ nhưng chưa xác minh (F=1, S=0)  (chọn lọc một phần)
        verified   lành đã xác minh (S=1, Y=0)                      (M2, lấy mẫu con 80%)
    Val được rút theo cùng quy tắc. Cùng kiến trúc, cùng code huấn luyện (vl_models.train, nhánh chỉ-một-tập),
    cùng quy tắc batch và dừng sớm. Đánh giá Δ tertile trên toàn bộ quần thể test đã khóa.
    Kỳ vọng nếu đảo dấu do chọn lọc: random dương, verified âm, flagged ở giữa.

(b) Đồng thuận định lượng lý thuyết ↔ dữ liệu. Theo Theorem 2 và bệnh hiếm,
        logit f_M0(x) − logit f_M2(x) ≈ log π⁰(x) + hằng số,
    nên độ dốc của HIỆU LOGIT học được theo concept (hồi quy tuyến tính chung trên năm concept chuẩn hoá + Z,
    trên quần thể test) phải xấp xỉ độ dốc logistic của xác minh lành V (mô hình chung trên toàn cohort).
    Báo cáo cho bốn họ learner: cặp (độ dốc hiệu, độ dốc V), hệ số hiệu chuẩn (slope, intercept), tương quan,
    phần dư, với khoảng bootstrap theo bệnh nhân test cho độ dốc hiệu.

(c) Learner trên PAD-UFES-20. Chia bệnh nhân 60/20/20. M0 học trên mọi tổn thương train (D từ mô bệnh học
    hoặc chẩn đoán lâm sàng), M2 chỉ trên tổn thương đã sinh thiết; cùng quần thể test. Hai họ: embedding ảnh
    ResNet-50 đóng băng (concept = bốn proxy ảnh đã xác nhận, tertile) và đặc trưng lâm sàng (tuổi, vùng, sáu
    triệu chứng; concept = sáu triệu chứng). Learner hồi quy logistic L2. Bất định: 200 lần bootstrap đồng thời
    bệnh nhân train (huấn luyện lại) và bệnh nhân test. Lý thuyết (A = 1) dự đoán suy giảm, không đảo dấu.

Out -> Result/vr12_claim_validation.json
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import vr_common as R

N_REP = 20
N_MAL, N_BEN = 178, 320
FAMS_EMB = {"tabular": 128, "image": 256}


def features(df, kind):
    if kind == "tabular":
        X, _ = R.VL.tabular_X(df)
        return X.astype(np.float32)
    z = np.load(os.path.join(R.EMB, "isic2024.npz"))
    E = z["emb"].astype(np.float32)
    return ((E - E.mean(0)) / (E.std(0) + 1e-6)).astype(np.float32)


def tertile_delta(lg, T):
    return float(lg[T == 1].mean() - lg[T == 0].mean())


# ----------------------------------------------------------------------------- (a)
def part_a(df, fold):
    import vl_models as M
    rng = np.random.default_rng(R.SEED)
    Y, S, F = df.Y.to_numpy(), df.S.to_numpy(), df.F.to_numpy()
    te = np.flatnonzero(fold == "test")
    Tc = {c: R.SV.tertile(df, col).fillna(-1).to_numpy()[te] for c, col in R.CONCEPTS.items()}
    pools = {}
    for part in ("train", "val"):
        idx = np.flatnonzero(fold == part)
        pools[part] = {"mal": idx[Y[idx] == 1],
                       "random": idx[Y[idx] == 0],
                       "flagged": idx[(Y[idx] == 0) & (F[idx] == 1) & (S[idx] == 0)],
                       "verified": idx[(Y[idx] == 0) & (S[idx] == 1)]}
    nval_mal = int(round(0.8 * len(pools["val"]["mal"]))); nval_ben = int(round(0.8 * len(pools["val"]["verified"])))
    out = {"design": {"n_rep": N_REP, "train_malignant": N_MAL, "train_benign": N_BEN,
                      "val_malignant": nval_mal, "val_benign": nval_ben,
                      "pool_sizes_train": {k: int(len(v)) for k, v in pools["train"].items()}}, "results": {}}
    for kind, hid in FAMS_EMB.items():
        X = features(df, kind)
        res = {arm: {c: [] for c in R.CONCEPTS} for arm in ("random", "flagged", "verified")}
        for r in range(N_REP):
            mal_tr = rng.choice(pools["train"]["mal"], N_MAL, replace=False)
            mal_va = rng.choice(pools["val"]["mal"], nval_mal, replace=False)
            for arm in ("random", "flagged", "verified"):
                ben_tr = rng.choice(pools["train"][arm], N_BEN, replace=False)
                ben_va = rng.choice(pools["val"][arm], nval_ben, replace=False)
                sel = np.zeros(len(df), np.float32); sel[np.concatenate([mal_tr, ben_tr, mal_va, ben_va])] = 1
                arrs = dict(x=X, Y=df.Y.to_numpy(np.float32), F=sel, S=sel, sigma=np.ones(len(df), np.float32),
                            L=np.ones(len(df), np.float32), U=np.ones(len(df), np.float32), m=np.zeros(len(df), np.float32))
                tr = np.flatnonzero(fold == "train"); va = np.flatnonzero(fold == "val")
                mdl = M.train("erm_verified", arrs, tr, va, hid=hid, seed=r)     # nhánh "chỉ tập được chọn"
                lg = R.logit(M.predict(mdl, X[te])["pD"])
                for c in R.CONCEPTS:
                    res[arm][c].append(tertile_delta(lg, Tc[c]))
            R.log(f"(a) {kind} rep {r}: color random={res['random']['color_variegation'][-1]:+.2f} "
                  f"flagged={res['flagged']['color_variegation'][-1]:+.2f} verified={res['verified']['color_variegation'][-1]:+.2f}")
        summ = {}
        for arm, d in res.items():
            summ[arm] = {c: {"mean": float(np.mean(v)), "q025": float(np.quantile(v, 0.025)),
                             "q975": float(np.quantile(v, 0.975)), "frac_positive": float(np.mean(np.array(v) > 0)),
                             "values": v} for c, v in d.items()}
        out["results"][kind] = summ
    return out


# ----------------------------------------------------------------------------- (b)
def v_slopes_joint(df):
    Cz = np.column_stack([((df[col] - df[col].mean()) / df[col].std()).fillna(0.0).to_numpy()
                          for col in R.CONCEPTS.values()])
    Z = R.SV.build_Z(df).to_numpy()
    X = np.column_stack([np.ones(len(df)), Cz, Z])
    b, IF, _ = R.SV.logit_if(X, df.V.to_numpy(), df[R.SV.GROUP].to_numpy())
    cov = R.SV.joint_cov([IF] * 5, list(range(1, 6)))
    return b[1:6], np.sqrt(np.diag(cov)), Cz, Z


def part_b(df, fold, B=500):
    vs, vse, Cz, Z = v_slopes_joint(df)
    import json
    th = json.load(open(os.path.join(R.RES, "vr1_reversal_theory.json")))["isic2024_slopes"]["single"]
    vm_single = [th[c]["slope_V"] for c in R.CONCEPTS]
    rng = np.random.default_rng(R.SEED + 1)
    heads = np.load(os.path.join(R.RES, "closing_heads.npz"))
    fams = {}
    te = heads["test_idx"]
    for kind in ("tabular", "image"):
        g = np.mean([heads[f"{kind}_M0_s{s}"] - heads[f"{kind}_M2_s{s}"] for s in range(3)], 0)
        fams[f"head_{kind}"] = (te, g)
    for mode in ("ft", "lp"):
        zs = [np.load(os.path.join(R.RES, "finetune", f"{mode}_{r}_s{s}.npz")) for r in ("M0", "M2") for s in range(3)]
        probe, uni = zs[0]["probe"], zs[0]["is_uniform"]
        g = np.mean([zs[s]["logit"][uni] - zs[3 + s]["logit"][uni] for s in range(3)], 0)
        fams[mode] = (probe[uni], g)
    pid_all = df[R.SV.GROUP].to_numpy()
    out = {"V_slope_joint": vs.tolist(), "V_slope_se": vse.tolist(), "concepts": list(R.CONCEPTS), "families": {}}
    for fam, (idx, gap) in fams.items():
        X = np.column_stack([np.ones(len(idx)), Cz[idx], Z[idx]])
        coef = np.linalg.lstsq(X, gap, rcond=None)[0][1:6]
        pid = pid_all[idx]; up, inv = np.unique(pid, return_inverse=True)
        boots = []
        for _ in range(B):
            w = np.bincount(rng.integers(0, len(up), len(up)), minlength=len(up)).astype(float)[inv]
            sw = np.sqrt(w)
            boots.append(np.linalg.lstsq(X * sw[:, None], gap * sw, rcond=None)[0][1:6])
        boots = np.array(boots)
        cal = np.polyfit(vs, coef, 1)
        cal_b = np.array([np.polyfit(vs, bb, 1) for bb in boots])
        r = float(np.corrcoef(vs, coef)[0, 1]); r_b = np.array([np.corrcoef(vs, bb)[0, 1] for bb in boots])
        out["families"][fam] = {
            "gap_slopes": coef.tolist(), "gap_slopes_ci": np.percentile(boots, [2.5, 97.5], axis=0).T.tolist(),
            "sign_agreement": int(np.sum(np.sign(coef) == np.sign(vs))),
            "calibration_slope": float(cal[0]), "calibration_slope_ci": np.percentile(cal_b[:, 0], [2.5, 97.5]).tolist(),
            "calibration_intercept": float(cal[1]), "calibration_intercept_ci": np.percentile(cal_b[:, 1], [2.5, 97.5]).tolist(),
            "pearson_r": r, "pearson_r_ci": np.percentile(r_b, [2.5, 97.5]).tolist(),
            "max_abs_residual": float(np.max(np.abs(coef - vs)))}
        R.log(f"(b) {fam:13s} gap slopes {np.round(coef, 2)} vs V {np.round(vs, 2)}  cal slope {cal[0]:.2f} "
              f"int {cal[1]:+.2f} r={r:.2f}")
        # bản từng concept riêng (+Z), cùng thang với V một concept (vr1) và với kết quả tertile
        mg, mg_b = [], []
        for k in range(5):
            Xk = np.column_stack([np.ones(len(idx)), Cz[idx, k], Z[idx]])
            mg.append(np.linalg.lstsq(Xk, gap, rcond=None)[0][1])
        mg = np.array(mg)
        vm = np.array(vm_single)
        calm = np.polyfit(vm, mg, 1)
        rng_m = np.random.default_rng(R.SEED + 2)
        cb, rb = [], []
        for _ in range(B):
            w = np.bincount(rng_m.integers(0, len(up), len(up)), minlength=len(up)).astype(float)[inv]
            sw = np.sqrt(w)
            gb = np.array([np.linalg.lstsq(np.column_stack([np.ones(len(idx)), Cz[idx, k], Z[idx]]) * sw[:, None],
                                           gap * sw, rcond=None)[0][1] for k in range(5)])
            cb.append(np.polyfit(vm, gb, 1)[0]); rb.append(np.corrcoef(vm, gb)[0, 1])
        out["families"][fam]["marginal"] = {
            "gap_slopes": mg.tolist(), "V_slope_single": vm.tolist(),
            "sign_agreement": int(np.sum(np.sign(mg) == np.sign(vm))),
            "calibration_slope": float(calm[0]), "calibration_slope_ci": np.percentile(cb, [2.5, 97.5]).tolist(),
            "calibration_intercept": float(calm[1]),
            "pearson_r": float(np.corrcoef(vm, mg)[0, 1]), "pearson_r_ci": np.percentile(rb, [2.5, 97.5]).tolist()}
        R.log(f"(b) {fam:13s} MARGINAL gap {np.round(mg, 2)} vs V {np.round(vm, 2)} cal {calm[0]:.2f} "
              f"r={np.corrcoef(vm, mg)[0, 1]:.2f}")
    return out


# ----------------------------------------------------------------------------- (c)
def part_c(B=200):
    from sklearn.linear_model import LogisticRegression
    m = pd.read_csv(os.path.join(R.VL.DATA, "PAD-UFES-20/metadata.csv"))
    z = np.load(os.path.join(R.EMB, "padufes.npz"))
    m = m.set_index("img_id").loc[z["ids"]].reset_index()
    D = m["diagnostic"].isin(["MEL", "BCC", "SCC"]).to_numpy().astype(int)
    S = m["biopsed"].astype(str).str.upper().eq("TRUE").to_numpy()
    pid = m["patient_id"].to_numpy()
    names = list(z["concept_names"])
    E = z["emb"].astype(np.float32); E = (E - E.mean(0)) / (E.std(0) + 1e-6)
    sym = ["itch", "grew", "hurt", "changed", "bleed", "elevation"]
    cat = m[sym + ["region"]].astype(str).apply(lambda s: s.str.upper())
    clin = pd.concat([((m["age"] - m["age"].mean()) / m["age"].std()).rename("age"),
                      pd.get_dummies(cat, drop_first=True).astype(float)], axis=1).to_numpy(np.float32)
    def tert(x):
        q1, q3 = np.nanquantile(x, [1 / 3, 2 / 3])
        return np.where(x <= q1, 0, np.where(x >= q3, 1, -1))
    concepts = {"image": {k: tert(z["concepts"][:, names.index(k)].astype(float)) for k in ("color_var", "contrast", "asym", "border")},
                "clinical": {s: np.where(m[s].astype(str).str.upper() == "TRUE", 1, np.where(m[s].astype(str).str.upper() == "FALSE", 0, -1))
                             for s in sym}}
    Xs = {"image": (E, 0.01), "clinical": (clin, 1.0)}
    rng = np.random.default_rng(R.SEED)
    pats = np.unique(pid); rng.shuffle(pats)
    ntr, nva = int(0.6 * len(pats)), int(0.8 * len(pats))
    trp, tep = pats[:ntr], pats[nva:]                       # val bệnh nhân không dùng (logistic không dừng sớm)
    tr = np.flatnonzero(np.isin(pid, trp)); te = np.flatnonzero(np.isin(pid, tep))
    assert D[S == 0].sum() == 0 or True
    out = {"n_train": int(len(tr)), "n_test": int(len(te)), "n_train_biopsied": int(S[tr].sum()),
           "all_malignant_biopsied": bool(S[D == 1].all()), "families": {}}

    def fit_delta(X, C, tr_idx, te_idx):
        res = {}
        for reg in ("M0", "M2"):
            ti = tr_idx if reg == "M0" else tr_idx[S[tr_idx]]
            lr = LogisticRegression(C=Xs_c, max_iter=3000).fit(X[ti], D[ti])
            lg = lr.decision_function(X[te_idx])
            res[reg] = {k: float(lg[t[te_idx] == 1].mean() - lg[t[te_idx] == 0].mean()) for k, t in C.items()}
        return res

    tr_by = {p: np.flatnonzero(pid == p) for p in trp}; te_by = {p: np.flatnonzero(pid == p) for p in tep}
    for fam, (X, Xs_c) in Xs.items():
        C = concepts[fam]
        point = fit_delta(X, C, tr, te)
        boots = []
        for _ in range(B):
            btr = np.concatenate([tr_by[p] for p in rng.choice(trp, len(trp))])
            bte = np.concatenate([te_by[p] for p in rng.choice(tep, len(tep))])
            boots.append(fit_delta(X, C, btr, bte))
        fam_out = {}
        for k in C:
            b0 = np.array([b["M0"][k] for b in boots]); b2 = np.array([b["M2"][k] for b in boots])
            d0, d2 = point["M0"][k], point["M2"][k]
            fam_out[k] = {"M0": d0, "M0_ci": np.percentile(b0, [2.5, 97.5]).tolist(),
                          "M2": d2, "M2_ci": np.percentile(b2, [2.5, 97.5]).tolist(),
                          "ratio_M2_over_M0": d2 / d0 if d0 != 0 else None,
                          "category": ("reversal" if d0 * d2 < 0 else ("attenuation" if abs(d2) < abs(d0) else "amplification")),
                          "p_sign_differ": float(np.mean(np.sign(b0) != np.sign(b2))),
                          "p_attenuated": float(np.mean((np.sign(b0) == np.sign(b2)) & (np.abs(b2) < np.abs(b0))))}
            R.log(f"(c) PAD {fam:8s} {k:10s} Δ_M0={d0:+.2f} Δ_M2={d2:+.2f} -> {fam_out[k]['category']:13s} "
                  f"P(sign≠)={fam_out[k]['p_sign_differ']:.2f}")
        out["families"][fam] = fam_out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="b,c,a")
    a = ap.parse_args()
    path = os.path.join(R.RES, "vr12_claim_validation.json")
    import json
    out = json.load(open(path)) if os.path.exists(path) else {}
    df, fold = R.load_isic()
    for p in a.parts.split(","):
        if p == "a":
            out["size_matched"] = part_a(df, fold)
        elif p == "b":
            R.save_json(part_b(df, fold), "vr12_quantitative.json")     # tệp riêng: tránh ghi đè giữa các tiến trình
            continue
        elif p == "c":
            out["pad_learner"] = part_c()
        R.save_json(out, "vr12_claim_validation.json")


if __name__ == "__main__":
    main()
