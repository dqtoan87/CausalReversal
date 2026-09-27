# Khóa phân tích VILRR

Khóa ngày 2026-09-25. Sau file này, dừng thí nghiệm và chuyển sang viết bài. Mọi số trong bản thảo phải truy được về một JSON trong `Result/`.

## 1. Checklist

| Mục | Trạng thái | Bằng chứng |
| --- | --- | --- |
| Bootstrap theo bệnh nhân cho đảo dấu chính | Xong | `vr9_closing.json`, 2,000 lần lấy lại bệnh nhân, 207–209 bệnh nhân test |
| CI cho Δ_M0, Δ_M2, Δ_M0 − Δ_M2 | Xong | `vr9_closing.json`, cả bốn họ learner |
| Độ nhạy với định nghĩa concept | Xong | 7 định nghĩa: tertile, quartile, quintile, median, slope, within_slope, rank_slope |
| Kiểm chứng ký hiệu T4/T5 | Xong, đạt | `vr10_symbolic.json`, `all_passed = true` |
| Unit test các trường hợp biên | Xong, 25/25 đạt | `Code/test_reversal_theory.py` |
| Khóa ký hiệu | Xong | Mục 2 |
| Khóa outcome chính và phụ | Xong | Mục 3 |
| Đánh dấu phân tích thăm dò | Xong | Mục 3 |
| Loại metric E0 → E1 → E2 | Xong | Mục 4 |
| Không dùng proxy size từ tile ảnh | Xong | Mục 4 |
| Không claim representation reversal tổng quát | Xong | Mục 5 |
| Không claim ICDL vượt trội trên mọi cohort | Xong | Mục 5 |

## 2. Ký hiệu (dùng thống nhất trong bài và trong code)

| Ký hiệu | Nghĩa | Tên trong code |
| --- | --- | --- |
| D | bệnh thật (ẩn ngoài tập đã xác minh) | `D` |
| F | được gắn cờ "lesion of interest" | `F` |
| S | có mô bệnh học (xác minh), S ⊂ F | `S` |
| Y = D·S | nhãn ác tính ghi nhận | `Y` |
| t ∈ {0, 1} | tertile dưới và trên của concept, cắt trên toàn cohort | `SV.tertile` |
| π_t¹, π_t⁰ | P(S=1 \| D=1, t), P(S=1 \| D=0, t) | `pi1`, `pi0` |
| A = π_1¹/π_0¹, B = π_1⁰/π_0⁰ | tỉ số xác minh ác tính và lành | `logA`, `logB` |
| θ = log OR_D | tương phản bệnh thật | `theta` |
| δ = log B − log A | tương phản chọn lọc vi phân | `delta` |
| q(x), σ(x) | P(Y=1 \| x), P(S=1 \| x) | `q`, `sigma` |
| s(x), s_min | P(S=1 \| D=1, x) và sàn giả định | `s`, `s_min` |
| 𝓘_D(x; s) | khoảng định danh của P(D=1 \| x) | `disease_interval` |
| M0, M2 | learner trên Y toàn bộ; learner chỉ trên S=1 | `erm_y`, `erm_verified` |
| Δ_m,k | tương phản học được: logit TB tertile trên − dưới, trên cùng quần thể test | `vr9` |

Đánh số trong bài (khác với đánh số trong proposal):

| Trong bài | Trong proposal và code |
| --- | --- |
| Theorem 1 (điều kiện đảo dấu) | T4 |
| Theorem 2 (đảo dấu của learner) | T5 |
| Proposition 1 (cận tương phản bệnh) | T1 |
| Proposition 2 (khoảng của B) | T2 |
| Proposition 3 (khoảng định danh từng tổn thương) | T3 |

## 3. Outcome

**Chính (confirmatory).**
- P1. Đảo dấu của tương phản học được giữa M0 và M2 trên cùng quần thể test, định nghĩa tertile, ở hai họ head (tabular, ảnh đóng băng), với CI bootstrap theo bệnh nhân. Ba concept chính theo v3: color variegation, size, lesion-skin contrast.
- P2. Khớp giữa đảo dấu quan sát và Theorem 1–2: lưới phase diagram (648 điểm), learner MLP hữu hạn mẫu (24 điểm), và đồng nhất thức độ dốc trên ISIC-2024.
- P3. PAD-UFES là trường hợp biên: A = 1 chính xác, suy giảm mà không đảo dấu.

**Phụ (secondary).**
- S1. Cùng P1 cho fine-tune và linear probe (4 họ learner).
- S2. Asymmetry và border irregularity.
- S3. Sáu định nghĩa concept thay thế.
- S4. Proposition 1–3 trên ISIC-2024.

**Thăm dò (exploratory), phải được ghi nhãn như vậy trong bài.**
- E1. ICDL, dự đoán theo giả định, tipping point cá thể.
- E2. ℋ có cấu trúc (theo tầng, dùng chung).
- E3. Chẩn đoán representation: probe, alignment, can thiệp layer 3, readout chung, CKA.

## 4. Loại bỏ

- Metric đổi dấu E0 → E1 → E2: không hợp lệ, vì nó trộn hành vi mô hình với chọn lọc quần thể đánh giá. Không dùng ở bất kỳ đâu.
- Proxy size đo từ tile ảnh: Spearman ≈ 0 với TBP. Size chỉ xuất hiện qua giá trị TBP.
- SVCL (L_stage, L_sep): không tạo khác biệt trong ablation. Không trình bày như một phương pháp.
- AUROC_D trên tập đã xác minh không được dùng để xếp hạng phương pháp, vì chính nó bị Berkson.

## 5. Claim được phép và không được phép

| Được phép | Không được phép |
| --- | --- |
| "can reverse", trong điều kiện δ vượt θ | đảo dấu là phổ quát |
| decision/learning reversal | representation reversal tổng quát |
| đảo dấu vững nhất ở color variegation (4/4 họ learner, 7/7 định nghĩa) | size đảo dấu vững sau fine-tune (CI của M2 chứa 0; 5/7 định nghĩa) |
| ICDL tránh dự đoán ngoài tập định danh, an toàn khi s giả định ≤ s thật | ICDL vượt ERM hay vượt M2 |
| thêm cấu trúc cho ℋ làm hẹp khoảng nhưng có thể làm mất coverage | ℋ theo tầng cho cận đáng tin trên ISIC-2024 |
| PAD-UFES phù hợp với biên của Theorem 1 | PAD-UFES chứng minh ICDL tổng quát hoá |
| M0 − M2 ≈ độ dốc xác minh lành (xấp xỉ, phần dư −0.11 đến +0.19) | đồng nhất thức đó đúng chính xác với learner một concept |

## 6. Kết quả then chốt đã khóa (nguồn)

| Kết quả | Giá trị | Nguồn |
| --- | --- | --- |
| Theorem 1, sai số đồng nhất thức tối đa | 1.8e-15 trên 20,000 mô hình | `vr1_reversal_theory.json` |
| Theorem 2, log-tuyến tính / sàn s ≥ 0.5 | sai số 1.1e-16 / 0.022, khớp đảo dấu 100% / 100% | `vr1_reversal_theory.json` |
| Phase diagram | 648 điểm: 469 đảo dấu, 128 suy giảm, 51 khuếch đại; khớp 0.998, từ quan sát 0.995, MLP 24/24 | `vr2_phase_diagram.json` |
| Đảo dấu learner, head tabular | color +1.02 [0.76, 1.25] / −0.95 [−1.07, −0.83] | `vr9_closing.json` |
| PAD-UFES | 10/11 suy giảm, 0 đảo dấu; color ảnh θ = 2.01, log B = 0.75 | `vr6_pad_boundary.json` |
| Fine-tune | Trường hợp B; readout chung 0/4 đảo dấu | `vr5_representation.json` |

## 7. Phân tích kiểm chứng claim sau khóa (post-lock claim validation), 2026-09-25

Các phân tích này kiểm các giải thích thay thế. Chúng KHÔNG đổi estimand chính, concept, định nghĩa tertile hay quần thể test đã khóa, và không đổi outcome chính dựa trên kết quả của chúng.

| Phân tích | Mục đích | Script / đầu ra |
| --- | --- | --- |
| Đối chứng khớp cỡ mẫu và số lớp, 20 lần lặp, 3 nguồn lành (ngẫu nhiên / gắn cờ / đã xác minh) | loại giải thích do cỡ mẫu; đo biến thiên tập huấn luyện | `vr12_claim_validation.py --parts a` → `vr12_claim_validation.json` |
| Đồng thuận định lượng: độ dốc hiệu logit M0 − M2 so với độ dốc xác minh lành, từng concept riêng (chính) và mô hình chung (độ nhạy) | nâng bằng chứng từ khớp dấu lên khớp định lượng | `vr12_claim_validation.py --parts b` → `vr12_quantitative.json` |
| Đổi head trên encoder fine-tune, 5 seed (thêm seed 3, 4) | định vị cơ chế; bằng chứng, không phải chứng minh | `vr13_headswap.py` → `vr13_headswap.json` |
| Learner M0/M2 trên PAD-UFES-20 | kiểm chuỗi lý thuyết → learner trên cohort thứ hai | `vr12_claim_validation.py --parts c` → `vr12_claim_validation.json` |

Không làm: learner ISIC-2019 (nhiễu với dịch chuyển miền), thêm kiến trúc hay loss, cố chứng minh representation reversal.

Cập nhật claim được phép: đồng thuận định lượng chỉ gần 1:1 cho learner đặc trưng đóng băng (hệ số hiệu chuẩn 1.13 và 1.15); fine-tune khuếch đại (3.18), linear probe thu nhỏ (0.63). Mô hình chung năm concept khớp kém (r từ 0.21 đến 0.89), nên thang từng concept riêng là thang so sánh hợp lệ, trùng estimand chính.

## 8. Vòng kiểm chứng thứ hai (2026-09-25)

Không đổi estimand chính, concept, tertile hay quần thể test. Thuật ngữ được khóa lại:
- θ_k: mục tiêu bệnh (log OR_D), không định danh điểm.
- Δ_{m,k}: tương phản học được.
- learner reversal: dấu Δ_M0 ≠ dấu Δ_M2 (quan sát được; nguyên nhân là chính sách chọn mẫu, can thiệp được).
- disease-relative reversal: dấu Δ_M2 ≠ dấu θ_k; chỉ khẳng định khi tập định danh xác định dấu θ_k (Proposition 2, s_min).

Đánh số trong bài: Proposition 1 = khoảng cách learner (mới, chính xác, phi tham số): logit f0 − logit f2 = log P(S=1|x,Y=0); Theorem 1 = điều kiện đảo dấu (hệ số thiên lệch chọn lọc kinh điển); Theorem 2 = learner logistic dưới LR log-tuyến tính; Proposition 2 = cận RR_D; Proposition 3 = khoảng của B; Proposition 4 = khoảng cấp tổn thương.

| Phân tích | Script / đầu ra |
| --- | --- |
| Proposition 1 trên dữ liệu, sau hiệu chỉnh Platt, ĝ cùng lớp MLP | `vr17_joint_bootstrap.py` (phần 1) → `calibrated_bridge` |
| Bootstrap đồng thời train + test, 200 lần, 2 họ chính, có Bonferroni | `vr17_joint_bootstrap.py` (phần 2) |
| Can thiệp liều chọn lọc η trên tổn thương lành | `vr15_dose_response.py` |
| Trong / giữa bệnh nhân, theo cơ sở, vùng hỗ trợ | `vr16_mechanism.py` |
| Ngữ nghĩa nhãn ISIC-2024 | bài mô tả SLICE-3D: tile không gắn báo cáo giải phẫu bệnh được gán "benign NOS" |

Claim không được phép: "B quan sát được" (chỉ B_V và g quan sát được); "A = 1 chính xác" với PAD (chỉ với chẩn đoán được ghi nhận); "rules out"; phân rã nhân quả head/encoder duy nhất; hệ số khớp độ lớn đúng cho learner phi tuyến.

## 9. Vòng kiểm chứng thứ ba (2026-09-25)

Không đổi concept, tertile, quần thể test hay họ learner. Các thay đổi dưới đây được viết vào code trước khi chạy vòng này (kết quả tạm thời của vùng hỗ trợ được xem trong lúc chạy, không làm thay đổi thiết kế):
- Estimand bệnh chính chuyển sang thang của learner: ψ_k = E[logit p | t=1] − E[logit p | t=0] (Proposition 2 mới, tập định danh sắc). θ_k = log OR_D chỉ còn là đối tượng kinh điển của Lemma 1.
- Proposition 1 viết ở dạng tổng quát theo biến chọn mẫu R (r₁ hằng số, r₀(x)); M2 và liều η là hai trường hợp riêng. Dạng theo input cho learner ảnh (xấp xỉ chuẩn).
- Đánh số: Proposition 1 = offset chọn mẫu; Lemma 1 = hệ số chọn lọc (Theorem 1 cũ); Lemma 2 = learner logistic (Theorem 2 cũ); Proposition 2 = tập định danh ψ. Cận RR_D, khoảng B, khoảng cấp tổn thương chuyển vào phụ lục (S1).
- Suy diễn chính: 6 so sánh (3 concept chính × 2 họ đặc trưng đóng băng), 1,000 lần bootstrap đồng thời train + test có huấn luyện lại, Bonferroni phân vị ở mức 1 − 0.05/6. Robust ⇔ khoảng Bonferroni của Δ_M0 và Δ_M2 nằm hai phía của 0. Concept phụ: khoảng 95%, ngoài họ bội.
- Vùng hỗ trợ: ba ngưỡng σ̂ (phân vị 1, 5, 10 trong tổn thương đã xác minh ở train) là phân tích độ nhạy; báo cáo cả khi reversal không giữ.

| Phân tích | Script / đầu ra |
| --- | --- |
| Bootstrap chính, ψ, điểm tới hạn s*, vùng hỗ trợ | `vr19_primary_bootstrap.py` → `vr19_primary_bootstrap.json`, `vr19/` |
| Liều η đã hiệu chỉnh, dự đoán theo input, ESS; nhánh cùng cỡ với dự đoán định lượng | `vr21_dose_calibrated.py` |
| Proposition 1 cấp tổn thương, 200 lần bootstrap đồng thời | `vr22_pointwise_bridge.py` |
| Không đồng nhất giữa cơ sở (DL, I²), bỏ từng cơ sở, rà soát concept trùng, rà soát PAD | `vr20_audit.py` |
| Mô phỏng Proposition 2 (oracle, plug-in, concept tương quan) | `vr23_psi_sim.py` |
| Kiểm ký hiệu P1.4 (offset tổng quát); 30 unit test | `vr10_symbolic.py`, `test_reversal_theory.py` |

Claim không được phép thêm: reversal bên trong vùng hỗ trợ của tổn thương đã xác minh (Δ_M0 co về 0 ở đó); độ dốc liều "khớp" với dự đoán cho mọi concept (chỉ trong khoảng 25 percent cho color); "B_V ổn định giữa cơ sở" (I² lớn).

**Kết quả cuối vòng ba (2026-09-25, B = 1.000).** Bền sau Bonferroni: 2/6 ô chính (color và size với đặc trưng ảnh). Color với đặc trưng bảng chỉ bền ở mức 95% (khoảng Bonferroni của Δ_M0 là [−0,11; 2,06]). Khoảng của khoảng cách Δ_M0 − Δ_M2 loại trừ 0 ở 10/10 ô. s* của color: 0,42 (bảng) và 0,50 (ảnh); phân vị 95: 0,77 và 0,79. Vùng hỗ trợ: M2 âm ở 18/18, khoảng M0 chạm 0 ở 15/18. Độ dốc từng tổn thương 0,89 [0,67; 1,09] (bảng), 0,86 [0,66; 1,08] (ảnh). Không được viết "bền trong cả hai họ" cho color.

## 10. Vòng kiểm chứng thứ tư (2026-09-25)

Không đổi concept, tertile, quần thể test, họ learner hay quy tắc robust. Các phân tích dưới đây được thêm SAU khi đã thấy kết quả chính; chúng là chẩn đoán bổ sung, không phải phân tích chính đã khóa.
- Proposition 1 phát biểu trực tiếp trên input của learner: r_y(x̃) = P(R=1 | Y=y, x̃). Bước lấy trung bình r₀(x̃) = E[r₀(X) | x̃, Y=0] chỉ đúng khi R ⟂ X̃ | (X, Y).
- Giả định sàn của Proposition 2 là sàn THEO TỪNG ĐIỂM s(x̃) ≥ s_min, không phải tỉ lệ ác tính được xác minh. Trên ISIC chỉ có tập plug-in (q̂ = M0 hiệu chỉnh), không phải tập định danh quần thể.
- Bootstrap chính mở rộng lên 5.000 lần mỗi họ (vr19, cùng quy tắc). Thêm bootstrap lấy lại cả val (vr24, 500 lần), sai số Monte Carlo và biến cố đồng thời ψ_L > 0 và Δ_M2 < 0 (vr29).
- s* điểm = trên đường ψ_L trung bình qua seed (không còn lấy max qua seed).
- Liều chọn mẫu chính: lấy mẫu Poisson với π_η = min{1, κ exp(η z)} biết chính xác (vr26); thiết kế Gumbel (vr21) thành phụ, có định lượng độ lệch xác suất đưa vào.

| Phân tích | Script / đầu ra |
| --- | --- |
| Bootstrap lấy lại train, val, test; s* Platt/isotonic; s* trong vùng hỗ trợ; biến cố đồng thời | `vr24_full_bootstrap.py` |
| Độ nhạy s* theo hiệu chỉnh và nuisance; hai sàn | `vr25_psi_sensitivity.py` |
| Liều Poisson mọi ô, đường học 320/1.000/3.000, kiểm Gumbel | `vr26_dose_poisson.py` |
| Hồi quy từng tổn thương hiệu chỉnh sai số trong biến (biến công cụ từ hai nửa) | `vr27_bridge_eiv.py` |
| Mô phỏng căng thẳng Proposition 2 (sàn phóng đại, kịch bản đối nghịch, MLP + Platt) | `vr28_psi_stress.py` |
| Sai số Monte Carlo của đầu mút Bonferroni, phân phối s*, biến cố đồng thời | `vr29_bootstrap_audit.py` |

Kết quả chốt (trừ vr19/vr29 đang chạy tới B = 5.000): nhãn robust không đổi khi lấy lại val. s* color điểm 0,38 (bảng), 0,49 (ảnh); cận một phía 95% khi lấy lại val: 0,82 cả hai. s* theo cách ước lượng q: 0,28–0,39 (bảng), 0,48–0,58 (ảnh) qua các cách hiệu chỉnh; gradient boosting nâng lên 0,86 / 0,76; M0 ảnh không hiệu chỉnh cho 0,17. Trong vùng hỗ trợ s* tăng (bảng 0,72 rồi không đạt; ảnh 0,71 / 0,95 / 0,86). Liều Poisson: tỉ số quan sát/dự đoán 1,18–1,30 (bảng), 0,89–1,01 (ảnh); đường học bảng 1,30 → 1,15 → 1,04. Hồi quy từng tổn thương sau hiệu chỉnh: 1,33 [1,01; 1,69] (bảng), 1,32 [0,92; 1,75] (ảnh). Mô phỏng căng thẳng: sàn hợp lệ không bao giờ chứng nhận sai dấu; sàn phóng đại trong kịch bản đối nghịch chứng nhận sai dấu 20% (oracle), tới 14% (ước lượng).

Claim không được phép thêm: s* là đại lượng được định danh từ cohort; phân vị 95 của s* là cận tin cậy (chỉ là ngưỡng ổn định bootstrap; cận một phía báo riêng); "sàn phóng đại vẫn an toàn"; "độ dốc từng tổn thương ≈ 1 chứng minh identity"; "tỉ lệ ác tính được xác minh ≥ s_min" thay cho sàn theo từng điểm; "cơ chế lâm sàng" từ phân tích trong/giữa bệnh nhân.

**Chốt B = 5.000 (2026-09-26).** Nhãn robust không đổi: Bonferroni cho color và size với đặc trưng ảnh; color với đặc trưng bảng chỉ ở mức 95% (Δ_M0 Bonferroni [−0,15; 2,08]). Sai số Monte Carlo của đầu mút tối đa 0,06; mỗi nhãn được lặp lại ở ≥ 99% lần lấy lại. Phân vị 95 của s* color: 0,79 (bảng), 0,80 (ảnh).

## 11. Vòng kiểm chứng thứ năm (2026-09-26)

Tiêu đề: "Causal Analysis of Learning Reversal under Selective Verification in Skin Cancer Classification". Không đổi quy tắc robust chính (vr19, B = 5.000). Mọi phân tích dưới đây thêm sau khi đã thấy kết quả chính.

| Phân tích | Script |
| --- | --- |
| Bootstrap lấy lại train, val, test (1.000/họ) với ba q̂ cùng tiêu chí (MLP-Platt, MLP-isotonic, GBM-Platt), chẩn đoán q̂ | `vr30_q_bootstrap.py` |
| Liều Poisson 20 lần lặp, m_k bằng ridge và MLP, đường học 320/1.000/3.000, độ lớn tràn sang | `vr32_dose_nuisance.py` |
| Cầu nối từng tổn thương với ĝ bằng GBM và trên vùng hỗ trợ | `vr33_bridge_alt.py` |
| Bán mô phỏng trên input thật của ISIC | `vr34_semisynth.py` |
| RR biên từ số đếm, bootstrap cụm bệnh nhân | `vr35_marginal_rr.py` |
| Lấy mẫu con một nửa bệnh nhân không hoàn lại | `vr36_subsample.py` |

Kết quả chốt: lấy lại cả val làm color (ảnh) mất nhãn Bonferroni (Δ_M0 [−0,01; 2,29]), chỉ còn 95%; lấy mẫu con giữ mọi nhãn và hẹp hơn. s* điểm của color: 0,38 / 0,28 / 0,75 (bảng; Platt / isotonic / GBM), 0,49 / 0,58 / 0,84 (ảnh); sàn bootstrap Platt 0,82 cả hai; biến cố đồng thời tại 0,8: 940 và 931 trên 1.000. GBM khớp test kém nhất. Liều bảng: tỉ số 1,24 → 1,14 → 1,01 [0,95; 1,06]; ảnh 0,99 với n = 320 nhưng 0,70–0,72 với 1.000–3.000 (hai đặc tả m_k trùng nhau). Bán mô phỏng: không chứng nhận sai dấu tại sàn thật, bao phủ 79–96%, ψ_L plug-in lệch xuống; độ dốc cầu nối khi identity đúng: trung vị 0,83 / 0,97. RR biên (cohort, từ số đếm): 1,97 [1,48; 2,59], sàn 0,51, một phía 0,64.

Claim không được phép thêm: một ngưỡng 0,82 chung cho đảo dấu so với bệnh; "dữ liệu xác lập đảo dấu so với bệnh"; độ lớn liều của learner ảnh; độ dốc IV là độ dốc thật; ψ của hai họ là cùng một estimand; RR biên kiểm chứng ψ.

## 12. Vòng kiểm chứng thứ sáu (2026-09-26)

Phân tích thêm sau khi đã thấy kết quả chính: `vr37_overlap_sigma.py` (overlap theo ba ước lượng σ), `vr38_logit_tail.py` (cơ chế đuôi logit), `vr34_semisynth.py gbm` (DGP dựa trên GBM, cầu nối với ĝ GBM), `vr35_marginal_rr.py` (biến cố đồng thời ở mức liên hệ biên).

Sửa định nghĩa: phân vị 95 của s* coi lần lặp không có s* là "trên 1" (phân vị không nội suy), nên bằng đúng sàn bootstrap (ψ_L > 0 ở ≥ 95% lần lặp). Áp dụng cho vr19, vr29, vr30.

Kết quả chốt: overlap phụ thuộc σ̂ (13–31% tổn thương đổi nhóm; M2 luôn âm; color bền ở 1–3/3 ngưỡng tùy σ̂). Đảo dấu ở mức liên hệ biên (cohort): 1.913/2.000 lần lặp tại sàn 0,65; test riêng: sàn 0,95. Cơ chế: Platt và isotonic cùng kỳ vọng ~25–27 sự kiện ở tertile dưới nhưng trung bình logit q lệch 1,48; GBM sai ngay ở mức tertile (34,4/36,7 so với 25/47). Bán mô phỏng chéo DGP: không chứng nhận sai dấu; tỉ lệ chứa 62–96%, GBM thấp nhất; độ dốc cầu nối với ĝ GBM 1,35 (bảng), 0,96–0,98 (ảnh).

Claim không được phép thêm: "đảo dấu chủ yếu dựa vào tổn thương ít được sinh thiết" (phụ thuộc σ̂); ô robust mới từ lấy mẫu con; độ lớn cầu nối qua mọi lớp ĝ; "coverage" cho bán mô phỏng; "its cause, the training selection".

## 13. Vòng kiểm chứng thứ bảy (2026-09-26)

Phân tích thêm sau khi đã thấy kết quả chính: `vr39_seed_variance.py` (200 lần lặp đầu của vr19 mỗi họ, thêm hai seed trên cùng mẫu bệnh nhân), `vr34_semisynth.py local` (sàn bị vi phạm ở 30% tổn thương có xác suất xác minh tổng thể thấp nhất, s = 0,20), `vr35_marginal_rr.py` mở rộng (A lớn nhất đạt 95% và B_V cấp cohort; các số cũ không đổi).

Định nghĩa: mục tiêu của bootstrap là tương phản của thuật toán huấn luyện lấy trung bình theo tính ngẫu nhiên (seed b ở lần lặp b); dấu dương của ψ_k chỉ cần sàn ở tertile dưới; ở mức liên hệ biên, RR_D > 1 ⇔ A < RR_Y, và sàn π₀¹ ≥ s ở tertile dưới đủ để A ≤ 1/s. Thuật ngữ: learner reversal, learner-scale disease-relative reversal, marginal association reversal.

Kết quả chốt: seed chiếm 17–64% phương sai bootstrap một seed; khoảng trung bình ba seed rộng 74–98% khoảng một seed (khoảng một seed thận trọng). Sàn vi phạm cục bộ: không chứng nhận sai dấu; tỉ lệ chứa oracle 50–75%, plug-in 62–92%. Liên hệ biên (cohort): sàn tertile dưới 0,65 ⇔ A ≤ 1,55; B_V = 4,96 [3,94; 6,25]. σ theo input của learner: color mất đảo dấu ở ngưỡng 5 và 10 (M0 âm, −0,82 đến −0,29); size ảnh bền mọi ngưỡng, mọi σ̂. Size mất đảo dấu dưới định nghĩa slope (ảnh −0,02; trong bệnh nhân +0,01).

Claim không được phép thêm: 0,65 là ngưỡng lâm sàng đã xác nhận; kết quả biên là xác nhận độc lập; một khoảng 0,28–0,84 chung cho hai họ; "weakly estimable" như kết quả hình thức; clinical verification là nguyên nhân đã được can thiệp; size bền theo mọi định nghĩa concept.

## 14. Vòng kiểm chứng thứ tám (2026-09-26)

Phân tích thêm sau khi đã thấy kết quả chính: `vr39_seed_variance.py` mở rộng tới 1.000 lần lặp mỗi họ với ba seed (bootstrap trực tiếp trung bình ba seed); `vr40_support_composition.py` (trọng số hóa vùng hỗ trợ về phân phối năm concept của toàn quần thể trong từng tertile; tương phản cân bằng theo concept khác như chẩn đoán phụ); `vr41_marginal_site.py` (liên hệ biên theo cơ sở, chuẩn hóa theo cơ sở, bỏ từng cơ sở; mốc gradient logit và Fig. S4).

Định nghĩa: đối tượng suy luận chính là quy trình huấn luyện một seed ngẫu nhiên; ước lượng điểm ba seed nhắm cùng kỳ vọng. "Bonferroni-robust" chỉ trong họ sáu so sánh lập sau khi xem kết quả, định nghĩa tertile, val cố định. σ̂ khớp lý thuyết là perceptron trên input của learner; σ̂ GBM bảng là định nghĩa lâm sàng chung. GBM cho q là đặc tả kiểm tra sức chịu, không phải ngưỡng có hỗ trợ thực nghiệm. "Bootstrap stability threshold" thay "bootstrap floor".

Kết quả chốt: nhãn ba seed trùng nhãn một seed và nhãn chính ở cả sáu ô. Trọng số hóa thành phần không khôi phục Δ_M0 của color trong vùng hỗ trợ ở cả 9 thiết lập (bảng −0,44 đến +0,78; ảnh −0,82 đến +1,00; toàn quần thể +1,02 và +1,64); M2 sau trọng số −1,25 đến −0,65. Tương phản cân bằng theo concept khác: M2 color −0,42 (bảng), −0,07 (ảnh). Liên hệ biên chuẩn hóa theo cơ sở: RR 2,05 [1,56; 2,75], sàn 0,62; bỏ từng cơ sở 0,40–0,76; hai cơ sở nhiều ca ác tính nhất RR 1,26 và 1,31 (sàn điểm 0,79 và 0,76). Mốc logit: π₀¹ ≥ 0,56 (color), 0,48 (size); mốc tỉ số bất khả khi π₀¹ > 0,20.

Claim không được phép thêm: M2 ngoại suy là nguyên nhân mất đảo dấu trong vùng hỗ trợ; khoảng một seed "thận trọng"; Bonferroni là suy luận xác nhận; B_V "as strongly" là cùng mức phụ thuộc; khoảng sàn qua các q̂ là khoảng bất định; tương phản concept là hiệu ứng khi giữ concept khác cố định.
Rà soát sau vòng 8: σ̂ GBM bảng gọi là "common" (không còn "primary"); ký hiệu điểm tới hạn viết s∗; Δ_q là tương phản của M0 đã hiệu chỉnh trong xấp xỉ s∗ ≈ exp(−Δ_q).

## 15. Vòng kiểm chứng thứ chín (2026-09-26)

Sửa lý thuyết: Proposition 2 phát biểu dạng tổng quát ψ_k = E[a_k(X̃) logit p(X̃)], a_k = P(t=1|x̃)/P(t=1) − P(t=0|x̃)/P(t=0); tập sharp chọn đầu mút theo dấu a_k. Công thức theo tertile quan sát là sharp khi t là hàm của x̃ (họ bảng: năm concept nằm trong input) và là cận ngoài hợp lệ cho họ ảnh. Sàn cho dấu dương chỉ cần nơi a_k < 0 (với công thức tertile: trên input xuất hiện ở tertile dưới). Phân tích thêm: `vr42_sharp_weighted.py` (mô phỏng lý thuyết; cận sharp plug-in với â_k ước lượng chéo trên test).

Kết quả chốt: bảng: hai công thức trùng nhau (s∗ 0,38/0,54/0,61; dấu â khớp 100%). Ảnh: cận sharp plug-in hạ s∗ color 0,49 → 0,29, size 0,50 → 0,24, contrast 0,64 → 0,55, nhưng dấu â khớp tertile quan sát chỉ 74–87%; dấu sai làm tập quá hẹp nên báo cận ngoài. Trọng số site trong vr41 là một phân phối chung cho hai tertile (đã kiểm tra code). Tương phản cân bằng: chỉ color bảng giữ dấu tách biệt.

Nhãn: "Bonferroni sign-separated" thay "Bonferroni-robust". Estimand Δ lấy kỳ vọng theo tính ngẫu nhiên của huấn luyện; là tương phản biên tertile.

Claim không được phép thêm: tập sharp cho họ ảnh theo công thức tertile; sàn "chỉ trên tertile dưới" cho họ ảnh ngoài nghĩa input xuất hiện ở tertile dưới; tương phản concept là hiệu ứng riêng của concept; kết quả biên áp dụng cho site mới.

## 16. Vòng kiểm chứng thứ mười (2026-09-26)

Không chạy thêm thí nghiệm. Sửa lập luận: tỉ lệ khớp giữa dấu â_k và tertile quan sát không đo độ chính xác dấu của a_k (khi t không là hàm của x̃, dấu a_k không nhất thiết trùng tertile của từng tổn thương). Lý do giữ cận ngoài cho họ ảnh: tập sharp cần dấu thật của a_k, thay bằng ước lượng thì chưa có tính hợp lệ mẫu hữu hạn. Kết quả trọng số ước lượng (s∗ 0,29 cho color ảnh) là thăm dò. Thuật ngữ: "sharp population identified set" (a_k thật), "stratum-based outer set", "estimated-weight approximation to the sharp set". Tương phản cân bằng: khoảng có điều kiện trên learner, không thuộc phân tích bội chính. Sàn 0,62 chuẩn hóa theo site: cho hỗn hợp site quan sát dưới sàn chung giữ trong từng site. GBM cho q: đặc tả độ nhạy, không phải lý do loại bỏ.

Claim không được phép thêm: 79% là độ chính xác dấu; tập trọng số ước lượng là sharp; "Selection associated with clinical verification" như kết luận nhân quả về quyết định sinh thiết.

## 17. Vòng kiểm chứng thứ mười một (2026-09-26)

Suy luận chính của Table 1 đổi sang bootstrap của đúng thống kê ước lượng điểm: trung bình ba seed, 1.000 lần lặp mỗi họ (`vr43_three_seed_primary.py`, dùng lại các lần huấn luyện của vr19 và vr39, không huấn luyện thêm). Phân tích một seed 5.000 lần là kiểm tra độ phân giải cao cho quy trình một seed ngẫu nhiên. Nhãn trùng nhau ở cả mười ô. Sai số Monte Carlo của đầu mút Bonferroni tối đa 0,11; nhãn color bảng (đầu mút dưới Δ_M0 −0,01) chỉ tái lập ở 56% lần lấy lại Monte Carlo. Kết quả trọng số ước lượng (0,29) chỉ ở phụ lục. Cách diễn đạt: "does not identify or validate transport to a new site".

## 18. Vòng kiểm chứng thứ mười hai (2026-09-27)

Bootstrap ba seed chính (Table 1) mở rộng từ 1.000 lên 5.000 lần lặp mỗi họ: `vr39_seed_variance.py` huấn luyện hai seed thêm cho b = 1000..4999 trên đúng mẫu bệnh nhân của vr19 (driver `run_round12.sh`, `run_round12b.sh`, `run_round12c.sh`), rồi `vr43_three_seed_primary.py` tính lại; kết quả 1.000 lần giữ ở `vr43_three_seed_primary_B1000.json`. `vr39_seed_variance.json` (Table S14) nay tính trên 5.000 lần lặp.

Quy tắc Monte Carlo (đặt trong vòng này, sau khi đã xem kết quả 1.000 lần): nhãn gọi là "resolved" khi được tái lập ở ≥ 95% trong 2.000 lần lấy lại các lần lặp đã lưu; ngược lại báo "unresolved", không báo như một kết luận chắc chắn.

Kết quả chốt: 2/6 ô Bonferroni sign-separated và resolved (color ảnh, size ảnh; tái lập 100%). Color bảng: đầu mút dưới Bonferroni của Δ_M0 là +0,01, tách dấu trên toàn bộ 5.000 lần nhưng chỉ tái lập 72% → unresolved. Ba ô còn lại "no", resolved. Sai số MC tối đa của đầu mút 0,07. Một seed 5.000 lần: color bảng [−0,15; 2,08] (no); các ô khác trùng nhãn.

Thuật ngữ: population tipping floor (Proposition 2), plug-in point tipping floor, bootstrap stability threshold; Table S39 và mục S7 gọi là "plug-in bounds". Câu overlap: "was not restored after standardizing the supported lesions on the five measured concepts under this weighting model".

Claim không được phép thêm: "3 of 6 cells" hoặc color bảng là Bonferroni sign-separated; ngưỡng 0,65 như sàn chính trong Conclusion; bootstrap stability threshold là sàn được định danh.

## 19. Vòng kiểm chứng thứ mười ba (2026-09-27)

Không huấn luyện thêm. `vr44_dose_estimand.py` định nghĩa và ước lượng estimand can thiệp τ_j(η₁, η₀) = E_ξ[Δ_j(ξ; η₁) − Δ_j(ξ; η₀)] từ các lần lặp đã lưu của vr32 (ghép cặp: ác tính train/val và seed cố định qua η; nhãn âm train và val rút theo η). `vr45_support_retention.py` lưu tỉ lệ tổn thương test giữ lại ở ngưỡng hỗ trợ để bài dựng lại được từ repo mà không cần `vl_nuisance.npz`.

Kết quả chốt: τ̂(1, −1) của concept được chọn: bảng −5,29 đến −3,53, ảnh −2,65 đến −1,22; mọi khoảng 95% < −1,15; đổi dấu trong cùng lần lặp 118/120. Khoảng chỉ phản ánh tính ngẫu nhiên của huấn luyện trên cohort cố định.

Trình bày: Proposition 2 viết trực tiếp ψ_L = E[a_k·L], ψ_U = E[a_k·H] (L chọn logit q khi a_k ≥ 0); công thức theo tertile là Corollary. "Sharp" chỉ tương đối với mô hình phi tham số của dữ liệu quan sát với Y = D·S và sàn theo điểm. Phân tích biên là mục IV-G riêng, không phải ước lượng hay kiểm tra ψ. Tên ngưỡng biên: "95 percent resampling stability threshold". Chuẩn hóa theo site là estimand hỗn hợp. Ghi rõ trong III-G: thống kê bootstrap chính đổi từ một seed sang trung bình ba seed sau khi xem kết quả.

Claim không được phép thêm: τ̂ có khoảng theo bệnh nhân; learner reversal là hiệu ứng của quy tắc chọn mẫu; kết quả biên xác nhận Proposition 2; đảo dấu biên đúng ở từng site.

## 20. Vòng kiểm chứng thứ mười bốn (2026-09-27)

Thí nghiệm liều chạy lại (`vr46_dose_fixed_val.py`, driver `run_round14.sh`) để can thiệp CHỈ lên tập huấn luyện: validation (ác tính và nhãn âm rút đều) cố định trong mỗi lần lặp, dừng sớm và Platt trên cùng tập đó; số ngẫu nhiên chung U_i cho việc chọn nhãn âm qua các mức η; tập ác tính train và seed dùng chung. Estimand: τ_j(η₁, η₀) = E_ω[Δ_j(η₁, ω) − Δ_j(η₀, ω)], có điều kiện trên cohort, split, validation và quần thể test.

Kết cục chính là tương phản logit thô (đối tượng của Proposition 1). Platt trên validation cố định nhắm phân phối chưa chọn; ở η = 1 hệ số Platt âm ở 29/200 lần khớp (không có ở mức khác), nên tương phản hiệu chỉnh chỉ báo ở phụ lục. Quyết định này đặt sau khi xem kết quả.

Kết quả chốt (n = 320): đường trung bình qua 0 ở 6/6; đổi dấu trong cùng lần lặp 120/120; lan sang đúng hướng 29/30; tỉ số quan sát/dự đoán Bayes-tối-ưu: bảng 0,99–1,14, ảnh 0,88–1,10; τ̂(1, −1) bảng −4,72 đến −2,59, ảnh −3,04 đến −1,11, mọi khoảng 95% < −0,99. Đường học color bảng: 0,99 → 0,93 → 0,86 (320 → 1.000 → 3.000), nên KHÔNG còn nói độ lớn tiến về dự đoán khi mẫu tăng. Thiết kế liều chung train+validation (vr32, vr44, vr21, vr26) giữ ở phụ lục như hiệu ứng của quy tắc chọn dữ liệu phát triển chung.

Claim không được phép thêm: τ là hiệu ứng siêu quần thể; khoảng của τ̂ phản ánh lấy mẫu bệnh nhân; độ lớn khớp chính xác với learner hữu hạn; tương phản hiệu chỉnh trên validation cố định là kết cục chính.

## 21. Rà soát trình bày (2026-09-27)

Không chạy thêm phân tích. Năm lượt rà soát: bỏ lặp giữa các mục (đoạn "Why two concepts", MC SE lặp giữa chữ và chú thích Table 1, số 0,84 lặp, hai câu cùng ý ở IV-D, hai dòng Table 2 trùng IV-C, định nghĩa ngưỡng ổn định lặp trong chú thích Table 3); IV-D viết lại gọn; IV-F đổi tên "Learner-scale disease target"; Discussion và Conclusion rút gọn, tách rõ hai estimand về bệnh. Sửa: Δ_j(η, ω) là tương phản của một lần khớp; định nghĩa f₀, f₂; "Platt applied to M0 and M2"; nhãn "exact" trong bảng thiết kế liều chung đổi thành "Bayes-optimal shift". Các số không đổi.

## 22. Vòng kiểm chứng thứ mười lăm (2026-09-27)

Không chạy thêm thí nghiệm. Kết quả nhân quả chính của thí nghiệm liều là τ̂(1, −1); "qua 0" chỉ khẳng định ở cỡ 320 nhãn âm (gần cỡ huấn luyện của M2). Với 3.000 nhãn âm, color ảnh có τ̂ = −1,79 [−2,05; −1,51] nhưng đường trung bình không qua 0 (đổi dấu 8/20). Tỉ số quan sát/dự đoán trên logit thô chỉ là mô tả (logit mạng hữu hạn không nằm trên thang log-odds quần thể). τ được định danh bởi thiết kế; Proposition 1 chỉ dự đoán hướng của dịch chuyển Bayes-tối-ưu. Validation cố định chỉ dùng cho dừng sớm; hiệu chỉnh không thuộc kết cục chính. "Post-inspection" ghi ngay ở lần đầu trình bày thí nghiệm (Abstract, Contribution 2).

Claim không được phép thêm: tỉ số quan sát/dự đoán là ước lượng tỉ lệ hiệu ứng nhân quả; "qua 0" ở mọi cỡ mẫu; Proposition 1 là lý do τ được định danh.

## 23. Vòng kiểm chứng thứ mười sáu (2026-09-27)

Không chạy thêm thí nghiệm; chỉ sửa cách diễn đạt. Logit thô của mạng hữu hạn "need not equal the Bayes-optimal log-odds of the selected training distribution" (không nói "không nằm trên thang log-odds"); tỉ số quan sát/dự đoán "does not test the magnitude predicted by Proposition 1". Câu kết thu hẹp về "learned concept contrast". IV-D: "coincided with", không "reflects"; hạn chế vùng hỗ trợ không định danh nguyên nhân. Tag v1.0-submission (d20501c) đã chứa vr46; sẽ dời tag khi đẩy bản mới.

## 24. Vòng kiểm chứng thứ mười bảy (2026-09-27)

Không chạy thêm thí nghiệm. Câu đầu Abstract giữ phân biệt D, S, Y = D·S: bệnh chỉ được xác nhận ở tổn thương sinh thiết, tổn thương chưa xác minh vào huấn luyện như nhãn âm ghi nhận; III-A: "disease is missing not at random among unverified lesions" (nhãn Y không thiếu). Tiêu đề IV-B: "Controlled selection of recorded negatives can induce the reversal". Bảng tái lập S8 đối chiếu với tệp thật trong repo; các .npz và checkpoint ghi rõ "not distributed".

## 25. Vòng kiểm chứng thứ mười tám (2026-09-27)

Không chạy thêm thí nghiệm. III-C: "For the Bayes-optimal learners, a reversal occurs exactly when…". Mọi chú thích hình và bảng (bài chính và phụ lục) chỉ còn một câu; phần mô tả chuyển vào bài, ngay trước bảng/hình ở phụ lục. Hình phụ lục đánh số lại theo thứ tự xuất hiện (phase diagram S1, fine-tune S2, logit benchmark S3, constrained learner S4) và đổi tên tệp tương ứng. Fig. 1(b) và Fig. 3(b) nay được nhắc trong phần chữ. Thêm trích dẫn cho Platt, isotonic, gradient boosting, percentile bootstrap (bài chính) và danh mục tài liệu phụ lục (Kish, DerSimonian-Laird, Kool, Kull, Meurer, Politis). Release v1.1-submission đã kiểm tra đầy đủ; bản này trích v1.2-submission.

## 26. Vòng kiểm chứng thứ mười chín (2026-09-27)

Không chạy thêm thí nghiệm. Nguồn định danh của τ: nhà phân tích trực tiếp thực thi từng quy tắc chọn mẫu trên cùng ω (không nói "assignment of η"); sửa đồng bộ ở III-B, Introduction, Conclusion. Mười chú thích bảng phụ lục dài hơn 14 từ được rút gọn, bổ sung vào câu chú giải trước bảng. Release v1.2-submission đã kiểm tra từ GitHub (đủ vr46, analysis_lock, vl_split; bảng S8 không thiếu tệp; dựng lại không đổi). Bài trích v1.3-submission.

## 27. Vòng kiểm chứng thứ hai mươi (2026-09-27)

Không chạy thêm thí nghiệm. Trong vr46, tập validation được rút lại ở mỗi lần lặp và dùng chung cho mọi mức η: ω nay gồm cả "validation-set draw used for early stopping"; τ và các khoảng của nó có điều kiện trên cohort, patient split và quần thể test (không còn "conditional on the validation set"); η không đổi validation, nên vẫn là training-selection effect. "Fixed validation set" trong thí nghiệm liều đổi thành "replicate-specific validation set shared across doses". Kết luận Table 1 vẫn có điều kiện trên tập validation thực tế (đúng). Theo quyết định của tác giả: tên hình/bảng giữ một câu; lưu ý suy diễn đặt thành đoạn "In Table x, …"/"In Fig. x, …" ngay trước hình/bảng (Table 1–3, Fig. 2–4; phụ lục S22, S43, S44, S50 kèm B_V là mốc tham chiếu). Bài trích v1.4-submission.

## 28. Vòng kiểm chứng thứ hai mươi mốt (2026-09-27)

Không chạy thêm thí nghiệm. Hai dòng thí nghiệm liều của Table 2: "Selection dose on training only; replicate-specific validation shared across doses" (không còn "validation fixed"). Cụm "fixed validation set" chỉ còn ở suy luận chính (Table 1), nơi đúng nghĩa. Lưu ý suy diễn giữ ở đoạn "In Table x, …" cạnh bảng theo quy tắc của tác giả. Bài trích v1.5-submission.
