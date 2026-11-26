# Causal Analysis of Learning Reversal under Selective Verification in Skin Cancer Classification

**Quang Toan Dao**¹·² (dqtoan@ioit.ac.vn), **Viet Anh Nguyen**¹ (anhnv@ioit.ac.vn)

¹ Institute of Information Technology, Vietnam Academy of Science and Technology, Ha Noi, Viet Nam

² Graduate University of Science and Technology, Vietnam Academy of Science and Technology, Ha Noi, Viet Nam

## Abstract

In skin cancer cohorts, disease is confirmed only for biopsied lesions, and unverified lesions enter training as recorded negatives. We ask whether selecting training data on verification can reverse the concept contrast a classifier learns, and what the data identify about disease. When training lesions are selected on the recorded label and the learner's input, the Bayes-optimal logit of the selected data equals the full-cohort logit plus the log ratio of the class-specific selection probabilities. Under biopsy-only selection, this offset is minus the observable log verification propensity among recorded negatives. On 401,059 ISIC-2024 lesions, learners trained on all and on biopsied lesions gave opposite-sign estimates of the tertile contrast for color variegation and size. In a post-inspection family of six comparisons, Bonferroni-adjusted intervals separated the signs for size with image features under every resampling scheme. They did so for color variegation with image features when the validation set was fixed. The color and size reversals persisted when training-set size was matched. In a post-inspection experiment, a designed concept-dependent rule changed only which recorded negatives entered training. Each rule moved its contrast in the direction the offset predicts, and with 320 expected recorded negatives the mean contrast crossed zero. For latent disease, a floor on malignant verification yields an identified set for a disease contrast on the learner's scale. The set is sharp with known nuisance functions; for image features we report a conservative outer set. On ISIC-2024, the floor at which the plug-in lower endpoint became positive depended on the estimator of recorded-label risk. A separate whole-cohort sensitivity analysis implied a disease association opposite to the verified-only one, but only under an assumed floor on malignant verification in the lower tertile. Changing which recorded negatives enter training can change the concept contrast a classifier learns.

**Keywords:** causal inference, selection bias, selective labels, case-control sampling, partial identification, verification bias, skin lesion classification

## I. Introduction

Most labels in medical imaging are verified selectively. A lesion receives a histopathologic diagnosis only if a clinician decided to biopsy it. In ISIC-2024, lesions never linked to a pathology report and not tagged as biopsied were recorded as benign [@kurtansky]. Only 1,068 of 401,059 lesions carry a tissue diagnosis. Image classifiers can match dermatologists on biopsy-verified test images [@esteva]. In cohorts such as ISIC-2024, however, training labels reflect disease and clinical work-up together.

Selection is usually treated as a problem of estimation or evaluation. Verification bias corrupts sensitivity and specificity [@begg]. Conditioning on selection can open noncausal paths, and whether they can be blocked depends on the graph and on the measured variables [@hernan]. Selective labels complicate algorithm evaluation [@lakkaraju]. Less studied is what verification does to the relationship a model learns between a visual feature and disease. A feature can accompany disease and also prompt the decision to biopsy. Training on verified lesions then conditions on that decision [@berkson].

We ask two questions. First, does the selection that produced a training set change the sign of the concept contrast a classifier learns? Second, when the learned sign differs between training regimes, which learner is reversed relative to latent disease, and under which assumption can that be identified? We do not estimate the effect of intervening on a concept.

We make three contributions.

**First, an observable offset for what selected training data teach a learner.** When training data are selected on the recorded label and the input, the Bayes-optimal logit of the selected data equals the full-cohort logit plus the log ratio of the class-specific selection probabilities. Under biopsy-only selection, the offset is minus the log verification propensity among recorded negatives. The identity belongs to the family of case-control and local case-control offsets [@prentice; @fithian]. We use it as a diagnostic that can be computed before a biopsy-trained learner is fitted. Concept by concept, it predicts the sign of the gap between learners trained on all and on verified lesions. It also predicts the direction in which a controlled change in training selection moves the learned contrast. We test both predictions.

**Second, evidence that selection on verification can reverse the contrast a classifier learns.** On ISIC-2024, learners trained under the two regimes learn opposite signs for color variegation and size on the full patient-disjoint test population. The strongest multiplicity-adjusted evidence for the marginal tertile contrast is for size with image features. Color variegation kept opposite point-estimate signs under all seven concept definitions in all four learner families, as did lesion-skin contrast. Within regions with stronger verified-training support, no color comparison kept separated joint intervals. The color and size reversals survive a size-matched control. After observing this reversal, we designed a selection rule for recorded negatives that depends on one concept. It does not intervene on clinical biopsy decisions, and we do not claim that it matches their strength. It identifies the effect of the implemented rule on the learned contrast. Each concept-specific rule shifted the corresponding contrast in the direction the offset predicts. With 320 expected recorded negatives, close to the training size of the biopsy-only learner, the mean contrast also crossed zero.

**Third, what the verification process leaves identified about disease.** We define the disease target on the same functional and population as the learned contrast and derive its population identified set under a floor on malignant verification. Turning pointwise risk bounds into a sharp set for this contrast needs the sign of a weight that changes between strata. With known nuisance functions the set is sharp. A stratum-based outer set, used for image features, needs no estimate of that weight. For the marginal association the condition reduces to a bound on one verification ratio. We analyzed ISIC-2024 after the primary results had been seen. There the framework does not establish disease-relative sign claims. Instead, it shows why they stay sensitive to the assumed floor and to the estimator of recorded-label risk. PAD-UFES-20, where every recorded malignancy is biopsied [@pacheco], serves as an external consistency check on the attenuating side of the boundary.

**How the parts fit.** The three parts rest on different kinds of evidence, and keeping them apart, more than any single identity, is the causal contribution. The effect of a training-selection rule is identified by design. The gap between the training regimes has an observable diagnostic. Latent disease is not point identified, so statements about it are sets indexed by an explicit floor on malignant verification.

## II. Related Work

Three literatures meet in this problem. Work on selection bias and case-control sampling describes how selection on an outcome shifts an association or a fitted logit. Work on selective labels and positive-unlabeled learning describes what can be learned or bounded when outcomes are observed only after a decision. We draw on both and keep the sampling mechanism separate from statements about disease.

**Selection bias and its direction.** Epidemiology expresses selection bias as a multiplicative factor on the odds ratio, built from the selection probabilities of the exposure-by-outcome table [@kleinbaum]. Quantitative bias analysis turns that factor into sensitivity parameters [@greenland; @lash]. Berkson described selection on a common effect [@berkson]. The direction of collider bias between binary variables has been characterized in closed form [@nguyen]. Reversal by selection has been analyzed for the obesity paradox [@banack], and collider bias affects modern cohorts [@griffith]. Our Lemma 1 restates these results with verification ratios, and we do not claim it as new.

**Case-control and subsampling offsets.** Case-control sampling that depends on the outcome alone leaves logistic slopes unchanged and moves the intercept [@prentice]. Local case-control sampling selects on the outcome and the covariates and corrects the fit by a known offset [@fithian]. Our Proposition 1 is an identity of this family, written for training data selected by clinical verification. The algebra is the same, but the use differs. The offset is set by clinical verification rather than chosen by the analyst, and we read its stratum contrast as a prediction of how two learned contrasts differ.

**Selective labels.** When outcomes are observed only after a human decision, evaluation and learning depend on that decision [@lakkaraju; @kleinberg]. Expert consistency [@dearteaga] and selective testing [@mullainathan] have been used to learn under such selection. Risk prediction under historical testing can be improved with domain constraints such as known prevalence [@balachandar]. Rambachan, Coston and Kennedy bound measures of predictive performance under sensitivity assumptions on unobserved confounding, with debiased nuisance estimation [@rambachan]. We do not propose a new bounding framework. Our target is a learner-matched logit contrast of latent disease whose weights change sign between strata. Proposition 2 adds the step from pointwise bounds on disease risk to a sharp set for such a contrast, and connects that set to an observable learner reversal.

**Positive-unlabeled learning.** Writing *Y* = *D*·*S* makes every unverified malignancy an unlabeled positive, the structure of positive-unlabeled learning [@elkan; @bekker], including feature-dependent labeling propensities [@bekkersar]. These methods use a labeling propensity to learn a classifier for the latent label. We do not learn the disease risk. We bound a contrast of it under a floor on the labeling propensity of malignant lesions.

**Verification bias, missing data and shortcuts.** Verification bias in diagnostic accuracy [@begg] can be corrected under ignorable verification [@alonzo]. Graphical criteria address recovery from selection [@bareinboim] and missing data [@mohan], and learning under feature-dependent [@zadrozny] and outcome-dependent selection [@heckman] is classical. Dermatology classifiers exploit skin markings [@winkler] and hidden strata [@oakden], shortcut learning is general [@geirhos], and acquisition shapes what a model can learn [@castro].

**Partial identification and probing.** When a parameter is not point identified, one can report the set of values compatible with the data and stated assumptions [@manski; @tamer], and learn robustly over such sets [@kallus]. Linear probes [@alain] and centered kernel alignment [@kornblith] serve only in supplementary localization diagnostics.

Together, these strands explain how selection can shift an association and how a latent target can be bounded. They leave open how one verification process acts on three objects at once. These are the contrast a learner absorbs from its training data, the effect of changing training selection, and the matching contrast of latent disease. Section III defines the three objects separately and states what identifies each.

## III. Method

Section III-A gives the causal model. Section III-B defines three objects: the learned contrast, the effect of a training-selection rule and the disease target. Sections III-C to III-E state what identifies or bounds each. Proposition 1 gives the offset of selected training data, Lemma 1 the classical selection factor for the marginal disease association, and Proposition 2 the identified set for the disease target. Sections III-F and III-G describe the data, the learners and the inference.

### A. Causal model

Let *D* be the latent disease state and *X* the lesion's appearance. Concepts are deterministic readings of the image, *C* = *h*(*X*). Diagnosis is anticausal: disease shapes appearance, *D* → *X* → *C*. Let *U* be patient-level risk and surveillance intensity, *G* the acquisition site and *W* clinical information outside the image, such as reported change. A clinician flags lesions of interest (*F*) and biopsies some of them (*S*, with *S* ≤ *F* as indicators). The recorded label is *Y* = *D*·*S*. Fig. 1(a) shows the graph, with latent nodes dashed. The arrow from *G* to *D* stands for differences in case mix between acquisition sites. The graph lets verification depend on disease through *W* and through appearance. Disease can then be missing not at random among unverified lesions, whose recorded label is nonetheless zero. Lesions with *Y* = 0 are recorded negatives. Only those with *S* = 1 are verified benign.

**Fig. 1.** Causal model of selective verification (a) and the regions of Lemma 1 (b).

A learner does not see *X* itself but an input *X̃*: tabular measurements or image features. A training regime is a selection *R* of lesions into the training sample. M0 trains on all lesions (*R* ≡ 1). M2 trains on verified lesions (*R* = *S*). The identities below use only the variables they name, so they hold for any graph consistent with *Y* = *D*·*S*. The graph describes how the data arose and is not the basis of identification.

### B. Targets

Split concept *k* at its cohort tertiles into lower (*t* = 0) and upper (*t* = 1) strata. For a learner *m*, the **learned contrast** Δ_{m,k} is the mean raw output logit, before any calibration, among upper-stratum lesions minus that among lower-stratum lesions. It is computed on a fixed evaluation population, with each lesion weighted equally, averaged over the randomness of training. It is a marginal tertile contrast: it includes differences in correlated concepts between the strata and is not the effect of concept *k* with the others held fixed. We drop *k* when the concept is clear. Write *p*(*x̃*) = P(*D* = 1 | *x̃*). The **disease target** on the same scale and population is

ψ_k = E[logit *p*(*X̃*) | *t* = 1] − E[logit *p*(*X̃*) | *t* = 0].

Because *p* depends on the input, ψ_k is specific to a learner family. The tabular-input and image-feature disease targets are different functionals of the same latent disease.

A **learner reversal** is a difference in sign between Δ_{M0,k} and Δ_{M2,k}. It is observable. But M0 and M2 differ in more than one respect, so a learner reversal is not by itself the effect of a selection rule. We define that effect next. A **learner-scale disease-relative reversal** is a difference in sign between Δ_{M2,k} and ψ_k. It can be established only as far as the identified set for ψ_k determines the sign of ψ_k. We also use the marginal disease contrast θ_k = log OR_D between the strata, the classical object of selection bias. A **marginal association reversal** is a positive θ_k with a negative verified-only association log OR_{D|S} (Lemma 1), or the reverse.

**Training-selection effect.** Let η index a rule, set by the analyst, for selecting recorded negatives into the training set only. The validation set used for early stopping is not changed by η, and calibration is not part of this outcome. Let ω collect the exogenous random numbers of the pipeline: the uniform variables that decide inclusion, the draw of malignant training lesions, the validation-set draw used for early stopping and the optimizer seed. The contrast of a single fit is then a potential outcome Δ_j(η, ω), and the training-selection effect on concept *j* is τ_j(η₁, η₀) = E_ω[Δ_j(η₁, ω) − Δ_j(η₀, ω)]. The analyst directly implements each selection rule while holding the paired algorithmic randomness ω fixed within a replicate. So τ_j is identified by design, conditional on the realized cohort, patient split and evaluation population, with no assumption about clinical verification. It is not the effect of a concept on disease, nor of a clinical biopsy policy, nor a superpopulation effect over repeated patient samples.

### C. The training-selection offset

**Proposition 1 (training-selection offset).** Write *r*_y(*x̃*) = P(*R* = 1 | *Y* = *y*, *x̃*) for the probability that a lesion with recorded label *y* and learner input *x̃* enters training. Then, for every *x̃* with 0 < P(*Y* = 1 | *x̃*) < 1 and *r*₀(*x̃*), *r*₁(*x̃*) > 0,

logit P(*Y* = 1 | *x̃*, *R* = 1) = logit P(*Y* = 1 | *x̃*) + log *r*₁(*x̃*) − log *r*₀(*x̃*).

*Proof sketch.* By Bayes' rule, the odds of *Y* = 1 given *x̃* and *R* = 1 equal P(*Y* = 1 | *x̃*)·*r*₁(*x̃*) divided by P(*Y* = 0 | *x̃*)·*r*₀(*x̃*). Supplementary Section S1 gives the proof. The identity is stated on the learner's own input and uses no disease label. Under biopsy-only selection the offset is −log *g* ≥ 0, so the biopsy-trained logit exceeds the full-cohort logit. For example, *g* = 0.1 at an input raises the Bayes-optimal logit there by log 10.

Suppose *R* is independent of *X̃* given *X* and *Y*, as when selection depends only on appearance and the recorded label. The selection probability then averages over what the input does not show: *r*₀(*x̃*) = E[*r*₀(*X*) | *x̃*, *Y* = 0]. Two cases matter here. *Training on verified lesions.* With *R* = *S*, every malignant training lesion is verified, so *r*₁ = 1. Then *r*₀(*x̃*) = *g*(*x̃*) = P(*S* = 1 | *x̃*, *Y* = 0), the verification propensity among recorded negatives. No further condition is needed. For the Bayes-optimal learners *f*₀ of M0 and *f*₂ of M2, logit *f*₀ − logit *f*₂ = log *g*, and therefore

Δ_{M0,k} − Δ_{M2,k} = E[log *g*(*X̃*) | *t* = 1] − E[log *g*(*X̃*) | *t* = 0].

Write *G*_k for the right-hand side. For the Bayes-optimal learners Δ_{M2,k} = Δ_{M0,k} − *G*_k, so they reverse exactly when Δ_{M0,k}(Δ_{M0,k} − *G*_k) < 0, for either sign of Δ_{M0,k}. The stratum means require logit *q* and log *g* to be integrable within each stratum. *A controlled selection dose.* Let each recorded-negative training lesion enter independently with probability ρ_η = min{1, κ_η exp(η*z*_k(*X*))}. Here *z*_k is the standardized concept, and κ_η fixes the expected number of selected recorded negatives. If recorded positives are drawn uniformly, *r*₀ is known by design. When *z*_k is part of the learner's input, *r*₀(*x̃*) = ρ_η(*z*_k). Without the cap, the contrast of any concept *j* then moves linearly: Δ_j(η) = Δ_j(0) − η·μ_{kj}. Here μ_{kj} is the difference in mean *z*_k between the strata of concept *j*. The Bayes-optimal training-distribution shift is then −(η₁ − η₀)·μ_{kj}. When the input determines *z*_k only partly, as for image features, *r*₀(*x̃*) = E[ρ_η(*z*_k(*X*)) | *x̃*, *Y* = 0]. We approximate it with a Gaussian model for *z*_k given *x̃* (Supplementary Section S1). Adding a constant to a logit cancels in Δ.

These statements are exact for the Bayes-optimal logit of the selected training distribution. A trained network adds approximation, optimization and early-stopping effects. For the effect τ_j of a trained pipeline, they predict direction only; agreement in magnitude is a descriptive diagnostic (Section IV).

### D. Classical identities for the disease association

**Lemma 1 (verification selection factor).** For the strata of concept *k*, let π_t^d = P(*S* = 1 | *D* = *d*, *t*), *A* = π₁¹/π₀¹, *B* = π₁⁰/π₀⁰ and δ = log *B* − log *A*. Then log OR_{D|S} = θ − δ. For θ > 0, verification amplifies the association if δ < 0, attenuates it if 0 < δ < θ, and reverses it if δ > θ. For θ < 0 the regions mirror, and if θ = 0 verification alone creates an association equal to −δ.

Fig. 1(b) shows these regions in the plane of θ and δ: reversal occurs above the diagonal for positive θ and below it for negative θ. Lemma 1 is the classical selection bias factor [@kleinbaum; @greenland] with verification ratios, and its reversal region is the known possibility of reversal by selection [@nguyen; @banack]. *B* is not observable, because π⁰ conditions on latent *D*. What is observed is *B*_V = *g*₁/*g*₀ with *g*_t = P(*S* = 1 | *Y* = 0, *t*), the stratum average of *g*(*x̃*) among recorded negatives. It differs from *B* by a factor close to one when disease is rare, and Supplementary Section S1 bounds it.

Supplementary Section S1 adds Lemma 2. When the log ratio of malignant to benign verification probabilities is linear in *x* with slope λ, a population-optimal verified-only logistic learner shifts the disease coefficients by λ.

### E. Identification of the disease target

Let *q*(*x̃*) = P(*Y* = 1 | *x̃*), and let σ(*x̃*) = P(*S* = 1 | *x̃*) be the overall verification propensity. Let *s*(*x̃*) = P(*S* = 1 | *D* = 1, *x̃*) be the probability that a malignant lesion with input *x̃* is verified. Assume the pointwise floor *s*(*x̃*) ≥ *s*_min for every *x̃*, with *s*_min ∈ (0, 1]. This is stronger than a floor on the share of malignant lesions that are verified, which can exceed *s*_min while the pointwise floor fails in part of the input space.

Write *a*_k(*x̃*) = P(*t* = 1 | *x̃*)/P(*t* = 1) − P(*t* = 0 | *x̃*)/P(*t* = 0), so that ψ_k = E[*a*_k(*X̃*) logit *p*(*X̃*)].

**Proposition 2 (identified set for the disease target).** Let *p̄*(*x̃*) = min{*q*(*x̃*)/*s*_min, *q*(*x̃*) + 1 − σ(*x̃*)}. Let *L*(*x̃*) = logit *q*(*x̃*) where *a*_k(*x̃*) ≥ 0 and *L*(*x̃*) = logit *p̄*(*x̃*) where *a*_k(*x̃*) < 0. Let *H* swap the two endpoints. Where *a*_k = 0, either endpoint contributes zero. Then ψ_k lies in [ψ_L, ψ_U], with ψ_L = E[*a*_k(*X̃*)·*L*(*X̃*)] and ψ_U = E[*a*_k(*X̃*)·*H*(*X̃*)]. The endpoints are finite when logit *q* and logit *p̄* are integrable under the weights; if *q* = 0 or *p̄* = 1 on a set of positive weight, an endpoint is infinite. With the true *q*, σ and *a*_k this interval is sharp relative to the nonparametric model of the observed data that satisfies *Y* = *D*·*S* and the pointwise floor. Further restrictions on verification, such as structural, smoothness, monotonicity or cross-input constraints that the graph of Fig. 1(a) might motivate, could narrow it. The sign of ψ_k is identified as positive when ψ_L > 0 and as negative when ψ_U < 0.

**Corollary (stratum formula).** When *t* is a function of *x̃*, ψ_L = E[logit *q* | *t* = 1] − E[logit *p̄* | *t* = 0] and ψ_U = E[logit *p̄* | *t* = 1] − E[logit *q* | *t* = 0]. For any other input these stratum formulas are valid outer bounds.

*Proof sketch.* Pointwise, *p* = *q*/*s* lies between *q* and *p̄*, and both endpoints are attained. The target ψ_k increases in logit *p* where *a*_k > 0 and decreases where *a*_k < 0, so its extremes combine the endpoints by the sign of *a*_k. The stratum formulas assign endpoints by observed stratum instead of by the sign of *a*_k, which can only widen the interval. Supplementary Section S1 gives the proof. The population tipping floor *s*∗ is the smallest *s*_min at which ψ_L > 0, and we compute it from the exact bound. With the stratum formula, write Δ_q = E[logit *q* | *t* = 1] − E[logit *q* | *t* = 0], the contrast of a calibrated M0. Where *q*/*s*_min is the active branch and small, ψ_L ≈ Δ_q + log *s*_min, so *s*∗ ≈ exp(−Δ_q). This approximation only interprets *s*∗. Near *s*∗ it can give the wrong sign. With *q* = 0.01 and 0.02 in the two strata and σ = 0.1, it gives 0.495, while the exact floor is 0.50. Above *s*∗, a biopsy-trained learner with a negative contrast is reversed relative to disease.

The floor enters ψ_L only through *p̄*, where *a*_k < 0. A positive sign thus needs the floor only there, and with the stratum formula only on inputs that occur in the lower stratum. We state the floor for every *x̃* for simplicity. The tabular input contains the five concepts, so the stratum formula is sharp for the tabular-input target. Image features determine the concept only partly, so for the image-feature target we report the stratum formula as an outer set, which needs no estimate of *a*_k.

Proposition 2 assumes *q* and σ known. On ISIC-2024 we replace *q* by M0 on the learner's own input, calibrated on the validation set. We replace σ by one out-of-fold gradient-boosting model on the tabular features, shared by both families. {{N:nuis_sentence}} Probabilities are clipped to [10⁻⁷, 1 − 10⁻⁷] before the logit, and isotonic estimates to [10⁻⁶, 1 − 10⁻⁶]. This gives a plug-in estimate of the identified set. Sharpness does not carry over to it, and the bootstrap describes its sampling variability for a given nuisance specification, not whether that specification is correct. The primary plug-in uses the Platt calibration [@platt] applied to M0 and M2. No estimator of *q* is privileged by theory, so we report three under one bootstrap: M0 with Platt or isotonic [@zadroznyelkan] calibration, and gradient boosting [@friedman]. The smallest floor at which the plug-in lower endpoint is positive is the plug-in tipping floor. The smallest floor at which it is positive in at least 95 percent of bootstrap replicates is the bootstrap stability threshold. Both describe the plug-in procedure under one nuisance specification; neither is a floor identified from the data.

### F. Data and learners

**Cohorts.** ISIC-2024 contains 401,059 lesion image crops, one per lesion, from 1,042 patients [@kurtansky]. A lesion identifier marks 22,058 clinician-tagged lesions (*F*). Of these, 1,068 carry histopathology (*S*) and 393 are recorded malignant (*Y*). Every malignant label is histopathologically confirmed, and lesions never linked to pathology were recorded as benign [@kurtansky], so the release follows *Y* = *D*·*S*. PAD-UFES-20 contains 2,298 smartphone images of 1,641 lesions from 1,373 patients [@pacheco]. Basal cell carcinoma, squamous cell carcinoma and melanoma form the malignant class, and actinic keratosis, nevus and seborrheic keratosis the non-malignant class. Placing actinic keratosis, a premalignant lesion, in the non-malignant class is a binary mapping chosen for this analysis. Every image recorded as malignant was biopsied, and non-malignant lesions that were not biopsied carry a clinical diagnosis.

**Concepts.** We use five concepts measured by the total-body photography system [@kurtansky]: color variegation, size, lesion-skin contrast, asymmetry and border irregularity. Color variation, asymmetry and border are aspects of the ABCD rule of dermoscopy [@nachbar]. The system measures them on total-body photographs, not on dermoscopic images. The analysis lock made color variegation, size and lesion-skin contrast primary and the other two secondary.

**Split and regimes.** Patients were split 60/20/20 into train, validation and test, stratified by whether a patient had any malignant label. The three splits hold 624, 209 and 209 patients with 222, 92 and 79 recorded malignancies. The test population holds 78,625 lesions. M2 is trained on 622 verified training lesions, of which 222 are malignant. Every malignant training lesion is verified, so the two regimes differ only in the size of the training set and in the source of its recorded negatives.

**Learner families.** (i) A two-layer perceptron on 47 tabular features: 29 measurements of the total-body photography system and 18 columns for age, sex, body site, acquisition site and skin tone. No identifier, flag, biopsy or diagnosis field enters the input. (ii) The same perceptron on frozen ResNet-50 ImageNet features [@he]. (iii) A linear probe on the frozen ResNet-50. (iv) ResNet-50 fine-tuned end to end. M0 and M2 share every setting except the training lesions. The two frozen-feature families are primary. Supplementary Section S5 lists every setting and input column.

### G. Analyses and inference

**Primary inference.** The primary family is the three primary concepts in the two frozen-feature families, six comparisons. Learned contrasts are on the raw logit scale; calibration enters only the plug-in nuisances of Proposition 2 and the gap diagnostics. Point estimates average three seeds of the original fits. In each of {{N:B_primary}} joint bootstrap replicates per family, we resample training patients, refit M0 and M2 with three seeds and calibrate them on the validation set. We then resample test patients and average the three seeds, so the bootstrap statistic is the point estimator itself. This inference conditions on the realized validation set. It treats the observed patients as a sample from a population like the observed sites, so it does not cover a new site; for the fixed test population it is a sensitivity analysis to patient resampling (Supplementary Section S6). A second analysis with {{N:B19}} replicates per family and one seed per replicate targets the randomized one-seed training procedure and serves as a check. Two sensitivity analyses resample training, validation and test patients together, in {{N:B30}} replicates per family, or subsample patients without replacement. A learner reversal is called Bonferroni sign-separated when the Bonferroni-adjusted percentile bootstrap intervals [@efron] of Δ_M0 and Δ_M2, at level 1 − 0.05/6, lie on opposite sides of zero. The claim concerns one cell. A false claim in either direction needs a one-sided miss of probability at most 0.05/12, so the error is at most 0.05/6 per cell and 0.05 over six cells if the bootstrap tails are nominal. The intervals are not simultaneous intervals for the twelve contrasts. The label refers to the tertile definition and, unless stated otherwise, to the fixed validation set. Secondary concepts are reported with 95 percent intervals and are not part of the multiplicity family. {{N:mc_rule}} All other joint intervals refit one seed per replicate, so they target the randomized one-seed training procedure. They are the support-restricted contrasts of Section IV-D and Table 2, the gap analyses of Section IV-C and Fig. 3(b), and Table 3. For the linear-probe and fine-tuned families, intervals resample test patients only and condition on the fitted models.

**Status of analyses.** An analysis lock fixed the primary and secondary outcomes after the primary comparison had first been inspected and before the analyses that test alternative explanations. The lock is dated 25 September 2026 and is part of the first public commit of the repository. It is a structured post-inspection plan, not a preregistration, and the primary analysis is partly data-informed. The Bonferroni adjustment controls multiplicity within this locked six-comparison family only, so it does not make the primary inference confirmatory. After inspecting the primary results, we changed the primary bootstrap statistic from a one-seed refit to the three-seed average used by the point estimator. We added the Monte Carlo resolution rule at the same stage, which made color variegation with tabular features unresolved. The alternative concept definitions and learner families are robustness analyses of the same lesions and are not independent tests. Every other analysis in Sections IV-B to IV-G, apart from the size-matched control, was added after the primary results had been seen. This includes the design and outcome of the dose experiment and the marginal analysis of Section IV-G. We also chose color variegation as the worked example at that stage.

**Checks and simulation.** Symbolic and unit-test checks verify the algebra and the code, and a phase diagram checks Lemmas 1 and 2 on 648 populations (Supplementary Sections S2 and S3). Simulations test the plug-in set of Proposition 2 in synthetic populations and in an adversarial design with an overstated floor. Semi-synthetic designs on the real ISIC-2024 inputs include one where the floor fails where verification is rarest.

Each result in Section IV is read under its own condition. The learner reversal compares two fitted pipelines on one test population. The training-selection effect is identified by design, conditional on the cohort and split. The gap prediction is exact only for Bayes-optimal learners. Statements about disease hold only under the floor, and only up to the estimation of *q* and σ.

## IV. Results

Section IV-A reports the learner reversal that the rest of the section examines. Sections IV-B to IV-D test it against training-set size, training selection, the verification propensity and support. Section IV-E checks the attenuating case on PAD-UFES-20. Sections IV-F and IV-G turn to disease, first on the learner's scale and then as a marginal association.

### A. Learner reversal on ISIC-2024

Table 1 reports the learned contrasts. {{N:jb_sentence}}

{{N:resample_sentence}}

Fig. 2 shows all four families. The linear-probe and fine-tuned families also reversed color variegation, with no test bootstrap resample of the same sign, conditional on the fitted models. {{N:size_defs_sentence}} {{N:balanced_sentence}}

In Table 1, contrasts are raw logit contrasts on the ISIC-2024 test population. The primary rows show Bonferroni-adjusted percentile intervals at level 1 − 0.05/6 from 5,000 three-seed joint resamples of training and test patients, conditional on the realized validation set. The secondary rows show 95 percent intervals. "Unresolved" marks a sign-separation label of the full replicate set that was reproduced in fewer than 95 percent of Monte Carlo resamples of the stored replicates. The same-sign count is descriptive, not a test.

**TABLE 1. Learned contrasts of the frozen-feature learners on the ISIC-2024 test population.**

{{TABLE:main_reversal}}

In Fig. 2, M0 is shown as circles and M2 as triangles on panel-specific scales. Its bars are test-patient bootstrap intervals conditional on the fitted models and therefore differ from the joint training-and-test bootstrap intervals of the primary inference in Table 1.

**Fig. 2.** Learned contrasts of M0 and M2 in the four learner families.

### B. Controlled selection of recorded negatives can induce the reversal

Table 2 summarizes the tests of alternative explanations, and Supplementary Section S6 reports each in full.

A size-matched control combined the same malignant lesions with 320 recorded negatives per replicate. For color variegation and size, in both families, the contrast was positive in 20 of 20 replicates when the recorded negatives were drawn at random. It was negative in 20 of 20 when they were verified ones. For color variegation it was also negative in 20 of 20 when they were flagged but never biopsied. Training-set size alone therefore does not explain these two reversals. Each arm also drew its validation set from its own pool, so the arms differ in their development data, not only in the source of training negatives. For lesion-skin contrast with image features the random arm was itself close to zero, so the control does not separate size from selection there. {{N:arms_sentence}}

A controlled selection dose changed only which recorded negatives entered training (Fig. 3(a)). In each of 20 replicates, a validation set was drawn. It was shared by all five doses, together with the malignant training lesions and the seed. Each recorded-negative training lesion received one uniform variable that decided its inclusion at every dose. Each replicate therefore fixes ω, and contrasts between doses are paired potential outcomes. The outcome is the raw-logit contrast, because Platt calibration on the unselected validation set would target the unselected distribution (Supplementary Section S6).

{{N:dose_sentence}}

In Fig. 3(a), points are means of the raw logit contrast of color variegation over the 20 paired replicates with 320 expected recorded negatives, with 2.5 to 97.5 percentile ranges. Its dashed lines are the Bayes-optimal training-distribution shift given each learner's input, anchored at η = 0 and shown for direction only. For image features, they use a ridge estimate of the conditional concept mean. In Fig. 3(b), points are calibrated medians over 200 one-seed joint resamples with 95 percent percentile intervals, against the identity line, and labels name the concept of each point in its family's color.

**Fig. 3.** Controlled selection dose (a) and observed against predicted learner gap (b).

In Table 2, size-matched intervals are 2.5 and 97.5 percentiles over 20 replicates, and each arm drew its own validation set. Dose rows use 320 expected recorded negatives and the paired-replicate bootstrap intervals described above, and their ratios are descriptive. Predicted-shift rows use 10 replicates per arm. Lesion-level rows use 200 one-seed joint resamples, or 100 for the errors-in-variables diagnostic. Overlap rows are point estimates of the original fits, and the PAD-UFES-20 row uses 200 joint resamples.

**TABLE 2. Tests of alternative explanations for the learner reversal.**

| Check | Setting | Result |
| --- | --- | --- |
| Size-matched control, color variegation | tabular; recorded negatives from all / flagged / verified lesions; validation drawn per arm | +1.22 [0.60, 2.14] / −0.79 [−1.21, −0.38] / −0.94 [−1.12, −0.70]; positive in 20 / 0 / 0 of 20 |
| Size-matched control, color variegation | image; same three sources | +0.59 [0.31, 0.90] / −0.82 [−1.05, −0.58] / −0.63 [−0.81, −0.40]; positive in 20 / 0 / 0 of 20 |
{{N:table2_rows}}

### C. The learner gap follows the recorded-negative verification propensity

{{N:bridge_sentence}} Fig. 3(b) plots the observed against the predicted concept-level gap. Agreement between observed and predicted gaps also reflects approximation error, calibration, estimation of *g* and optimization. It is a diagnostic, not a test of the identity; the symbolic and unit tests verify the identity itself (Supplementary Section S2).

Verification of recorded negatives depended on appearance mainly within patients. A one standard deviation increase in color variegation relative to the patient's own lesions raised the log odds of verification by 0.93 (SE 0.04). For the patient's mean, the increase was 0.10 (SE 0.15). This describes a verification pattern, consistent with clinicians biopsying a patient's atypical-looking lesions, and does not identify the clinical mechanism. {{N:site_sentence}}

### D. Overlap

{{N:overlap_paragraph}}

### E. External consistency check on PAD-UFES-20

On PAD-UFES-20, every image recorded as malignant was biopsied, so *A* = 1 under the recorded-diagnosis operationalization and Lemma 1 then reduces to log OR_{D|S} = θ − log *B*. Verification attenuated ten of eleven association features, amplified one and reversed none (Fig. 4). The learner contrasts showed the same qualitative attenuation. With patients split 60/20/20, M2 trained on the 814 biopsied training images had a smaller contrast than M0 for all six symptoms. For image color variegation, the contrast fell from +2.06 [1.47, 2.84] to +0.70 [0.22, 1.23]. The cohorts differ in modality, population, prevalence and workflow. PAD-UFES-20 illustrates the attenuating regime of Lemma 1 and validates neither the mechanism on ISIC-2024 nor Lemma 1 for latent disease. If some clinically diagnosed lesions were malignant, *A* would differ from one. The direction depends on where the missed malignancies fall. Missed cases only in the lower tertile would raise *A* above one, and only in the upper tertile would lower it below one.

In Fig. 4, the PAD-UFES-20 features are placed at *A* = 1 under the recorded-diagnosis operationalization. Each ISIC-2024 path uses whole-cohort counts. For π₀¹ = 0.5 and *A* from 0.5 to 2, it computes the disease risks of both tertiles and then θ and δ exactly. Every point satisfies θ − δ = log OR_{D|S}, the verified-only log odds ratio of the counts. Each path therefore has slope one and keeps the sign of the verified-only association, while θ crosses zero exactly at *A* = RR_Y. Open circles mark *A* = 1; moving π₀¹ from 0.25 to 1 shifts them by less than 0.01. The segments are sensitivity paths, not confidence intervals, and the two cohorts are not matched counterfactuals of each other.

**Fig. 4.** ISIC-2024 and PAD-UFES-20 in the plane of Lemma 1.

### F. Learner-scale disease target: an assumption-indexed sensitivity analysis

{{N:psi_primary_sentence}} {{N:sharp_sentence}}

{{N:psi_est_sentence}} {{N:tail_sentence}}

In Table 3, Δ_q is the difference in mean logit *q* between the outer tertiles of the test population for the original fits. The point floor uses the exact bound on the seed-averaged curve of ψ_L, and the Monte Carlo range of the stability threshold comes from 500 resamples of the replicates. Ranges across estimators of *q* summarize specification sensitivity and are not statistical confidence intervals. Joint counts are resampling frequencies under the stated plug-in procedure, not probabilities that a disease-relative reversal holds. Log-loss is on the test population for the original fits.

**TABLE 3. Plug-in tipping floor for color variegation under three estimates of *q*.**

{{TABLE:psi_est}}


{{N:psi_sim_sentence}}

### G. Separate post-inspection marginal sensitivity analysis

{{N:rr_sentence}} {{N:site_sentence_marg}} {{N:benchmark_sentence}}

Taken together, the results separate three kinds of evidence. The learner reversal and its response to a designed change in training selection concern the fitted pipelines and hold under the stated designs. The gap prediction from the verification propensity held in direction for every concept, but its size depended on the propensity model, and color variegation lost sign separation within better-supported regions. Statements about disease depended on an assumed floor and, on the learner's scale, also on the estimator of *q*.

## V. Discussion

We first state what the evidence supports and what it does not, then the main threats to validity, and finally what the results imply for reading the contrasts of a classifier.

**What the evidence supports.** A designed change in which recorded negatives entered training moved what a classifier learned in the predicted direction. With 320 expected recorded negatives, the mean learned contrast also crossed zero. The observed learner reversal between M0 and M2 persisted when training-set size was matched, with each arm using its own validation set. The gap between the two learners had the sign predicted from the recorded-negative verification propensity for all five concepts in both frozen-feature families. Still, the reversal itself is not a randomized contrast. Relative to disease, only the separate whole-cohort marginal analysis indicated a reversal. It did so under a lower-tertile floor on malignant verification that these data cannot check. The result is a mixture over the observed acquisition sites, and the present analysis does not identify or validate transport to a new site.

**What the evidence does not support.** We do not claim that verification always reverses learned relationships; in the phase diagram no population reversed when benign verification did not depend on appearance. We do not claim a Bonferroni sign-separated learner reversal for every concept, or a learner reversal of color variegation within better-supported regions, where the M0 contrast fell and could turn negative. We do not claim a learner-scale disease-relative reversal. That would need a floor on malignant verification where verification is rarest, which this cohort cannot check, and a better estimate of recorded-label risk in the rare-event tail. The controlled experiment intervenes on training selection, not on clinical biopsy decisions, and we do not claim that its doses match the strength of clinical verification. Localization diagnostics on the fine-tuned encoders found no evidence of a general representation-level reversal (Supplementary Section S5).

**Threats to validity.** *Concepts.* The concepts are measurements from total-body photography, and the results apply to them as measured. *Inference.* The test population holds 209 patients, and site-level heterogeneity in verification was substantial. Fine-tuning under M0 was unstable across seeds, which is why that family is secondary. *Estimation.* The width of the identified set under the floor does not shrink with more data. {{N:local_sentence}}

**Implications.** Concept-level readings of a classifier trained on biopsied lesions should not be read as statements about disease. This holds for the stratum contrasts studied here and, plausibly, for directional derivatives along concept vectors such as TCAV [@kim], which we did not study. Marginal concept contrasts should not be read as isolated concept effects. Comparisons across data sets should record how each was verified. The selection propensity can be estimated only if a data set records the full cohort of candidate lesions with an indicator of biopsy. Recording the clinical reasons for biopsy could support point identification of disease risk. This would require conditional exchangeability of verification with respect to disease given the recorded information, positivity of verification, and correct measurement of that information.

In short, the data support conclusions about the pipeline: which training data a learner sees changes the concept contrast it learns, in the direction the verification propensity predicts. They do not support conclusions about disease without a floor on malignant verification that the cohort cannot check.

## VI. Conclusion

We asked whether selective verification changes what a classifier learns. The training-selection offset is exact for Bayes-optimal learners and observable for biopsy-based training. In trained learners its predicted direction held, while agreement in magnitude was only descriptive. On ISIC-2024, the two regimes gave color variegation and size contrasts of opposite sign on the full test population. Size with image features kept Bonferroni sign separation under every resampling scheme, but only for the tertile contrast. Color kept opposite point-estimate signs under every concept definition, but lost sign separation within better-supported regions. Because the designed change in training selection was applied on shared algorithmic randomness, its effect is identified by design, conditional on the cohort. Each selected contrast moved in the predicted direction and, with 320 expected recorded negatives, its mean crossed zero. Relative to latent disease, Proposition 2 identifies a learner-scale set only under a floor on malignant verification. On ISIC-2024, the floor at which its plug-in lower endpoint became positive moved with the estimator of recorded-label risk. So the cohort does not establish a learner-scale disease-relative reversal. The separate post-inspection marginal analysis reached a disease association of color variegation opposite to its verified-only association, and only by assuming a floor on malignant verification. Changing by experiment which recorded negatives entered training altered the learned concept contrast. Training selection can therefore shape the statistical relationships a classifier learns, not only filter how it is evaluated.

## Statements

**Data availability.** Both data sets are public. ISIC-2024 is distributed through the ISIC Archive [@kurtansky] and PAD-UFES-20 through Mendeley Data [@pacheco]. No new patient data were collected.

**Code availability.** The scripts that produce every number, table and figure in this paper are available at https://github.com/dqtoan87/CausalReversal, together with the symbolic checks, the unit tests and the analysis lock. The version reported here is release v1.7-submission. Supplementary Section S8 maps each result to its script.

**Ethics.** This is a secondary analysis of two de-identified public data sets released under the approvals documented by their providers. No further approval was required.

**Author contributions.** Quang Toan Dao designed the study, derived the theory, implemented the pipeline, ran the analyses and drafted the manuscript. Viet Anh Nguyen supervised the study and revised the manuscript.

**Funding.** This work received no specific grant from any funding agency.

**Conflicts of interest.** The authors declare no competing interests.

<!-- REFERENCES -->
