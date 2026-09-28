# Causal Analysis of Learning Reversal under Selective Verification in Skin Cancer Classification

**Quang Toan Dao**¹·² (dqtoan@ioit.ac.vn), **Viet Anh Nguyen**¹ (anhnv@ioit.ac.vn)

¹ Institute of Information Technology, Vietnam Academy of Science and Technology, Ha Noi, Viet Nam

² Graduate University of Science and Technology, Vietnam Academy of Science and Technology, Ha Noi, Viet Nam

## Abstract

In skin cancer cohorts, disease is confirmed only for biopsied lesions, while unverified lesions enter training as recorded negatives. We ask whether selecting training data on verification can reverse the concept contrast a classifier learns, and what is identified about disease. When training lesions are selected on the recorded label and the learner's input, the Bayes-optimal logit shifts by the log selection probability. For biopsy-only training, this shift equals the observable log verification propensity among recorded negatives. On 401,059 ISIC-2024 lesions, learners trained on all and on biopsied lesions gave opposite-sign estimates for color variegation and size. Within a post-inspection family of six comparisons, Bonferroni-adjusted intervals separated the signs for image-feature size under every resampling scheme and for image-feature color variegation with validation fixed. Tabular-feature color variegation was unresolved under Monte Carlo error. The color and size reversals persisted after matching training-set size. In a post-inspection experiment, concept-specific doses changed only which recorded negatives entered training. Each dose shifted its primary contrast in the predicted direction, and across zero near the training size of the biopsy-only regime. For latent disease, a floor on malignant verification yields a population identified set, sharp with known nuisance functions; for image features we report a conservative outer bound. On ISIC-2024, learner-scale sign certification depended on the estimator of recorded-label risk. A separate post-inspection marginal analysis implied a disease association opposite to the verified-only one, but only under assumptions on malignant verification. Changing verification-like training selection can alter the concept contrast a classifier learns.

**Keywords:** causal inference, selection bias, selective labels, case-control sampling, partial identification, verification bias, skin lesion classification

## I. Introduction

Most labels in medical imaging are verified selectively. A lesion receives a histopathologic diagnosis only if a clinician decided to biopsy it. In ISIC-2024, lesions never linked to a pathology report and not tagged as biopsied were recorded as benign [1]. Only 1,068 of 401,059 lesions carry a tissue diagnosis. Classifiers trained on such data are reported to rival dermatologists [2], yet their labels reflect disease and clinical work-up together.

Selection is usually treated as a problem of estimation or evaluation. Verification bias corrupts sensitivity and specificity [3]. Conditioning on a selection node opens paths that no adjustment set closes [4]. Selective labels complicate algorithm evaluation [5]. Less studied is what verification does to the relationship a model learns between a visual feature and disease. A feature can accompany disease and also prompt the decision to biopsy. Training on verified lesions then conditions on that decision [6].

We ask two questions. First, does the selection that produced a training set change the sign of the concept contrast a classifier learns? Second, when the learned sign differs between training regimes, which learner is reversed relative to latent disease, and under which assumption can that be identified? We do not estimate the effect of intervening on a concept.

We make three contributions.

**First, an observable offset for what selected training data teach a learner.** Selecting training data on the recorded label and the input offsets the Bayes-optimal logit by the log selection probability. For biopsy-based training, this is the log verification propensity among recorded negatives. The identity belongs to the family of case-control and local case-control offsets [7], [8] and characterizes the Bayes-optimal shift exactly. We apply it to selective clinical labels. For trained learners, it gives an observable diagnostic, concept by concept. The diagnostic covers the gap between learners trained on all and on verified lesions, and the response to a controlled change in training selection. We test the predicted direction of both.

**Second, evidence that selection on verification can reverse the contrast a classifier learns.** On ISIC-2024, learners trained under the two regimes learn opposite signs for color variegation and size on the full patient-disjoint test population. The strongest multiplicity-adjusted evidence for the marginal tertile contrast is for size with image features. Color variegation is the one concept whose reversal held under every concept definition, but it does not carry over to regions with stronger verified-training support. The color and size reversals survive a size-matched control. After observing this reversal, we designed a controlled, verification-like selection of training data. It does not intervene on clinical biopsy decisions. It identifies the effect of the implemented selection rule on the learned contrast. Each concept-specific dose shifted the corresponding contrast in the direction the offset predicts. At a training size close to that of the biopsy-only learner, it also crossed zero.

**Third, what the verification process leaves identified about disease.** We define the disease target on the same functional and population as the learned contrast and derive its population identified set under a floor on malignant verification. The set is sharp when the nuisance functions are known; a stratum-based outer set, used for image features, needs no estimate of how the input predicts the stratum. For the marginal association the condition reduces to a bound on one verification ratio. We analyzed ISIC-2024 after the primary results had been seen. There the framework does not establish disease-relative sign claims. Instead, it shows why they stay sensitive to the assumed floor and to the estimator of recorded-label risk. PAD-UFES-20, where every recorded malignancy is biopsied [9], serves as an external consistency check on the attenuating side of the boundary.

**How the parts fit.** The three parts rest on different kinds of evidence. Keeping them apart, more than any single identity, is the causal contribution. We implement each selection rule for training data directly on shared algorithmic randomness, so its effect on the learned contrast, conditional on the cohort, is identified by design. The observed verification process yields, through a classical sampling identity, a diagnostic for the gap between the two training regimes. Latent disease is not point identified from these data, so statements about it are sets indexed by an explicit floor on malignant verification.

## II. Related Work

**Selection bias and its direction.** Epidemiology expresses selection bias as a multiplicative factor on the odds ratio, built from the selection probabilities of the exposure-by-outcome table [10]. Quantitative bias analysis turns that factor into sensitivity parameters [11], [12]. Berkson described selection on a common effect [6]. The direction of collider bias between binary variables has been characterized in closed form [13]. Reversal by selection has been analyzed for the obesity paradox [14], and collider bias affects modern cohorts [15]. Our Lemma 1 restates these results with verification ratios, and we do not claim it as new.

**Case-control and subsampling offsets.** Case-control sampling that depends on the outcome alone leaves logistic slopes unchanged and moves the intercept [7]. Local case-control sampling selects on the outcome and the covariates and corrects the fit by a known offset [8]. Our Proposition 1 is an identity of this family, written for training data selected by clinical verification.

**Selective labels.** When outcomes are observed only after a human decision, evaluation and learning depend on that decision [5], [16]. Expert consistency [17] and selective testing [18] have been used to learn under such selection. Risk prediction under historical testing can be improved with domain constraints such as known prevalence [19]. Bounds on predictive performance under selectively observed outcomes and unobserved confounding use debiased nuisance estimation [20]. Fairness has been characterized over the set of good models under selective labels [21]. We do not propose a new bounding framework. We bound a learner-matched logit contrast of disease and connect it to an observable learner reversal.

**Positive-unlabeled learning.** Writing *Y* = *D*·*S* makes every unverified malignancy an unlabeled positive, the structure of positive-unlabeled learning [22], [23], including feature-dependent labeling propensities [24]. We use its decomposition of *q* to bound the disease contrast when the labeling propensity is unknown.

**Verification bias, missing data and shortcuts.** Verification bias in diagnostic accuracy [3] can be corrected under ignorable verification [25]. Graphical criteria address recovery from selection [26] and missing data [27], and learning under feature-dependent [28] and outcome-dependent selection [29] is classical. Dermatology classifiers exploit skin markings [30] and hidden strata [31], shortcut learning is general [32], and acquisition shapes what a model can learn [33].

**Partial identification and probing.** When a parameter is not point identified, one can report the set of values compatible with the data and stated assumptions [34], [35], and learn robustly over such sets [36]. Linear probes [37] and centered kernel alignment [38] serve only in supplementary localization diagnostics.

## III. Method

### A. Causal model

Let *D* be the latent disease state and *X* the lesion's appearance. Concepts are deterministic readings of the image, *C* = *h*(*X*). Diagnosis is anticausal: disease shapes appearance, *D* → *X* → *C*. Let *U* be patient-level risk and surveillance intensity, *G* the acquisition site and *W* clinical information outside the image, such as reported change. A clinician flags lesions of interest (*F*) and biopsies some of them (*S*, with *S* ⊂ *F*). The recorded label is *Y* = *D*·*S*. Fig. 1(a) shows the graph, with latent nodes dashed. Verification depends on disease through *W* and through appearance, so disease is missing not at random among unverified lesions, whose recorded label is nonetheless zero. Lesions with *Y* = 0 are recorded negatives. Only those with *S* = 1 are verified benign.

**Fig. 1.** Causal model of selective verification (a) and the regions of Lemma 1 (b).

A learner does not see *X* itself but an input *X̃*: tabular measurements or image features. A training regime is a selection *R* of lesions into the training sample. M0 trains on all lesions (*R* ≡ 1). M2 trains on verified lesions (*R* = *S*). The identities below use only the variables they name, so they hold for any graph consistent with *Y* = *D*·*S*. The graph describes how the data arose and is not the basis of identification.

### B. Targets

Split concept *k* at its cohort tertiles into lower (*t* = 0) and upper (*t* = 1) strata. For a learner *m*, the **learned contrast** Δ_{m,k} is the mean logit among upper-stratum lesions minus that among lower-stratum lesions, on a fixed evaluation population, averaged over the randomness of training. It is a marginal tertile contrast: it includes differences in correlated concepts between the strata and is not the effect of concept *k* with the others held fixed. We drop *k* when the concept is clear. Write *p*(*x̃*) = P(*D* = 1 | *x̃*). The **disease target** on the same scale and population is

ψ_k = E[logit *p*(*X̃*) | *t* = 1] − E[logit *p*(*X̃*) | *t* = 0].

Because *p* depends on the input, ψ_k is specific to a learner family. The tabular-input and image-feature disease targets are different functionals of the same latent disease.

A **learner reversal** is a difference in sign between Δ_{M0,k} and Δ_{M2,k}. It is observable. But M0 and M2 differ in more than one respect, so a learner reversal is not by itself the effect of a selection rule. We define that effect next. A **learner-scale disease-relative reversal** is a difference in sign between Δ_{M2,k} and ψ_k. It can be established only as far as the identified set for ψ_k determines the sign of ψ_k. We also use the marginal disease contrast θ_k = log OR_D between the strata, the classical object of selection bias. A **marginal association reversal** is a positive θ_k with a negative verified-only association log OR_{D|S} (Lemma 1), or the reverse.

**Training-selection effect.** Let η index a rule, set by the analyst, for selecting recorded negatives into the training set only. The validation set used for early stopping is not changed by η, and calibration is not part of this outcome. Let ω collect the exogenous random numbers of the pipeline: the uniform variables that decide inclusion, the draw of malignant training lesions, the validation-set draw used for early stopping and the optimizer seed. The contrast of a single fit is then a potential outcome Δ_j(η, ω), and the training-selection effect on concept *j* is τ_j(η₁, η₀) = E_ω[Δ_j(η₁, ω) − Δ_j(η₀, ω)]. The analyst directly implements each selection rule while holding the paired algorithmic randomness ω fixed within a replicate. So τ_j is identified by design, conditional on the realized cohort, patient split and evaluation population, with no assumption about clinical verification. It measures how an experimental selection rule changes what a learner learns. It is not the effect of a concept on disease, nor of a clinical biopsy policy, nor a superpopulation effect over repeated patient samples.

### C. The training-selection offset

**Proposition 1 (training-selection offset).** Write *r*_y(*x̃*) = P(*R* = 1 | *Y* = *y*, *x̃*) for the probability that a lesion with recorded label *y* and learner input *x̃* enters training. Then

logit P(*Y* = 1 | *x̃*, *R* = 1) = logit P(*Y* = 1 | *x̃*) + log *r*₁(*x̃*) − log *r*₀(*x̃*).

*Proof sketch.* By Bayes' rule, the odds of *Y* = 1 given *x̃* and *R* = 1 equal P(*Y* = 1 | *x̃*)·*r*₁(*x̃*) divided by P(*Y* = 0 | *x̃*)·*r*₀(*x̃*). Supplementary Section S1 gives the proof. The identity is stated on the learner's own input and uses no disease label.

Suppose *R* is independent of *X̃* given *X* and *Y*, as when selection depends only on appearance and the recorded label. The selection probability then averages over what the input does not show: *r*₀(*x̃*) = E[*r*₀(*X*) | *x̃*, *Y* = 0]. Two cases matter here. *Training on verified lesions.* With *R* = *S*, every malignant training lesion is verified, so *r*₁ = 1. Then *r*₀(*x̃*) = *g*(*x̃*) = P(*S* = 1 | *x̃*, *Y* = 0), the verification propensity among recorded negatives. No further condition is needed. For the Bayes-optimal learners *f*₀ of M0 and *f*₂ of M2, logit *f*₀ − logit *f*₂ = log *g*, and therefore

Δ_{M0,k} − Δ_{M2,k} = E[log *g*(*X̃*) | *t* = 1] − E[log *g*(*X̃*) | *t* = 0].

For the Bayes-optimal learners, a reversal occurs exactly when this contrast exceeds the M0 contrast in the same direction. *A controlled selection dose.* Let each recorded-negative training lesion enter independently with probability π_η = min{1, κ_η exp(η*z*_k(*X*))}. Here *z*_k is the standardized concept, and κ_η fixes the expected number. If recorded positives are drawn uniformly, *r*₀ is known by design. When *z*_k is part of the learner's input, *r*₀(*x̃*) = π_η(*z*_k). Without the cap, the contrast of any concept *j* then moves linearly: Δ_j(η) = Δ_j(0) − η·μ_{kj}. Here μ_{kj} is the difference in mean *z*_k between the strata of concept *j*. The Bayes-optimal training-distribution shift is then −(η₁ − η₀)·μ_{kj}. When the input determines *z*_k only partly, as for image features, *r*₀(*x̃*) = E[π_η(*z*_k(*X*)) | *x̃*, *Y* = 0]. We approximate it with a Gaussian model for *z*_k given *x̃* (Supplementary Section S1). Adding a constant to a logit cancels in Δ.

These statements are exact for the Bayes-optimal logit of the selected training distribution. A trained network adds approximation, optimization and early-stopping effects. For the effect τ_j of a trained pipeline, they predict direction only; agreement in magnitude is a descriptive diagnostic (Section IV).

### D. Classical identities for the disease association

**Lemma 1 (verification selection factor).** For the strata of concept *k*, let π_t^d = P(*S* = 1 | *D* = *d*, *t*), *A* = π₁¹/π₀¹, *B* = π₁⁰/π₀⁰ and δ = log *B* − log *A*. Then log OR_{D|S} = θ − δ. For θ > 0, verification amplifies the association if δ < 0, attenuates it if 0 < δ < θ, and reverses it if δ > θ. For θ < 0 the regions mirror, and if θ = 0 verification alone creates an association equal to −δ.

Fig. 1(b) shows these regions in the plane of θ and δ: reversal occurs above the diagonal for positive θ and below it for negative θ. Lemma 1 is the classical selection bias factor [10], [11] with verification ratios, and its reversal region is the known possibility of reversal by selection [13], [14]. *B* is not observable, because π⁰ conditions on latent *D*. What is observed is *B*_V = *g*₁/*g*₀ with *g*_t = P(*S* = 1 | *Y* = 0, *t*). It differs from *B* by a factor close to one when disease is rare, and Supplementary Section S1 bounds it.

Supplementary Section S1 adds Lemma 2. When the log ratio of malignant to benign verification probabilities is linear in *x* with slope λ, a population-optimal verified-only logistic learner shifts the disease coefficients by λ.

### E. Identification of the disease target

Let *q*(*x̃*) = P(*Y* = 1 | *x̃*), and let σ(*x̃*) = P(*S* = 1 | *x̃*) be the overall verification propensity. Let *s*(*x̃*) = P(*S* = 1 | *D* = 1, *x̃*) be the probability that a malignant lesion with input *x̃* is verified. Assume the pointwise floor *s*(*x̃*) ≥ *s*_min for every *x̃*. This is stronger than a floor on the share of malignant lesions that are verified, which can exceed *s*_min while the pointwise floor fails in part of the input space.

Write *a*_k(*x̃*) = P(*t* = 1 | *x̃*)/P(*t* = 1) − P(*t* = 0 | *x̃*)/P(*t* = 0), so that ψ_k = E[*a*_k(*X̃*) logit *p*(*X̃*)].

**Proposition 2 (identified set for the disease target).** Let *U*(*x̃*) = min{*q*(*x̃*)/*s*_min, *q*(*x̃*) + 1 − σ(*x̃*)}. Let *L*(*x̃*) = logit *q*(*x̃*) where *a*_k(*x̃*) ≥ 0 and *L*(*x̃*) = logit *U*(*x̃*) where *a*_k(*x̃*) < 0. Let *H* swap the two endpoints. Where *a*_k = 0, either endpoint contributes zero. Then ψ_k lies in [ψ_L, ψ_U], with ψ_L = E[*a*_k(*X̃*)·*L*(*X̃*)] and ψ_U = E[*a*_k(*X̃*)·*H*(*X̃*)]. With the true *q*, σ and *a*_k this interval is sharp relative to the nonparametric model of the observed data that satisfies *Y* = *D*·*S* and the pointwise floor. Further restrictions on verification, such as structural, smoothness, monotonicity or cross-input constraints that the graph of Fig. 1(a) might motivate, could narrow it. The sign of ψ_k is identified as positive when ψ_L > 0 and as negative when ψ_U < 0.

**Corollary (stratum formula).** When *t* is a function of *x̃*, ψ_L = E[logit *q* | *t* = 1] − E[logit *U* | *t* = 0] and ψ_U = E[logit *U* | *t* = 1] − E[logit *q* | *t* = 0]. For any other input these stratum formulas are valid outer bounds.

*Proof sketch.* Pointwise, *p* = *q*/*s* lies between *q* and *U*, and both endpoints are attained. The target ψ_k increases in logit *p* where *a*_k > 0 and decreases where *a*_k < 0, so its extremes combine the endpoints by the sign of *a*_k. The stratum formulas assign endpoints by observed stratum instead of by the sign of *a*_k, which can only widen the interval. Supplementary Section S1 gives the proof. With the stratum formula and small *q*, ψ_L ≈ Δ_q + log *s*_min. Here Δ_q = E[logit *q* | *t* = 1] − E[logit *q* | *t* = 0] is the contrast of a calibrated M0. The sign of ψ_k is then identified once *s*_min exceeds the population tipping floor *s*∗ ≈ exp(−Δ_q). Above that floor, a biopsy-trained learner with a negative contrast is reversed relative to disease.

The floor enters ψ_L only through *U*, where *a*_k < 0. A positive sign thus needs the floor only there, and with the stratum formula only on inputs that occur in the lower stratum. We state the floor for every *x̃* for simplicity. The tabular input contains the five concepts, so the stratum formula is sharp for the tabular-input target. Image features determine the concept only partly, so for the image-feature target we report the stratum formula as an outer set, which needs no estimate of *a*_k.

Proposition 2 assumes *q* and σ known. On ISIC-2024 we replace *q* by the calibrated M0 and σ by an out-of-fold model, which gives a plug-in estimate of the identified set. Sharpness does not carry over to it, and the bootstrap describes its sampling variability for a given nuisance specification, not whether that specification is correct. The primary plug-in uses the Platt calibration [39] applied to M0 and M2. No estimator of *q* is privileged by theory, so we report three under one bootstrap: M0 with Platt or isotonic [40] calibration, and gradient boosting [41]. The smallest floor at which the plug-in lower endpoint is positive is the plug-in tipping floor. The smallest floor at which it is positive in at least 95 percent of bootstrap replicates is the bootstrap stability threshold. Both describe the plug-in procedure under one nuisance specification; neither is a floor identified from the data.

### F. Data and learners

**Cohorts.** ISIC-2024 contains 401,059 lesion tiles from 1,042 patients [1]. A lesion identifier marks 22,058 clinician-tagged lesions (*F*). Of these, 1,068 carry histopathology (*S*) and 393 are recorded malignant (*Y*). Every malignant label is histopathologically confirmed, and lesions never linked to pathology were recorded as benign [1], so the release follows *Y* = *D*·*S*. PAD-UFES-20 contains 2,298 smartphone images of 1,641 lesions from 1,373 patients [9]. Basal cell carcinoma, squamous cell carcinoma and melanoma form the malignant class, and actinic keratosis, nevus and seborrheic keratosis the non-malignant class. Placing actinic keratosis, a premalignant lesion, in the non-malignant class is a binary mapping chosen for this analysis. Every image recorded as malignant was biopsied, and non-malignant lesions that were not biopsied carry a clinical diagnosis.

**Concepts.** We use five concepts from the total-body photography metadata: color variegation, size, lesion-skin contrast, asymmetry and border irregularity. The first three are primary, following clinical dermoscopy criteria [42].

**Split and regimes.** Patients were split 60/20/20 into train, validation and test, stratified by whether a patient had any malignant label. The test population holds 78,625 lesions from 209 patients. M2 is trained on 622 verified training lesions, of which 222 are malignant. Every malignant training lesion is verified, so the two regimes differ only in the size of the training set and in the source of its recorded negatives.

**Learner families.** (i) A two-layer perceptron on 47 tabular features. (ii) The same perceptron on frozen ResNet-50 ImageNet features [43]. (iii) A linear probe on the frozen ResNet-50. (iv) ResNet-50 fine-tuned end to end. M0 and M2 share every setting except the training lesions. The two frozen-feature families are primary. Supplementary Section S5 lists every setting.

### G. Analyses and inference

**Primary inference.** The primary family is the three primary concepts in the two frozen-feature families, six comparisons. Point estimates average three seeds of the original fits. In each of 5,000 joint bootstrap replicates per family, we resample training patients, refit M0 and M2 with three seeds and calibrate them on the validation set. We then resample test patients and average the three seeds, so the bootstrap statistic is the point estimator itself. This inference conditions on the realized validation set. A second analysis with 5,000 replicates per family and one seed per replicate targets the randomized one-seed training procedure and serves as a check. Two sensitivity analyses resample training, validation and test patients together, in 1,000 replicates per family, or subsample patients without replacement. A learner reversal is called Bonferroni sign-separated when the Bonferroni-adjusted percentile bootstrap intervals [44] of Δ_M0 and Δ_M2, at level 1 − 0.05/6, lie on opposite sides of zero. The label refers to the tertile definition and, unless stated otherwise, to the fixed validation set. Secondary concepts are reported with 95 percent intervals and are not part of the multiplicity family. Because the Bonferroni endpoints lie far in the tails, we resampled the stored replicates 2,000 times. We call a label resolved at the Monte Carlo resolution when at least 95 percent of these resamples reproduce it. All other joint intervals refit one seed per replicate, so they target the randomized one-seed training procedure. They are the support-restricted contrasts of Section IV-D and Table 2, the gap analyses of Section IV-C and Fig. 3(b), and Table 3. For the linear-probe and fine-tuned families, intervals resample test patients only and condition on the fitted models.

**Status of analyses.** An analysis lock fixed the primary and secondary outcomes after the primary comparison had first been inspected and before the analyses that test alternative explanations. It is a structured post-inspection plan, not a preregistration, and the primary analysis is partly data-informed. The Bonferroni adjustment controls multiplicity within this locked six-comparison family only, so it does not make the primary inference confirmatory. After inspecting the primary results, we changed the primary bootstrap statistic from a one-seed refit to the three-seed average used by the point estimator. This makes the resampled statistic match the reported one. We added the Monte Carlo resolution rule at the same stage. This changed the classification of one primary cell: color variegation with tabular features is reported as unresolved, with the one-seed and three-seed results both shown. The alternative concept definitions and learner families are robustness analyses of the same lesions and are not independent tests. Every other analysis in Sections IV-B to IV-G, apart from the size-matched control, was added after the primary results had been seen. This includes the design and outcome of the dose experiment and the marginal analysis of Section IV-G. We also chose color variegation as the worked example at that stage.

**Checks and simulation.** Symbolic and unit-test checks verify the algebra and the code, and a phase diagram checks Lemmas 1 and 2 on 648 populations (Supplementary Sections S2 and S3). Simulations test the plug-in set of Proposition 2 in synthetic populations and in an adversarial design with an overstated floor. Semi-synthetic designs on the real ISIC-2024 inputs include one where the floor fails where verification is rarest.

## IV. Results

### A. Learner reversal on ISIC-2024

Table 1 reports the learned contrasts. The primary rule uses 5,000 joint resamples per family of the three-seed average. Under it, two of the six primary cells showed a Bonferroni sign-separated learner reversal with a resolved label: color variegation and size, both with image features. In three other cells, whose labels were also resolved, the Bonferroni interval of Δ_M0 reached zero once training variability was included. The remaining cell, color variegation with tabular features, sits on the boundary. Its lower Bonferroni endpoint of Δ_M0 was +0.01, so the two intervals were sign-separated in the full set of replicates. Yet this label was reproduced in only 72 percent of Monte Carlo resamples, and we report it as unresolved. The other five labels were reproduced in every Monte Carlo resample, and the Monte Carlo standard error of the endpoints was at most 0.07. The gap Δ_M0 − Δ_M2 was more stable than the full reversal: its 95 percent interval excluded zero in all 10 primary and secondary cells.

Three checks vary the resampling. The 5,000-replicate one-seed analysis targets the randomized one-seed training procedure and had Monte Carlo standard errors of at most 0.06. It agreed on every label except color variegation with tabular features, whose one-seed Bonferroni interval of Δ_M0 was [−0.15, 2.08]. Resampling validation patients as well, in 1,000 one-seed replicates per family, changed only the labels of color variegation with tabular and image features: their Bonferroni intervals of Δ_M0 then reached zero. Half-sampling patients without replacement reversed no label, and we use it only as this check (Supplementary Section S6).

Fig. 2 shows all four families. The linear-probe and fine-tuned families also reversed color variegation, with no test bootstrap resample of the same sign, conditional on the fitted models. The two concepts are robust along different axes. Size with image features is the strongest multiplicity-adjusted result but rests on the tertile definition: with slope definitions its M2 contrast was −0.02 and, within patients, +0.01. Color variegation reversed under all seven concept definitions in all four families (Supplementary Section S4). Balancing the tertiles on the other four measured concepts changes the estimand. This removed the reversal of size with image features (M0 +0.09, M2 +0.30) and nearly that of color variegation with image features (M2 −0.07). Color variegation with tabular features kept separated signs. These intervals condition on the fitted learners and are not part of the primary joint-bootstrap multiplicity analysis (Supplementary Section S6).

In Table 1, the primary rows show Bonferroni-adjusted percentile intervals at level 1 − 0.05/6 from the three-seed joint bootstrap, conditional on the realized validation set. The secondary rows show 95 percent intervals. "Unresolved" marks a sign-separation label of the full replicate set that was reproduced in fewer than 95 percent of Monte Carlo resamples of the stored replicates. The same-sign count is descriptive, not a test.

**TABLE 1. Learned contrasts of the frozen-feature learners on the ISIC-2024 test population.**

| Block | Concept | Features | Δ_M0 [interval] | Δ_M2 [interval] | Δ_M0 minus Δ_M2 [95% CI] | Same-sign resamples | Bonferroni sign-separated |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| Primary | Color variegation | tabular | +1.02 [0.01, 1.88] | −0.95 [−1.50, −0.51] | +1.97 [1.23, 2.57] | 19 of 5,000 | unresolved (yes in the full set) |
| Primary | Size | tabular | +0.65 [−0.25, 1.38] | −0.46 [−0.79, −0.07] | +1.12 [0.37, 1.55] | 182 of 5,000 | no |
| Primary | Lesion-skin contrast | tabular | +0.52 [−0.76, 1.60] | −1.50 [−2.17, −0.92] | +2.02 [1.09, 2.68] | 815 of 5,000 | no |
| Primary | Color variegation | image | +1.64 [0.26, 2.33] | −0.66 [−1.24, −0.42] | +2.30 [1.36, 2.81] | 3 of 5,000 | yes |
| Primary | Size | image | +1.56 [0.55, 2.10] | −0.32 [−0.72, −0.16] | +1.89 [1.17, 2.35] | 0 of 5,000 | yes |
| Primary | Lesion-skin contrast | image | +1.01 [−0.63, 1.93] | −1.07 [−1.88, −0.73] | +2.09 [1.00, 2.78] | 481 of 5,000 | no |
| Secondary | Asymmetry | tabular | −0.29 [−1.28, 0.72] | +1.36 [0.99, 1.91] | −1.65 [−2.60, −0.84] | 1,518 of 5,000 | no |
| Secondary | Border irregularity | tabular | −0.19 [−1.26, 0.86] | +1.51 [1.12, 2.09] | −1.71 [−2.73, −0.84] | 1,875 of 5,000 | no |
| Secondary | Asymmetry | image | −0.50 [−0.97, 0.62] | +1.09 [0.90, 1.68] | −1.59 [−2.20, −0.71] | 1,651 of 5,000 | no |
| Secondary | Border irregularity | image | −0.09 [−0.63, 1.04] | +1.18 [0.96, 1.80] | −1.27 [−1.93, −0.36] | 3,341 of 5,000 | no |

In Fig. 2, M0 is shown as circles and M2 as triangles on panel-specific scales. Its bars are test-patient bootstrap intervals conditional on the fitted models and therefore differ from the joint training-and-test bootstrap intervals of the primary inference in Table 1.

**Fig. 2.** Learned contrasts of M0 and M2 in the four learner families.

### B. Controlled selection of recorded negatives can induce the reversal

Table 2 summarizes the tests of alternative explanations, and Supplementary Section S6 reports each in full.

A size-matched control combined the same malignant lesions with 320 recorded negatives per replicate. For color variegation and size, in both families, the contrast was positive in 20 of 20 replicates when the recorded negatives were drawn at random. It was negative in 20 of 20 when they were verified ones. For color variegation it was also negative in 20 of 20 when they were flagged but never biopsied. This argues against training-set size as the explanation for these two concepts. For lesion-skin contrast with image features the random arm was itself close to zero, so the control does not separate size from selection there. Proposition 1 predicted the sign of the shift from the random arm in 20 of 20 cases, with mean absolute differences in size from 0.05 to 0.63.

A controlled selection dose changed only which recorded negatives entered training (Fig. 3(a)). In each of 20 replicates, a validation set was drawn. It was shared by all five doses, together with the malignant training lesions and the seed. Each recorded-negative training lesion received one uniform variable that decided its inclusion at every dose. Each replicate therefore fixes ω, and contrasts between doses are paired potential outcomes. The outcome is the learner's raw-logit contrast; Proposition 1 characterizes the corresponding Bayes-optimal training-distribution contrast. Platt calibration on the replicate-specific validation set, which is not selected by η, would target the unselected distribution instead. At the highest dose, its fitted slope was negative in 29 of 200 fits, so we report calibrated contrasts only in Supplementary Section S6.

The main result is the training-selection effect τ of Section III-B. With 320 selected recorded negatives, close to the training size of the biopsy-only regime, τ̂_k(1, −1) of the selected concept ranged from −4.72 to −2.59 with tabular features. With image features it ranged from −3.04 to −1.11, and every 95 percent interval lay below −0.99. These are bootstrap intervals over the 20 paired replicates, conditional on the cohort, patient split and test population. They average over the algorithmic randomness ω, including the validation-set draw used for early stopping, and do not reflect patient sampling. At this size, the mean contrast of the selected concept crossed zero in 6 of 6 combinations of concept and family. Within the same replicate, it changed sign in 120 of 120 replicates. The contrasts of the five concepts moved in the predicted direction in 29 of the 30 combinations of family, selected concept and responding concept.

Whether a contrast crosses zero also depends on its baseline value and on the dose range. With 3,000 selected recorded negatives, color variegation with image features had τ̂ = −1.79 [−2.05, −1.51], yet its mean contrast did not cross zero and changed sign in only 8 of 20 replicates. The raw-logit slopes pointed in the direction of the Bayes-optimal training-distribution shift. Their sizes were also close to it in this implementation, at 0.88 to 1.14 times the prediction with 320 recorded negatives and 0.78 to 0.93 times with 1,000 or 3,000. Raw finite-network logits need not equal the Bayes-optimal log-odds of the selected training distribution, so these ratios are descriptive and do not test the magnitude predicted by Proposition 1. In short, the paired experiment identifies the finite-pipeline effect τ, and Proposition 1 separately predicts the direction of the matching Bayes-optimal shift. Their numerical agreement is a diagnostic, not an identification result (Supplementary Section S6).

In Fig. 3(a), points are means of the raw logit contrast of color variegation over the 20 paired replicates with 320 selected recorded negatives, with 2.5 to 97.5 percentile ranges. Its dashed lines are the Bayes-optimal training-distribution shift given each learner's input, anchored at η = 0 and shown for direction only. For image features, they use a ridge estimate of the conditional concept mean. In Fig. 3(b), points are calibrated medians over 200 one-seed joint resamples with 95 percent percentile intervals, against the identity line.

**Fig. 3.** Controlled selection dose (a) and observed against predicted learner gap (b).

In Table 2, size-matched intervals are 2.5 and 97.5 percentiles over replicates. Dose intervals are the paired-replicate bootstrap intervals described above, and the ratios are descriptive. The other intervals come from one-seed joint bootstrap resamples of training and test patients.

**TABLE 2. Tests of alternative explanations for the learner reversal.**

| Check | Setting | Result |
| --- | --- | --- |
| Size-matched control, color variegation | tabular; recorded negatives from all / flagged / verified lesions | +1.22 [0.60, 2.14] / −0.79 [−1.21, −0.38] / −0.94 [−1.12, −0.70]; positive in 20 / 0 / 0 of 20 |
| Size-matched control, color variegation | image; same three sources | +0.59 [0.31, 0.90] / −0.82 [−1.05, −0.58] / −0.63 [−0.81, −0.40]; positive in 20 / 0 / 0 of 20 |
| Selection dose on training only; replicate-specific validation shared across doses | tabular; 320 selected recorded negatives; three primary concepts | τ̂(1, −1) −4.72 to −2.59; sign change in 60 of 60 replicates; descriptive slope ratio 0.99 to 1.14 |
| Selection dose on training only; replicate-specific validation shared across doses | image; 320 selected recorded negatives; three primary concepts, ridge conditional mean | τ̂(1, −1) −3.04 to −1.11; sign change in 60 of 60 replicates; descriptive slope ratio 0.88 to 1.10 |
| Selection dose, learning curve | tabular; color variegation | 1,000: τ̂ −3.95 [−4.17, −3.72], sign change in 20 of 20, mean crosses zero, ratio 0.93; 3,000: τ̂ −3.69 [−3.90, −3.47], sign change in 19 of 20, mean crosses zero, ratio 0.86 |
| Selection dose, learning curve | image; color variegation | 1,000: τ̂ −1.60 [−1.78, −1.44], sign change in 18 of 20, mean crosses zero, ratio 0.78; 3,000: τ̂ −1.79 [−2.05, −1.51], sign change in 8 of 20, mean does not cross zero, ratio 0.91 |
| Verified against random recorded negatives, predicted shift | tabular; five concepts | sign 5 of 5; calibration slope 1.38 |
| Verified against random recorded negatives, predicted shift | image; five concepts | sign 5 of 5; calibration slope 0.53 |
| Proposition 1, lesion-level gap on log ĝ | tabular; calibrated; least squares, then split-sample errors-in-variables diagnostic | 0.89 [0.67, 1.09]; 1.33 [1.01, 1.69] |
| Proposition 1, lesion-level gap on log ĝ | image; calibrated; least squares, then split-sample errors-in-variables diagnostic | 0.86 [0.66, 1.08]; 1.32 [0.92, 1.75] |
| Overlap sensitivity, color variegation, common tabular σ̂ | tabular; Δ_M0 against Δ_M2 | 1st percentile +0.35 against −1.04; 5th percentile −0.05 against −1.27; 10th percentile −0.13 against −1.31 |
| Overlap sensitivity, color variegation, common tabular σ̂ | image; Δ_M0 against Δ_M2 | 1st percentile +0.78 against −0.70; 5th percentile +0.13 against −0.81; 10th percentile +0.34 against −0.84 |
| PAD-UFES-20 learners | six symptoms / image color | 6 of 6 attenuated, P(signs differ) at most 0.04 / +2.06 [1.47, 2.84] to +0.70 [0.22, 1.23] |

### C. The learner gap follows the recorded-negative verification propensity

Proposition 1 is exact only for Bayes-optimal learners, so for trained networks we examine direction first. With a perceptron for ĝ, the predicted concept-level gap had the observed sign in at least 97.5 percent of joint resamples in 10 of 10 combinations of concept and family. With gradient boosting, it did so in 10 of 10. Magnitude depended on the propensity model, with calibration slopes of observed on predicted gap of 0.82 to 1.43 and 1.81 to 1.97. Lesion by lesion, the calibrated logit gap rose with log ĝ with slope 0.89 [0.67, 1.09] and 0.86 [0.66, 1.08]. Even where the identity holds exactly, in a semi-synthetic design, these slopes ranged from 0.83 to 1.35 across propensity models (Supplementary Section S2). Direction thus agreed across propensity models, while size agreed only for some. Fig. 3(b) plots the observed against the predicted concept-level gap. Agreement between observed and predicted gaps also reflects approximation error, calibration, estimation of *g* and optimization. It is a diagnostic, not a test of the identity; the symbolic and unit tests verify the identity itself (Supplementary Section S2).

Verification of recorded negatives depended on appearance mainly within patients. A one standard deviation increase in color variegation relative to the patient's own lesions raised the log odds of verification by 0.93 (SE 0.04). For the patient's mean, the increase was 0.10 (SE 0.15). This describes a verification pattern, consistent with clinicians biopsying a patient's atypical-looking lesions, and does not identify the clinical mechanism. The verified-benign contrast log *B*_V for color variegation was positive in all 5 acquisition sites where it could be estimated. Its pooled value was 1.70 [1.17, 2.23], with substantial heterogeneity (I² = 0.74). Two further sites were not estimable (Supplementary Section S6). Removing any single site left both signs unchanged.

### D. Overlap

M2 learns from verified lesions but is evaluated on the whole test population, which could force extrapolation. We therefore restricted the test population to lesions whose estimated verification propensity σ̂ reached the 1st, 5th or 10th percentile among verified training lesions. The theory-matched σ̂ is fitted on each learner's own input, as in Proposition 2, and kept 19 to 73 percent of test lesions. A second σ̂ on the tabular features gives a common support that admits joint intervals. The M2 contrast stayed below zero throughout. Its joint intervals were below zero in 18 of 18 combinations of primary comparison and threshold. Size with image features kept 95 percent intervals on opposite sides of zero at every threshold under every estimate of σ. The M0 contrast of color variegation fell instead. With the input-matched σ̂, it was between −0.82 and −0.29 at the 5th and 10th thresholds. Under the common support, the joint interval of M0 reached zero in 15 of 18 combinations. Standardizing the supported lesions on the five measured concepts did not restore it. It ranged from −0.82 to +1.00, against +1.02 and +1.64 on the full population, while the standardized M2 contrast stayed between −1.25 and −0.65. In better-supported regions, the loss of the color learner reversal thus coincided with a smaller M0 contrast among lesions more likely to be verified. The restriction changes the evaluation population and does not identify why. The full-population learner reversal describes the held-out test population, and the thresholds are a sensitivity analysis, not a proof of positivity (Supplementary Section S6).

### E. External consistency check on PAD-UFES-20

On PAD-UFES-20, every image recorded as malignant was biopsied, so *A* = 1 under the recorded-diagnosis operationalization and Lemma 1 then reduces to log OR_{D|S} = θ − log *B*. Verification attenuated ten of eleven association features, amplified one and reversed none (Fig. 4). The learner contrasts showed the same qualitative attenuation. With patients split 60/20/20, M2 trained on the 814 biopsied training images had a smaller contrast than M0 for all six symptoms. For image color variegation, the contrast fell from +2.06 [1.47, 2.84] to +0.70 [0.22, 1.23]. The cohorts differ in modality, population, prevalence and workflow. PAD-UFES-20 illustrates the attenuating regime of Lemma 1 and validates neither the mechanism on ISIC-2024 nor Lemma 1 for latent disease. If some clinically diagnosed lesions were malignant, *A* would fall below one by an amount the data do not identify.

In Fig. 4, the PAD-UFES-20 features are placed at *A* = 1 under the recorded-diagnosis operationalization. The three primary ISIC-2024 concepts are reference positions under the assumption *A* = 1, with segments for *A* in [0.5, 2]. The segments are sensitivity paths, not confidence intervals, and the two cohorts are not matched counterfactuals of each other.

**Fig. 4.** ISIC-2024 and PAD-UFES-20 in the plane of Lemma 1.

### F. Learner-scale disease target: an assumption-indexed sensitivity analysis

With the primary plug-in, the Platt-calibrated M0, the lower endpoint for color variegation became positive at a floor of 0.38 for the tabular-input target and 0.49 for the image-feature target. Resampling training, validation and test patients together, the bootstrap stability threshold was 0.82 for both targets, with Monte Carlo ranges of [0.79, 0.85] and [0.81, 0.86], for this specification of *q* only. It is not a confidence bound for the floor or for ψ. At a floor of 0.8, ψ_L was positive and Δ_M2 negative in the same replicate in 940 and 931 of 1,000 replicates. For the image-feature target, these floors come from the outer stratum formula. An exploratory estimated-weight approximation to the sharp set gave narrower bounds. Its finite-sample validity was not established, so it is not used for certification (Supplementary Section S7).

The plug-in tipping floor moved with the estimator of *q* far more than with sampling (Table 3). For the tabular-input target, the point floor was 0.28 with isotonic calibration and 0.75 with gradient boosting, against 0.38 with Platt calibration. For the image-feature target, it was 0.58 and 0.84 against 0.49. With gradient boosting for the image-feature target, no floor up to one made ψ_L positive in 95 percent of replicates, so nuisance uncertainty cannot be summarized by one threshold. Gradient boosting had the poorest calibration of tertile-level event counts among the three and the most conservative tipping behavior (Supplementary Section S7). We report it as a sensitivity specification rather than privileging any nuisance model. Calibration approximately matches aggregate event risk, a sum of *q*. It does not match the mean of logit *q*, which is steep near zero. In the lower tertile of color variegation with tabular features, Platt and isotonic calibration expected 26.9 and 25.3 events against 25 observed. Yet their mean logit *q* differed by 1.48 (Supplementary Section S7). In the rare-event tail, the logit-scale target is therefore empirically unstable to the estimator of *q*.

In Table 3, the point floor uses the seed-averaged curve of ψ_L, and the Monte Carlo range of the stability threshold comes from 500 resamples of the replicates. Ranges across estimators of *q* summarize specification sensitivity and are not statistical confidence intervals. Joint counts are resampling frequencies under the stated plug-in procedure, not probabilities that a disease-relative reversal holds. Log-loss is on the test population for the original fits.

**TABLE 3. Plug-in tipping floor for color variegation under three estimates of *q*.**

| Features | Estimate of *q* | Plug-in point tipping floor | Bootstrap stability threshold [Monte Carlo range] | Joint count at 0.8 | Test log-loss |
| --- | --- | ---: | --- | ---: | ---: |
| tabular | perceptron, Platt | 0.38 | 0.82 [0.79, 0.85] | 940 of 1,000 | 0.0058 |
| tabular | perceptron, isotonic | 0.28 | 0.80 [0.76, 0.82] | 956 of 1,000 | 0.0060 |
| tabular | gradient boosting, Platt | 0.75 | 0.97 [0.96, 0.97] | 174 of 1,000 | 0.0074 |
| image | perceptron, Platt | 0.49 | 0.82 [0.81, 0.86] | 931 of 1,000 | 0.0071 |
| image | perceptron, isotonic | 0.58 | 0.85 [0.84, 0.87] | 881 of 1,000 | 0.0071 |
| image | gradient boosting, Platt | 0.84 | none | 5 of 1,000 | 0.0075 |


Proposition 2 identifies a population set, but its plug-in estimate need not be reliable. An overstated floor certified the wrong sign in 20 percent of adversarial synthetic settings even with true nuisances. In semi-synthetic designs on the ISIC-2024 inputs, disease risk was built from the perceptron or from gradient boosting. No false-sign certification occurred at the true floor. However, the plug-in sets contained the true target in only 62 to 96 percent of settings. Their lower endpoints were mostly below the oracle ones (mean differences −0.22 to +0.01), so they have no demonstrated nominal coverage. In one design, the floor failed only among the 30 percent of lesions with the lowest verification propensity. There, even the oracle set contained the true target in only 50 to 75 percent of settings (Supplementary Section S2).

### G. Separate post-inspection marginal sensitivity analysis

This analysis is neither an estimator nor a check of ψ, and it does not validate Proposition 2. It uses the whole cohort rather than the learner's evaluation population. It asks a different question: whether the disease association between the outer tertiles has the sign opposite to the verified-only association. Because RR_Y = RR_D·*A* exactly (Supplementary Section S1), the disease risk ratio exceeds one when *A* < RR_Y. A floor π₀¹ ≥ *s*_min on the average malignant verification in the lower tertile alone gives *A* ≤ 1/*s*_min. The analysis uses counts only. For color variegation RR_Y was 1.97 [1.48, 2.59], whereas the verified-only log odds ratio was −0.93 [−1.26, −0.57]. The marginal association reversal held in 1,913 of 2,000 patient-cluster resamples at a lower-tertile floor of 0.65. This is the 95 percent resampling stability threshold on a grid of step 0.05. On the continuous *A* scale, it corresponds to about 1.55. This threshold is not a confidence bound for the floor, and nothing controls error for the choice of concept and analysis, both made after the primary results had been seen. On the test population alone the threshold was 0.95. Under a common lower-tertile floor assumed to hold within every observed site, the site-standardized whole-cohort threshold was 0.62. Leaving out one site at a time, it ranged from 0.40 to 0.76. This is a mixture estimand for the observed sites, not evidence of a reversal that holds in each site. Site-specific RR_Y ranged from 1.26 to 10.08 across the six estimable sites. The two sites with the most malignant lesions had 1.26 and 1.31, implying point floors of 0.79 and 0.76. One site could not be estimated, and in one site the verified-only log odds ratio was positive (Supplementary Section S7). Whether such a floor is plausible depends on how appearance dependence is transported from benign to malignant verification, which the data cannot identify. The crude whole-cohort benign ratio *B*_V was 4.96 [3.94, 6.25]. On a ratio scale this would exceed every *A* compatible with π₀¹ above 0.20. With the same logit gradient, log *B*_V, *A* stays below 1.55 once π₀¹ ≥ 0.56 (Supplementary Fig. S3). Both are benchmarks, not estimates of *A*.

## V. Discussion

**What the evidence supports.** A controlled change in training selection moved what a classifier learned in the predicted direction. With 320 selected recorded negatives, the learned contrast also crossed zero. The observed learner reversal between M0 and M2 persisted when training-set size was matched. The gap between the two learners had the sign predicted from the recorded-negative verification propensity for every concept. Still, the reversal itself is not a randomized contrast. Relative to disease, only the separate whole-cohort marginal analysis indicated a reversal. It did so under a lower-tertile floor on malignant verification that these data cannot check. The result is a mixture over the observed acquisition sites, and the present analysis does not identify or validate transport to a new site.

**What the evidence does not support.** We do not claim that verification always reverses learned relationships; in the phase diagram no population reversed when benign verification did not depend on appearance. We do not claim a Bonferroni sign-separated learner reversal for every concept, or a learner reversal of color variegation within better-supported regions, where the M0 contrast fell and could turn negative. We do not claim a learner-scale disease-relative reversal. That conclusion would need two things. The first is a floor on malignant verification over the relevant part of the input space, including where verification is rarest; this cohort cannot check it. The second is a specification of the recorded-label risk, which moved the plug-in tipping floor. The plug-in set often missed the true target in simulation even under a valid floor. Using the bound in practice needs better estimation of the rare-event tail, or more events, and not only a defensible floor. The controlled experiment intervenes on training selection, not on clinical biopsy decisions. Localization diagnostics on the fine-tuned encoders found no evidence of a general representation-level reversal (Supplementary Section S5).

**Threats to validity.** *Concepts.* The concepts are measurements from total-body photography, and the results apply to them as measured. *Inference.* The test population holds 209 patients, and site-level heterogeneity in verification was substantial. Fine-tuning under M0 was unstable across seeds, which is why that family is secondary. *Estimation.* The width of the identified set under the floor does not shrink with more data. A floor that fails only where verification is rarest lowered even the oracle containment (Section IV-F). The two-floor analysis of Supplementary Section S7 parameterizes this assumption more flexibly without testing it.

**Implications.** Concept attributions of a classifier trained on biopsied lesions should not be read as statements about disease. Marginal concept contrasts should not be read as isolated concept effects. Comparisons across data sets should record how each was verified. The selection propensity can be estimated only if a data set records the full cohort of candidate lesions with an indicator of biopsy. Recording the clinical reasons for biopsy could support point identification of disease risk. This would require conditional exchangeability of verification with respect to disease given the recorded information, positivity of verification, and correct measurement of that information.

## VI. Conclusion

We asked whether selective verification changes what a classifier learns. The training-selection offset is exact for Bayes-optimal learners and observable for biopsy-based training. In trained learners its predicted direction held, while agreement in magnitude was only descriptive. On ISIC-2024, the two regimes gave color variegation and size contrasts of opposite sign on the full test population. Size was the most stable across resampling schemes. Color was stable across concept definitions but not within better-supported regions. Because the controlled change in training selection was applied on shared algorithmic randomness, its effect is identified by design, conditional on the cohort. Each selected contrast moved in the predicted direction and, with 320 selected recorded negatives, crossed zero. Relative to latent disease, Proposition 2 identifies a learner-scale set only under a floor on malignant verification. On ISIC-2024, the floor at which its plug-in certifies the sign moved with the estimator of recorded-label risk. So the cohort does not establish a learner-scale disease-relative reversal. The separate post-inspection marginal analysis reached a disease association of color variegation opposite to its verified-only association, and only by assuming a floor on malignant verification. Changing verification-like training selection by experiment altered the learned concept contrast. Training selection can therefore shape the statistical relationships a classifier learns, not only filter how it is evaluated.

## Statements

**Data availability.** Both data sets are public. ISIC-2024 is distributed through the ISIC Archive [1] and PAD-UFES-20 through Mendeley Data [9]. No new patient data were collected.

**Code availability.** The scripts that produce every number, table and figure in this paper are available at https://github.com/dqtoan87/CausalReversal, together with the symbolic checks, the unit tests and the analysis lock. The version reported here is release v1.6-submission. Supplementary Section S8 maps each result to its script.

**Ethics.** This is a secondary analysis of two de-identified public data sets released under the approvals documented by their providers. No further approval was required.

**Author contributions.** Quang Toan Dao designed the study, derived the theory, implemented the pipeline, ran the analyses and drafted the manuscript. Viet Anh Nguyen supervised the study and revised the manuscript.

**Funding.** This work received no specific grant from any funding agency.

**Conflicts of interest.** The authors declare no competing interests.

## References

[1] Kurtansky, N. R., D'Alessandro, B. M., Gillis, M. C., Betz-Stablein, B., Cerminara, S. E., Garcia, R., … & Rotemberg, V. (2024). The SLICE-3D dataset: 400,000 skin lesion image crops extracted from 3D TBP for skin cancer detection. *Scientific Data*, *11*(1), 884.

[2] Esteva, A., Kuprel, B., Novoa, R. A., Ko, J., Swetter, S. M., Blau, H. M., & Thrun, S. (2017). Dermatologist-level classification of skin cancer with deep neural networks. *Nature*, *542*(7639), 115–118.

[3] Begg, C. B., & Greenes, R. A. (1983). Assessment of diagnostic tests when disease verification is subject to selection bias. *Biometrics*, *39*(1), 207–215.

[4] Hernán, M. A., Hernández-Díaz, S., & Robins, J. M. (2004). A structural approach to selection bias. *Epidemiology*, *15*(5), 615–625.

[5] Lakkaraju, H., Kleinberg, J., Leskovec, J., Ludwig, J., & Mullainathan, S. (2017). The selective labels problem: Evaluating algorithmic predictions in the presence of unobservables. In *Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 275–284).

[6] Berkson, J. (1946). Limitations of the application of fourfold table analysis to hospital data. *Biometrics Bulletin*, *2*(3), 47–53.

[7] Prentice, R. L., & Pyke, R. (1979). Logistic disease incidence models and case-control studies. *Biometrika*, *66*(3), 403–411.

[8] Fithian, W., & Hastie, T. (2014). Local case-control sampling: Efficient subsampling in imbalanced data sets. *The Annals of Statistics*, *42*(5), 1693–1724.

[9] Pacheco, A. G. C., Lima, G. R., Salomão, A. S., Krohling, B., Biral, I. P., de Angelo, G. G., … & de Barros, L. F. (2020). PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones. *Data in Brief*, *32*, 106221.

[10] Kleinbaum, D. G., Kupper, L. L., & Morgenstern, H. (1982). *Epidemiologic Research: Principles and Quantitative Methods*. Lifetime Learning Publications.

[11] Greenland, S. (1996). Basic methods for sensitivity analysis of biases. *International Journal of Epidemiology*, *25*(6), 1107–1116.

[12] Lash, T. L., Fox, M. P., & Fink, A. K. (2009). *Applying Quantitative Bias Analysis to Epidemiologic Data*. Springer.

[13] Nguyen, T. Q., Dafoe, A., & Ogburn, E. L. (2019). The magnitude and direction of collider bias for binary variables. *Epidemiologic Methods*, *8*(1), 20170013.

[14] Banack, H. R., & Kaufman, J. S. (2014). The obesity paradox: Understanding the effect of obesity on mortality among individuals with cardiovascular disease. *Preventive Medicine*, *62*, 96–102.

[15] Griffith, G. J., Morris, T. T., Tudball, M. J., Herbert, A., Mancano, G., Pike, L., … & Hemani, G. (2020). Collider bias undermines our understanding of COVID-19 disease risk and severity. *Nature Communications*, *11*(1), 5749.

[16] Kleinberg, J., Lakkaraju, H., Leskovec, J., Ludwig, J., & Mullainathan, S. (2018). Human decisions and machine predictions. *The Quarterly Journal of Economics*, *133*(1), 237–293.

[17] De-Arteaga, M., Dubrawski, A., & Chouldechova, A. (2018). Learning under selective labels in the presence of expert consistency. Presented at the Workshop on Fairness, Accountability, and Transparency in Machine Learning (FAT/ML). *arXiv preprint* arXiv:1807.00905.

[18] Mullainathan, S., & Obermeyer, Z. (2022). Diagnosing physician error: A machine learning approach to low-value health care. *The Quarterly Journal of Economics*, *137*(2), 679–727.

[19] Balachandar, S., Garg, N., & Pierson, E. (2024). Domain constraints improve risk prediction when outcome data is missing. In *International Conference on Learning Representations (ICLR)*.

[20] Rambachan, A., Coston, A., & Kennedy, E. H. (2022). Robust design and evaluation of predictive algorithms under unobserved confounding. *arXiv preprint* arXiv:2212.09844.

[21] Coston, A., Rambachan, A., & Chouldechova, A. (2021). Characterizing fairness over the set of good models under selective labels. In *Proceedings of the 38th International Conference on Machine Learning*, PMLR 139 (pp. 2144–2155).

[22] Elkan, C., & Noto, K. (2008). Learning classifiers from only positive and unlabeled data. In *Proceedings of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 213–220).

[23] Bekker, J., & Davis, J. (2020). Learning from positive and unlabeled data: A survey. *Machine Learning*, *109*(4), 719–760.

[24] Bekker, J., Robberechts, P., & Davis, J. (2019). Beyond the selected completely at random assumption for learning from positive and unlabeled data. In *Machine Learning and Knowledge Discovery in Databases: ECML PKDD 2019*, LNCS 11907 (pp. 71–85). Springer.

[25] Alonzo, T. A., & Pepe, M. S. (2005). Assessing accuracy of a continuous screening test in the presence of verification bias. *Journal of the Royal Statistical Society: Series C*, *54*(1), 173–190.

[26] Bareinboim, E., & Pearl, J. (2012). Controlling selection bias in causal inference. In *Proceedings of the 15th International Conference on Artificial Intelligence and Statistics* (pp. 100–108).

[27] Mohan, K., & Pearl, J. (2021). Graphical models for processing missing data. *Journal of the American Statistical Association*, *116*(534), 1023–1037.

[28] Zadrozny, B. (2004). Learning and evaluating classifiers under sample selection bias. In *Proceedings of the 21st International Conference on Machine Learning* (p. 114).

[29] Heckman, J. J. (1979). Sample selection bias as a specification error. *Econometrica*, *47*(1), 153–161.

[30] Winkler, J. K., Fink, C., Toberer, F., Enk, A., Deinlein, T., Hofmann-Wellenhof, R., … & Haenssle, H. A. (2019). Association between surgical skin markings in dermoscopic images and diagnostic performance of a deep learning convolutional neural network for melanoma recognition. *JAMA Dermatology*, *155*(10), 1135–1141.

[31] Oakden-Rayner, L., Dunnmon, J., Carneiro, G., & Ré, C. (2020). Hidden stratification causes clinically meaningful failures in machine learning for medical imaging. In *Proceedings of the ACM Conference on Health, Inference, and Learning* (pp. 151–159).

[32] Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*, *2*(11), 665–673.

[33] Castro, D. C., Walker, I., & Glocker, B. (2020). Causality matters in medical imaging. *Nature Communications*, *11*(1), 3673.

[34] Manski, C. F. (2003). *Partial Identification of Probability Distributions*. Springer.

[35] Tamer, E. (2010). Partial identification in econometrics. *Annual Review of Economics*, *2*, 167–195.

[36] Kallus, N., & Zhou, A. (2018). Confounding-robust policy improvement. In *Advances in Neural Information Processing Systems*, *31*, 9269–9279.

[37] Alain, G., & Bengio, Y. (2017). Understanding intermediate layers using linear classifier probes. In *International Conference on Learning Representations, Workshop Track*.

[38] Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of neural network representations revisited. In *Proceedings of the 36th International Conference on Machine Learning* (pp. 3519–3529).

[39] Platt, J. C. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In A. J. Smola, P. Bartlett, B. Schölkopf, & D. Schuurmans (Eds.), *Advances in Large Margin Classifiers* (pp. 61–74). MIT Press.

[40] Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. In *Proceedings of the 8th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 694–699).

[41] Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics*, *29*(5), 1189–1232.

[42] Nachbar, F., Stolz, W., Merkle, T., Cognetta, A. B., Vogt, T., Landthaler, M., … & Plewig, G. (1994). The ABCD rule of dermatoscopy: High prospective value in the diagnosis of doubtful melanocytic skin lesions. *Journal of the American Academy of Dermatology*, *30*(4), 551–559.

[43] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 770–778).

[44] Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.

