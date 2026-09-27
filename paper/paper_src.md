# Causal Analysis of Learning Reversal under Selective Verification in Skin Cancer Classification

**Quang Toan Dao**¹·² (dqtoan@ioit.ac.vn), **Viet Anh Nguyen**¹ (anhnv@ioit.ac.vn)

¹ Institute of Information Technology, Vietnam Academy of Science and Technology, Ha Noi, Viet Nam

² Graduate University of Science and Technology, Vietnam Academy of Science and Technology, Ha Noi, Viet Nam

## Abstract

In skin cancer cohorts, disease is confirmed only for biopsied lesions, while unverified lesions enter training as recorded negatives. We ask whether selecting training data on verification can reverse the concept contrast a classifier learns, and what is identified about disease. When training lesions are selected on the recorded label and the learner's input, the Bayes-optimal logit shifts by the log selection probability; for biopsy-only training this is the observable log verification propensity among recorded negatives. On 401,059 ISIC-2024 lesions, learners trained on all and on biopsied lesions gave opposite-sign point estimates for color variegation and size. {{N:abs_inference}} The color and size reversals persisted after matching training-set size. In a post-inspection experiment, concept-specific doses that changed only which recorded negatives entered training shifted each primary contrast in the predicted direction and, at the training size of the biopsy-only regime, across zero. For latent disease, a floor on malignant verification yields a population identified set, sharp when the nuisance functions are known; image features use a conservative outer bound. On ISIC-2024, learner-scale sign certification depended on the estimator of recorded-label risk, and a separate post-inspection marginal analysis implied a disease association opposite to the verified-only one only under assumptions on malignant verification. Experimentally changing verification-analogous training selection can alter the concept contrast a classifier learns.

**Keywords:** causal inference, selection bias, selective labels, case-control sampling, partial identification, verification bias, skin lesion classification

## I. Introduction

Most labels in medical imaging are verified selectively. A lesion receives a histopathologic diagnosis only if a clinician decided to biopsy it. In ISIC-2024, lesions never linked to a pathology report and not tagged as biopsied were recorded as benign [@kurtansky]. Only 1,068 of 401,059 lesions carry a tissue diagnosis. Classifiers trained on such data are reported to rival dermatologists [@esteva], yet their labels reflect disease and clinical work-up together.

Selection is usually treated as a problem of estimation or evaluation: verification bias corrupts sensitivity and specificity [@begg], conditioning on a selection node opens paths that no adjustment set closes [@hernan], and selective labels complicate algorithm evaluation [@lakkaraju]. Less studied is what verification does to the relationship a model learns between a visual feature and disease. A feature can accompany disease and also prompt the decision to biopsy. Training on verified lesions then conditions on that decision [@berkson].

We ask two questions. First, does the selection that produced a training set change the sign of the concept contrast a classifier learns? Second, when the learned sign differs between training regimes, which learner is reversed relative to latent disease, and under which assumption can that be identified? We do not estimate the effect of intervening on a concept.

We make three contributions.

**First, an observable offset for what selected training data teach a learner.** Selecting training data on the recorded label and the input offsets the Bayes-optimal logit by the log selection probability, which for biopsy-based training is the log verification propensity among recorded negatives. The identity belongs to the family of case-control and local case-control offsets [@prentice; @fithian] and characterizes the Bayes-optimal shift exactly. What we add is its use for selective clinical labels: for trained learners it motivates an observable diagnostic, concept by concept, of the gap between learners trained on all and on verified lesions and of the response to a controlled change in training selection, whose direction we test.

**Second, evidence that selection on verification can reverse the contrast a classifier learns.** On ISIC-2024, learners trained under the two regimes learn opposite signs for color variegation and size on the full patient-disjoint test population. The strongest multiplicity-adjusted evidence for the marginal tertile contrast is for size with image features; color variegation is the concept whose reversal held under every concept definition, but it does not carry over to regions with stronger verified-training support. The color and size reversals survive a size-matched control. A controlled selection experiment, designed after the primary learner reversal had been observed and an analogue of selective verification rather than an intervention on clinical biopsy decisions, identifies the effect of the selection rule on the learned contrast: each concept-specific dose shifted the corresponding contrast in the direction the offset predicts, and across zero at the training size of the biopsy-only regime.

**Third, what the verification process leaves identified about disease.** We define the disease target on the same functional and population as the learned contrast and derive its population identified set under a floor on malignant verification. The set is sharp when the nuisance functions are known; a stratum-based outer set, used for image features, needs no estimate of how the input predicts the stratum. For the marginal association the condition reduces to a bound on one verification ratio. On ISIC-2024, analyzed after the primary results had been seen, the framework shows why disease-relative sign claims remain sensitive to the assumed floor and to the estimator of recorded-label risk, rather than establishing them. PAD-UFES-20, where every recorded malignancy is biopsied [@pacheco], serves as an external consistency check on the attenuating side of the boundary.

**How the parts fit.** The three parts have different evidential status, and separating them is the causal contribution rather than any single identity. The selection rule for training data is an intervention we control, so its effect on the learned contrast, conditional on the cohort, is identified by design. The observed verification process yields, through a classical sampling identity, a diagnostic for the shift between learners trained on all and on verified lesions. Latent disease is not point identified from these data, so statements about it are sets indexed by an explicit floor on malignant verification.

## II. Related Work

**Selection bias and its direction.** Epidemiology expresses selection bias as a multiplicative factor on the odds ratio, built from the selection probabilities of the exposure-by-outcome table [@kleinbaum]. Quantitative bias analysis turns that factor into sensitivity parameters [@greenland; @lash]. Berkson described selection on a common effect [@berkson]. The direction of collider bias between binary variables has been characterized in closed form [@nguyen]. Reversal by selection has been analyzed for the obesity paradox [@banack], and collider bias affects modern cohorts [@griffith]. Our Lemma 1 restates these results with verification ratios, and we do not claim it as new.

**Case-control and subsampling offsets.** Case-control sampling that depends on the outcome alone leaves logistic slopes unchanged and moves the intercept [@prentice]. Local case-control sampling selects on the outcome and the covariates and corrects the fit by a known offset [@fithian]. Our Proposition 1 is an identity of this family, written for training data selected by clinical verification.

**Selective labels.** When outcomes are observed only after a human decision, evaluation and learning depend on that decision [@lakkaraju; @kleinberg]. Expert consistency [@dearteaga] and selective testing [@mullainathan] have been used to learn under such selection. Risk prediction under historical testing can be improved with domain constraints such as known prevalence [@balachandar]. Bounds on predictive performance under selectively observed outcomes and unobserved confounding use debiased nuisance estimation [@rambachan], and fairness has been characterized over the set of good models under selective labels [@coston]. We do not propose a new bounding framework. We bound a learner-matched logit contrast of disease and connect it to an observable learner reversal.

**Positive-unlabeled learning.** Writing *Y* = *D*·*S* makes every unverified malignancy an unlabeled positive, the structure of positive-unlabeled learning [@elkan; @bekker], including feature-dependent labeling propensities [@bekkersar]. We use its decomposition of *q* to bound the disease contrast when the labeling propensity is unknown.

**Verification bias, missing data and shortcuts.** Verification bias in diagnostic accuracy [@begg] can be corrected under ignorable verification [@alonzo]. Graphical criteria address recovery from selection [@bareinboim] and missing data [@mohan], and learning under feature-dependent [@zadrozny] and outcome-dependent selection [@heckman] is classical. Dermatology classifiers exploit skin markings [@winkler] and hidden strata [@oakden], shortcut learning is general [@geirhos], and acquisition shapes what a model can learn [@castro].

**Partial identification and probing.** When a parameter is not point identified, one can report the set of values compatible with the data and stated assumptions [@manski; @tamer], and learn robustly over such sets [@kallus]. Linear probes [@alain] and centered kernel alignment [@kornblith] serve only in supplementary localization diagnostics.

## III. Method

### A. Causal model

Let *D* be the latent disease state and *X* the lesion's appearance. Concepts are deterministic readings of the image, *C* = *h*(*X*). Diagnosis is anticausal: disease shapes appearance, *D* → *X* → *C*. Let *U* be patient-level risk and surveillance intensity, *G* the acquisition site and *W* clinical information outside the image, such as reported change. A clinician flags lesions of interest (*F*) and biopsies some of them (*S*, with *S* ⊂ *F*). The recorded label is *Y* = *D*·*S*. Fig. 1(a) shows the graph. Verification depends on disease through *W* and through appearance, so disease is missing not at random among unverified lesions, whose recorded label is nonetheless zero. Lesions with *Y* = 0 are recorded negatives. Only those with *S* = 1 are verified benign.

**Fig. 1.** Causal model and reversal regions. (a) Disease shapes appearance, from which concepts are read. Verification depends on appearance, on clinical information outside the image, on the patient and on the site. The recorded label equals disease only where the lesion was verified. Dashed nodes are latent. (b) Regions of Lemma 1 in the plane of the disease contrast θ and the differential verification contrast δ. Reversal of the verified-only association relative to disease occurs above the diagonal for positive θ and below it for negative θ.

A learner does not see *X* itself but an input *X̃*: tabular measurements or image features. A training regime is a selection *R* of lesions into the training sample. M0 trains on all lesions (*R* ≡ 1). M2 trains on verified lesions (*R* = *S*). The identities below use only the variables they name, so they hold for any graph consistent with *Y* = *D*·*S*. The graph describes how the data arose and is not the basis of identification.

### B. Targets

Split concept *k* at its cohort tertiles into lower (*t* = 0) and upper (*t* = 1) strata. For a learner *m*, the **learned contrast** Δ_{m,k} is the mean logit among upper-stratum lesions minus that among lower-stratum lesions, on a fixed evaluation population, averaged over the randomness of training. It is a marginal tertile contrast: it includes differences in correlated concepts between the strata and is not the effect of concept *k* with the others held fixed. We drop *k* when the concept is clear. Write *p*(*x̃*) = P(*D* = 1 | *x̃*). The **disease target** on the same scale and population is

ψ_k = E[logit *p*(*X̃*) | *t* = 1] − E[logit *p*(*X̃*) | *t* = 0].

Because *p* depends on the input, ψ_k is specific to a learner family. The tabular-input and image-feature disease targets are different functionals of the same latent disease.

A **learner reversal** is a difference in sign between Δ_{M0,k} and Δ_{M2,k}. It is observable, but M0 and M2 differ in more than one respect, so it is not by itself the effect of a selection rule. That effect is defined next. A **learner-scale disease-relative reversal** is a difference in sign between Δ_{M2,k} and ψ_k. It can be established only as far as the identified set for ψ_k determines the sign of ψ_k. We also use the marginal disease contrast θ_k = log OR_D between the strata, the classical object of selection bias. A **marginal association reversal** is a positive θ_k with a negative verified-only association log OR_{D|S} (Lemma 1), or the reverse.

**Training-selection effect.** Let η index a rule, set by the analyst, for selecting recorded negatives into the training set only, with the validation set used for early stopping held fixed; calibration is not part of this outcome. Let ω collect the exogenous random numbers of the pipeline: the uniform variables that decide inclusion, the draw of malignant training lesions and the optimizer seed. The contrast of a single fit is then a potential outcome Δ_j(η, ω), and the training-selection effect on concept *j* is τ_j(η₁, η₀) = E_ω[Δ_j(η₁, ω) − Δ_j(η₀, ω)]. Conditional on the realized cohort, split, validation set and evaluation population, assignment of η identifies τ_j, averaged over the algorithmic randomness in ω, without assumptions about clinical verification. It is an effect of an experimental selection rule on what a learner learns, not an effect of a concept on disease or of a clinical biopsy policy, and not a superpopulation effect over repeated patient samples.

### C. The training-selection offset

**Proposition 1 (training-selection offset).** Write *r*_y(*x̃*) = P(*R* = 1 | *Y* = *y*, *x̃*) for the probability that a lesion with recorded label *y* and learner input *x̃* enters training. Then

logit P(*Y* = 1 | *x̃*, *R* = 1) = logit P(*Y* = 1 | *x̃*) + log *r*₁(*x̃*) − log *r*₀(*x̃*).

*Proof sketch.* By Bayes' rule, the odds of *Y* = 1 given *x̃* and *R* = 1 equal P(*Y* = 1 | *x̃*)·*r*₁(*x̃*) divided by P(*Y* = 0 | *x̃*)·*r*₀(*x̃*). Supplementary Section S1 gives the proof. The identity is stated on the learner's own input and uses no disease label.

When *R* is independent of *X̃* given *X* and *Y*, as when selection depends only on appearance and the recorded label, the selection probability averages over what the input does not show: *r*₀(*x̃*) = E[*r*₀(*X*) | *x̃*, *Y* = 0]. Two instances matter here. *Training on verified lesions.* With *R* = *S*, every malignant training lesion is verified, so *r*₁ = 1, and *r*₀(*x̃*) = *g*(*x̃*) = P(*S* = 1 | *x̃*, *Y* = 0), the verification propensity among recorded negatives. No further condition is needed. For the Bayes-optimal learners *f*₀ of M0 and *f*₂ of M2, logit *f*₀ − logit *f*₂ = log *g*, and therefore

Δ_{M0,k} − Δ_{M2,k} = E[log *g*(*X̃*) | *t* = 1] − E[log *g*(*X̃*) | *t* = 0].

A learner reversal occurs exactly when this contrast exceeds the M0 contrast in the same direction. *A controlled selection dose.* If each recorded-negative training lesion enters independently with probability π_η = min{1, κ_η exp(η*z*_k(*X*))}, where *z*_k is the standardized concept and κ_η fixes the expected number, and recorded positives are drawn uniformly, then *r*₀ is known by design. When *z*_k is part of the learner's input, *r*₀(*x̃*) = π_η(*z*_k), and without the cap the contrast of any concept *j* moves linearly, Δ_j(η) = Δ_j(0) − η·μ_{kj}, with μ_{kj} the difference in mean *z*_k between the strata of concept *j*, so that the Bayes-optimal training-distribution shift is −(η₁ − η₀)·μ_{kj}. When the input determines *z*_k only partly, as for image features, *r*₀(*x̃*) = E[π_η(*z*_k(*X*)) | *x̃*, *Y* = 0], which we approximate with a Gaussian model for *z*_k given *x̃* (Supplementary Section S1). Adding a constant to a logit cancels in Δ.

These statements are exact for the Bayes-optimal logit of the selected training distribution. A trained network adds approximation, optimization and early-stopping effects, so for the effect τ_j of a trained pipeline they predict direction, and agreement in magnitude is only a descriptive diagnostic (Section IV).

### D. Classical identities for the disease association

**Lemma 1 (verification selection factor).** For the strata of concept *k*, let π_t^d = P(*S* = 1 | *D* = *d*, *t*), *A* = π₁¹/π₀¹, *B* = π₁⁰/π₀⁰ and δ = log *B* − log *A*. Then log OR_{D|S} = θ − δ. For θ > 0, verification amplifies the association if δ < 0, attenuates it if 0 < δ < θ, and reverses it if δ > θ. For θ < 0 the regions mirror, and if θ = 0 verification alone creates an association equal to −δ.

Lemma 1 is the classical selection bias factor [@kleinbaum; @greenland] with verification ratios, and its reversal region is the known possibility of reversal by selection [@nguyen; @banack]. *B* is not observable, because π⁰ conditions on latent *D*. What is observed is *B*_V = *g*₁/*g*₀ with *g*_t = P(*S* = 1 | *Y* = 0, *t*). It differs from *B* by a factor close to one when disease is rare, and Supplementary Section S1 bounds it.

Supplementary Section S1 adds Lemma 2: when the log ratio of malignant to benign verification probabilities is linear in *x* with slope λ, a population-optimal verified-only logistic learner shifts the disease coefficients by λ.

### E. Identification of the disease target

Let *q*(*x̃*) = P(*Y* = 1 | *x̃*), σ(*x̃*) = P(*S* = 1 | *x̃*), the overall verification propensity, and *s*(*x̃*) = P(*S* = 1 | *D* = 1, *x̃*), the probability that a malignant lesion with input *x̃* is verified. Assume the pointwise floor *s*(*x̃*) ≥ *s*_min for every *x̃*. This is stronger than a floor on the share of malignant lesions that are verified, which can exceed *s*_min while the pointwise floor fails in part of the input space.

Write *a*_k(*x̃*) = P(*t* = 1 | *x̃*)/P(*t* = 1) − P(*t* = 0 | *x̃*)/P(*t* = 0), so that ψ_k = E[*a*_k(*X̃*) logit *p*(*X̃*)].

**Proposition 2 (identified set for the disease target).** Let *U*(*x̃*) = min{*q*(*x̃*)/*s*_min, *q*(*x̃*) + 1 − σ(*x̃*)}. Let *L*(*x̃*) = logit *q*(*x̃*) where *a*_k(*x̃*) ≥ 0 and *L*(*x̃*) = logit *U*(*x̃*) where *a*_k(*x̃*) < 0, and let *H* swap the two endpoints; where *a*_k = 0 either endpoint contributes zero. Then ψ_k lies in [ψ_L, ψ_U], with ψ_L = E[*a*_k(*X̃*)·*L*(*X̃*)] and ψ_U = E[*a*_k(*X̃*)·*H*(*X̃*)]. With the true *q*, σ and *a*_k this interval is sharp relative to the nonparametric model of the observed data that satisfies *Y* = *D*·*S* and the pointwise floor. Further restrictions on verification, such as structural, smoothness, monotonicity or cross-input constraints that the graph of Fig. 1(a) might motivate, could narrow it. The sign of ψ_k is identified as positive when ψ_L > 0 and as negative when ψ_U < 0.

**Corollary (stratum formula).** When *t* is a function of *x̃*, ψ_L = E[logit *q* | *t* = 1] − E[logit *U* | *t* = 0] and ψ_U = E[logit *U* | *t* = 1] − E[logit *q* | *t* = 0]. For any other input these stratum formulas are valid outer bounds.

*Proof sketch.* Pointwise, *p* = *q*/*s* lies between *q* and *U*, and both endpoints are attained. ψ_k increases in logit *p* where *a*_k > 0 and decreases where *a*_k < 0, so its extremes combine the endpoints by the sign of *a*_k. The stratum formulas assign endpoints by observed stratum instead of by the sign of *a*_k, which can only widen the interval. Supplementary Section S1 gives the proof. With the stratum formula and small *q*, ψ_L ≈ Δ_q + log *s*_min, where Δ_q = E[logit *q* | *t* = 1] − E[logit *q* | *t* = 0] is the contrast of a calibrated M0. The sign of ψ_k is then identified once *s*_min exceeds the population tipping floor *s*∗ ≈ exp(−Δ_q), and a biopsy-trained learner with a negative contrast is reversed relative to disease above that floor.

The floor enters ψ_L only through *U*, where *a*_k < 0. A positive sign therefore needs the floor only there, and with the stratum formula only on inputs that occur in the lower stratum. We state the floor for every *x̃* for simplicity. The tabular input contains the five concepts, so the stratum formula is sharp for the tabular-input target. Image features determine the concept only partly, so for the image-feature target we report the stratum formula as an outer set, which needs no estimate of *a*_k.

Proposition 2 assumes *q* and σ known. On ISIC-2024 we replace *q* by the calibrated M0 and σ by an out-of-fold model, which gives a plug-in estimate of the identified set. Sharpness does not carry over to it, and the bootstrap describes its sampling variability for a given nuisance specification, not whether that specification is correct. The primary plug-in uses the Platt calibration applied to M0 and M2. No estimator of *q* is privileged by theory, so we report three under one bootstrap. The smallest floor at which the plug-in lower endpoint is positive is the plug-in tipping floor, and the smallest floor at which it is positive in at least 95 percent of bootstrap replicates is the bootstrap stability threshold. Both describe the plug-in procedure under one nuisance specification; neither is a floor identified from the data.

### F. Data and learners

**Cohorts.** ISIC-2024 contains 401,059 lesion tiles from 1,042 patients [@kurtansky]. A lesion identifier marks 22,058 clinician-tagged lesions (*F*). Of these, 1,068 carry histopathology (*S*) and 393 are recorded malignant (*Y*). Every malignant label is histopathologically confirmed, and lesions never linked to pathology were recorded as benign [@kurtansky], so the release follows *Y* = *D*·*S*. PAD-UFES-20 contains 2,298 smartphone images of 1,641 lesions from 1,373 patients [@pacheco]. Basal cell carcinoma, squamous cell carcinoma and melanoma form the malignant class, and actinic keratosis, nevus and seborrheic keratosis the non-malignant class. Placing actinic keratosis, a premalignant lesion, in the non-malignant class is a binary mapping chosen for this analysis. Every image recorded as malignant was biopsied, and non-malignant lesions that were not biopsied carry a clinical diagnosis.

**Concepts.** We use five concepts from the total-body photography metadata: color variegation, size, lesion-skin contrast, asymmetry and border irregularity. The first three are primary, following clinical dermoscopy criteria [@nachbar].

**Split and regimes.** Patients were split 60/20/20 into train, validation and test, stratified by whether a patient had any malignant label. The test population holds 78,625 lesions from 209 patients. M2 is trained on 622 verified training lesions, of which 222 are malignant. Every malignant training lesion is verified, so the two regimes differ only in the size of the training set and in the source of its recorded negatives.

**Learner families.** (i) A two-layer perceptron on 47 tabular features. (ii) The same perceptron on frozen ResNet-50 ImageNet features [@he]. (iii) A linear probe on the frozen ResNet-50. (iv) ResNet-50 fine-tuned end to end. M0 and M2 share every setting except the training lesions. The two frozen-feature families are primary. Supplementary Section S5 lists every setting.

### G. Analyses and inference

**Primary inference.** The primary family is the three primary concepts in the two frozen-feature families, six comparisons. Point estimates average three seeds of the original fits. Each of {{N:B_primary}} joint bootstrap replicates per family resamples training patients, refits M0 and M2 with three seeds, calibrates them on the validation set, resamples test patients and averages the three seeds, so the bootstrap statistic is the point estimator itself. This inference conditions on the realized validation set. A second analysis with {{N:B19}} replicates per family and one seed per replicate targets the randomized one-seed training procedure and serves as a higher-resolution check. Two sensitivity analyses resample training, validation and test patients together, in {{N:B30}} replicates per family, or subsample patients without replacement. A learner reversal is called Bonferroni sign-separated when the Bonferroni-adjusted percentile intervals of Δ_M0 and Δ_M2, at level 1 − 0.05/6, lie on opposite sides of zero. The label refers to the tertile definition and, unless stated otherwise, to the fixed validation set. Secondary concepts are reported with 95 percent intervals and are not part of the multiplicity family. {{N:mc_rule}} Every other joint interval in the paper refits one seed per replicate and so targets the randomized one-seed training procedure: the support-restricted contrasts of Section IV-D and Table 2, the gap analyses of Section IV-C and Fig. 3(b), and Table 3. For the linear-probe and fine-tuned families, intervals resample test patients only and condition on the fitted models.

**Status of analyses.** An analysis lock fixed the primary and secondary outcomes after the primary comparison had first been inspected and before the analyses that test alternative explanations. It is a structured post-inspection plan, not a preregistration, and the primary analysis is partly data-informed. The Bonferroni adjustment controls multiplicity within this locked six-comparison family only, so it does not make the primary inference confirmatory. After the primary results had been inspected, the primary bootstrap statistic was changed from a one-seed refit to the three-seed average used by the point estimator, so that the resampled statistic matches the reported one, and the Monte Carlo resolution rule was added at the same stage. This changed the classification of one primary cell: color variegation with tabular features is reported as unresolved, with the one-seed and three-seed results both shown. The alternative concept definitions and learner families are robustness analyses of the same lesions and are not independent tests. Every other analysis in Sections IV-B to IV-G, apart from the size-matched control, was added after the primary results had been seen, including the design and outcome of the dose experiment and the marginal analysis of Section IV-G, and color variegation was chosen as the worked example at that stage.

**Checks and simulation.** Symbolic and unit-test checks verify the algebra and the code, and a phase diagram checks Lemmas 1 and 2 on 648 populations (Supplementary Section S2 and Section S3). Simulations test the plug-in set of Proposition 2 in synthetic populations, in an adversarial design with an overstated floor, and in semi-synthetic designs on the real ISIC-2024 inputs, including one where the floor fails where verification is rarest.

## IV. Results

### A. Learner reversal on ISIC-2024

Table 1 reports the learned contrasts. {{N:jb_sentence}} {{N:resample_sentence}} Fig. 2 shows all four families. The linear-probe and fine-tuned families also reversed color variegation, with no test bootstrap resample of the same sign, conditional on the fitted models. {{N:size_defs_sentence}} {{N:balanced_sentence}}

**TABLE 1. Learned contrasts on the ISIC-2024 test population, frozen-feature families. Point estimates average three seeds of the original fits. Intervals come from {{N:B_primary}} joint bootstrap resamples of training and test patients, each refitting both learners with three seeds and averaging them as for the point estimate. The primary family is the six cells in the upper block. Its intervals are Bonferroni-adjusted percentile intervals at level 1 − 0.05/6, and the Bonferroni sign-separated label is judged by them. {{N:table1_mc}} The lower block is secondary and shows 95 percent intervals. The same-sign count is descriptive and is not a test. Intervals condition on the fixed validation set.**

{{TABLE:main_reversal}}

**Fig. 2.** Learned contrasts under the two training regimes on one test population. Circles are M0, trained on all lesions with the recorded label. Triangles are M2, trained on verified lesions only. The four panels are the four learner families, and the horizontal scale differs between panels. Bars are test-only bootstrap intervals conditional on the fitted models, unlike the joint intervals of Table 1.

### B. Controlled selection of recorded negatives can induce the reversal

Table 2 summarizes the tests of alternative explanations, and Supplementary Section S6 reports each in full.

A size-matched control combined the same malignant lesions with 320 recorded negatives per replicate. For color variegation and size, in both families, the contrast was positive in 20 of 20 replicates when the recorded negatives were drawn at random and negative in 20 of 20 when they were verified ones. For color variegation it was also negative in 20 of 20 when they were flagged but never biopsied. This argues against training-set size as the explanation for these two concepts. For lesion-skin contrast with image features the random arm was itself close to zero, so the control does not separate size from selection there. {{N:arms_sentence}}

A controlled selection dose changed only which recorded negatives entered training (Fig. 3(a)). In each of 20 replicates the validation set, the malignant training lesions and the seed were shared by all five doses, and each recorded-negative training lesion received one uniform variable that decided its inclusion at every dose, so the replicate fixes ω and contrasts between doses are paired potential outcomes. The outcome is the learner's raw-logit contrast; Proposition 1 characterizes the corresponding Bayes-optimal training-distribution contrast. Platt calibration on the fixed validation set would target the unselected distribution instead, and at the highest dose its fitted slope was negative in 29 of 200 fits, so calibrated contrasts are reported only in Supplementary Section S6. {{N:dose_sentence}}

**Fig. 3.** Controlled selection and the learner gap. (a) Learned logit contrast of color variegation against the selection dose η on recorded-negative training lesions, each included with known probability, for tabular and image learners, with 320 selected recorded negatives and the validation set fixed across doses. Points are means over 20 paired replicates, with 2.5 to 97.5 percentile ranges. Dashed lines are the Bayes-optimal training-distribution shift of Proposition 1 given each learner's input, anchored at η = 0 and shown for direction, since raw finite-network logits need not equal the Bayes-optimal log-odds of the selected training distribution, with a ridge estimate of the conditional concept mean for image features. (b) Observed against predicted learner gap Δ_M0 − Δ_M2 for the five concepts and both frozen-feature families, after calibration. Points are medians over 200 joint resamples, and bars are 95 percent percentile intervals. The line is the identity.

**TABLE 2. Tests of alternative explanations. Size-matched intervals are 2.5 and 97.5 percentiles over replicates. Dose intervals are bootstrap intervals over the 20 paired replicates, conditional on the fixed cohort, and dose slope ratios are descriptive. Other intervals come from joint bootstrap resamples of training and test patients with one seed per replicate, which target the one-seed training procedure rather than the three-seed statistic of Table 1.**

| Check | Setting | Result |
| --- | --- | --- |
| Size-matched control, color variegation | tabular; recorded negatives from all / flagged / verified lesions | +1.22 [0.60, 2.14] / −0.79 [−1.21, −0.38] / −0.94 [−1.12, −0.70]; positive in 20 / 0 / 0 of 20 |
| Size-matched control, color variegation | image; same three sources | +0.59 [0.31, 0.90] / −0.82 [−1.05, −0.58] / −0.63 [−0.81, −0.40]; positive in 20 / 0 / 0 of 20 |
{{N:table2_rows}}

### C. The learner gap follows the recorded-negative verification propensity

{{N:bridge_sentence}} Agreement between observed and predicted gaps also reflects approximation error, calibration, estimation of *g* and optimization, so it is a diagnostic, not a test of the identity, which the symbolic and unit tests verify (Supplementary Section S2).

Verification of recorded negatives depended on appearance mainly within patients: a one standard deviation increase in color variegation relative to the patient's own lesions raised the log odds of verification by 0.93 (SE 0.04), against 0.10 (SE 0.15) for the patient's mean. This describes a verification pattern, consistent with clinicians biopsying a patient's atypical-looking lesions, and does not identify the clinical mechanism. {{N:site_sentence}}

### D. Overlap

{{N:overlap_paragraph}}

### E. External consistency check on PAD-UFES-20

On PAD-UFES-20, every image recorded as malignant was biopsied, so *A* = 1 under the recorded-diagnosis operationalization and Lemma 1 then reduces to log OR_{D|S} = θ − log *B*. Verification attenuated ten of eleven association features, amplified one and reversed none (Fig. 4). The learner contrasts showed the same qualitative attenuation. With patients split 60/20/20, M2 trained on the 814 biopsied training images had a smaller contrast than M0 for all six symptoms, and for image color variegation the contrast fell from +2.06 [1.47, 2.84] to +0.70 [0.22, 1.23]. The cohorts differ in modality, population, prevalence and workflow, so PAD-UFES-20 illustrates the attenuating regime of Lemma 1 and validates neither the mechanism on ISIC-2024 nor Lemma 1 for latent disease. If some clinically diagnosed lesions were malignant, *A* would fall below one by an amount the data do not identify.

**Fig. 4.** ISIC-2024 and PAD-UFES-20 in the plane of Lemma 1. Diamonds are the eleven PAD-UFES-20 association features, placed at *A* = 1 under the recorded-diagnosis operationalization; for latent disease *A* is not identified. Open circles are reference positions of the three primary ISIC-2024 concepts under the assumption *A* = 1. Segments show how each reference moves as *A* ranges over [0.5, 2]. They are sensitivity paths, not confidence intervals, and the two cohorts are not matched counterfactuals of each other.

### F. Learner-scale disease target: an assumption-indexed sensitivity analysis

{{N:psi_primary_sentence}} {{N:sharp_sentence}}

{{N:psi_est_sentence}} {{N:tail_sentence}}

**TABLE 3. Estimator-indexed sensitivity analysis: plug-in point tipping floor and bootstrap stability threshold for color variegation under three estimates of q, with training, validation and test patients resampled together in {{N:B30}} replicates per family. The point value uses the seed-averaged curve of ψ_L. The bootstrap stability threshold (Section III-E) has a Monte Carlo range from 500 resamples of the replicates and describes sampling variability for one estimate of q. The range across rows is a range across specifications, not a statistical uncertainty interval. The joint count of replicates with ψ_L > 0 at a floor of 0.8 and Δ_M2 < 0 is a resampling frequency, not a probability that a learner-scale disease-relative reversal holds. Log-loss is on the test population for the original fits.**

{{TABLE:psi_est}}


{{N:psi_sim_sentence}}

### G. Separate post-inspection marginal sensitivity analysis

{{N:rr_sentence}} {{N:site_sentence_marg}} {{N:benchmark_sentence}}

## V. Discussion

**What the evidence supports.** A controlled change in training selection shifted what a classifier learned in the predicted direction, and across zero at the training size of the biopsy-only regime. The observed learner reversal between M0 and M2 persisted when training-set size was matched, and the gap between the two learners had the sign predicted from the recorded-negative verification propensity for every concept, but the reversal itself is not a randomized contrast. Relative to disease, only the separate whole-cohort marginal analysis indicated a reversal, under a lower-tertile floor on malignant verification that these data cannot check; it is a mixture over the observed acquisition sites, and the present analysis does not identify or validate transport to a new site.

**What the evidence does not support.** We do not claim that verification always reverses learned relationships; in the phase diagram it did not when benign verification depended weakly on appearance. We do not claim a Bonferroni sign-separated learner reversal for every concept, or a learner reversal of color variegation within better-supported regions, where the M0 contrast fell and could turn negative. We do not claim a learner-scale disease-relative reversal. That conclusion needs a floor on malignant verification over the relevant part of the input space, including where verification is rarest, which this cohort cannot check, and a specification of the recorded-label risk, which moved the plug-in tipping floor. Because the plug-in set often missed the true target in simulation even under a valid floor, practical use of the bound needs better estimation of the rare-event tail, or more events, and not only a defensible floor. The controlled experiment intervenes on training selection, not on clinical biopsy decisions. Localization diagnostics on the fine-tuned encoders found no evidence of a general representation-level reversal (Supplementary Section S5).

**Threats to validity.** *Concepts.* The concepts are measurements from total-body photography, and the results apply to them as measured. *Inference.* The test population holds 209 patients, and site-level heterogeneity in verification was substantial. Fine-tuning under M0 was unstable across seeds, which is why that family is secondary. *Estimation.* The width of the identified set under the floor does not shrink with more data. {{N:local_sentence}}

**Implications.** Concept attributions of a classifier trained on biopsied lesions should not be read as statements about disease, nor marginal concept contrasts as isolated concept effects, and comparisons across data sets should record how each was verified. The selection propensity can be estimated only if a data set records the full cohort of candidate lesions with an indicator of biopsy. Recording the clinical reasons for biopsy could support point identification of disease risk, but only under conditional exchangeability of verification with respect to disease given the recorded information, positivity of verification, and correct measurement of that information.

## VI. Conclusion

We asked whether selective verification changes what a classifier learns. Selecting training data on the recorded label offsets the Bayes-optimal logit by the log selection probability, which is observable for biopsy-based training; for trained learners its predicted direction held, while agreement in magnitude was descriptive. On ISIC-2024, the two regimes gave color variegation and size contrasts of opposite sign on the full test population, size being the most stable across resampling schemes and color across concept definitions but not within better-supported regions. A controlled change in training selection, whose effect is identified by design conditional on the cohort, shifted each selected contrast in the predicted direction, across zero at the training size of the biopsy-only regime. Relative to latent disease, Proposition 2 identifies a learner-scale set only under a floor on malignant verification, and on ISIC-2024 the floor at which its plug-in certifies the sign moved with the estimator of recorded-label risk, so the cohort does not establish a learner-scale disease-relative reversal; a separate post-inspection marginal analysis implied a disease association of color variegation opposite to the verified-only one only under a floor assumption. Experimentally changing verification-analogous training selection altered the learned concept contrast, showing that training selection can shape learned statistical relationships rather than acting only as a filter on evaluation.

## Statements

**Data availability.** Both data sets are public. ISIC-2024 is distributed through the ISIC Archive [@kurtansky] and PAD-UFES-20 through Mendeley Data [@pacheco]. No new patient data were collected.

**Code availability.** The scripts that produce every number, table and figure in this paper, the symbolic checks, the unit tests and the analysis lock are available at https://github.com/dqtoan87/CausalReversal; the version reported here is release v1.1-submission. Supplementary Section S8 maps each result to its script.

**Ethics.** This is a secondary analysis of two de-identified public data sets released under the approvals documented by their providers. No further approval was required.

**Author contributions.** Quang Toan Dao designed the study, derived the theory, implemented the pipeline, ran the analyses and drafted the manuscript. Viet Anh Nguyen supervised the study and revised the manuscript.

**Funding.** This work received no specific grant from any funding agency.

**Conflicts of interest.** The authors declare no competing interests.

<!-- REFERENCES -->
