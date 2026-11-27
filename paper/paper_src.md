# Causal Analysis of Learning Reversal under Selective Verification in Skin Cancer Classification

**Quang Toan Dao**¹·² (dqtoan@ioit.ac.vn), **Viet Anh Nguyen**¹ (anhnv@ioit.ac.vn)

¹ Institute of Information Technology, Vietnam Academy of Science and Technology, Ha Noi, Viet Nam

² Graduate University of Science and Technology, Vietnam Academy of Science and Technology, Ha Noi, Viet Nam

## Abstract

In the ISIC-2024 skin cancer cohort, malignancy is recorded only for biopsied lesions, and unverified lesions enter training as recorded negatives. We ask whether selecting training data on verification can reverse the concept contrast a classifier learns, and what the data identify about disease. When training lesions are selected on the recorded label and the input, the Bayes-optimal logit of the selected data equals the full-cohort logit plus the log ratio of class-specific selection probabilities. Under biopsy-only selection, this offset is minus the observable log verification propensity among recorded negatives. On 401,059 lesions, learners trained on all and on biopsied lesions gave opposite-sign contrasts for color variegation and size. In a post-inspection family of six comparisons, Bonferroni-adjusted intervals separated the signs with image features, for size under every bootstrap scheme and for color variegation when the validation set was fixed. Both reversals persisted with matched training-set size. A post-inspection experiment changed only which recorded negatives entered training. Each designed rule moved its contrast as predicted, and the mean contrast changed sign across the dose range. For latent disease, a floor on malignant verification yields an identified set on the learner's scale, sharp with known nuisances. The floor needed for a disease-relative sign depended on the estimator of recorded-label risk. A crude pooled marginal analysis implied an opposite disease association for color variegation under an assumed floor, but a fixed six-site standardization did not support it. Changing which recorded negatives enter training can change the concept contrast a classifier learns.

**Keywords:** causal inference, selection bias, selective labels, case-control sampling, partial identification, verification bias, skin lesion classification

## I. Introduction

In skin cancer imaging, labels are often verified selectively. A lesion receives a histopathologic diagnosis only if a clinician decided to biopsy it. In ISIC-2024, lesions never linked to a pathology report and not tagged as biopsied were recorded as benign [@kurtansky]. Only 1,068 of 401,059 lesions carry a tissue diagnosis. Image classifiers can match dermatologists on biopsy-verified test images [@esteva]. In cohorts such as ISIC-2024, however, training labels reflect disease and clinical work-up together.

Selection is usually treated as a problem of estimation or evaluation. Verification bias corrupts sensitivity and specificity [@begg]. Conditioning on selection can open noncausal paths, and whether they can be blocked depends on the graph and on the measured variables [@hernan]. Selective labels complicate algorithm evaluation [@lakkaraju]. Less studied is what verification does to the relationship a model learns between a visual feature and disease. A feature can accompany disease and also prompt the decision to biopsy. Training on verified lesions then conditions on that decision [@berkson].

We ask two questions. First, does the selection that produced a training set change the sign of the concept contrast a classifier learns? Second, when the learned sign differs between training regimes, which learner is reversed relative to latent disease, and under which assumption can that be identified? We do not estimate the effect of intervening on a concept.

We make three contributions.

**First, an observable offset for what selected training data teach a learner.** When training data are selected on the recorded label and the input, the Bayes-optimal logit of the selected data equals the full-cohort logit plus the log ratio of the class-specific selection probabilities. Under biopsy-only selection, the offset is minus the log verification propensity among recorded negatives. The identity belongs to the family of case-control and local case-control offsets [@prentice; @fithian]. Its stratum contrast can be computed from verification records before a biopsy-trained learner is fitted. For Bayes-optimal learners it gives the gap between training on all and on verified lesions, and together with the full-cohort contrast it says whether they reverse. For trained networks it is a predicted direction that we check. It also predicts the direction in which a controlled change in training selection moves the learned contrast. We test both predictions.

**Second, evidence that selection on verification can reverse the contrast a classifier learns.** On ISIC-2024, learners trained under the two regimes learn opposite signs for color variegation and size on the full patient-disjoint test population. The strongest multiplicity-adjusted evidence is for size with image features. Color variegation kept opposite point-estimate signs under all seven concept definitions in all four learner families, but not separated joint intervals on common-support subsets. Both reversals survive a size-matched control. After observing them, we designed a selection rule for recorded negatives that depends on one concept. It does not intervene on clinical biopsy decisions, and we do not claim that it matches their strength. Each concept-specific rule shifted the corresponding contrast in the predicted direction, and at a training size close to that of the biopsy-only learner the mean contrast changed sign.

**Third, what the verification process leaves identified about disease.** We define the disease target as the same stratum contrast on the log-odds scale and population, applied to the unclipped disease risk, and derive its identified set under a floor on malignant verification. The set is sharp with known nuisance functions, and a stratum-based outer set needs no estimate of how the input predicts the stratum. For the marginal association the condition reduces to a bound on one verification ratio. On ISIC-2024, analyzed after the primary results had been seen, the framework does not establish disease-relative sign claims; it shows why they stay sensitive to the floor and to the estimator of recorded-label risk. PAD-UFES-20, where every recorded malignancy is biopsied [@pacheco], is an external check on the attenuating side of the boundary.

**How the parts fit.** Together, the three parts audit one contrast. They ask what gap the verification propensity predicts, how a trained pipeline responds to an implemented selection rule, and which disease sign follows under which assumption. Each part rests on its own evidence: an identity, a design, or an explicit floor on malignant verification.

## II. Related Work

Four strands of work bear on this problem. Selection bias and case-control sampling describe how selection shifts an association or a fitted logit. Selective labels and positive-unlabeled learning describe what can be learned when outcomes are observed only after a decision. Verification bias and shortcut learning describe how acquisition distorts medical evaluation and learning. Partial identification describes what can be said when a target is not point identified.

**Selection bias and its direction.** Epidemiology expresses selection bias as a multiplicative factor on the odds ratio, built from the selection probabilities of the exposure-by-outcome table [@kleinbaum]. Quantitative bias analysis turns that factor into sensitivity parameters [@greenland; @lash]. Berkson described selection on a common effect [@berkson]. The direction of collider bias between binary variables has been characterized in closed form [@nguyen]. Reversal by selection has been analyzed for the obesity paradox [@banack], and collider bias affects modern cohorts [@griffith]. Our Lemma 1 restates these results with verification ratios, and we do not claim it as new.

**Case-control and subsampling offsets.** Case-control sampling that depends on the outcome alone leaves logistic slopes unchanged and moves the intercept [@prentice]. Local case-control sampling selects on the outcome and the covariates and corrects the fit by a known offset [@fithian]. Our Proposition 1 is an identity of this family, written for training data selected by clinical verification. The algebra is the same, but the use differs. The offset is set by clinical verification rather than chosen by the analyst, and we read its stratum contrast as a prediction of how two learned contrasts differ.

**Selective labels.** When outcomes are observed only after a human decision, evaluation and learning depend on that decision [@lakkaraju; @kleinberg]. Expert consistency [@dearteaga] and selective testing [@mullainathan] have been used to learn under such selection. Risk prediction under historical testing can be improved with domain constraints such as known prevalence [@balachandar]. Rambachan, Coston and Kennedy bound measures of predictive performance under sensitivity assumptions on unobserved confounding, with debiased nuisance estimation [@rambachan]. We do not propose a new bounding framework. Our target is a learner-matched logit contrast of latent disease whose weights change sign between strata. Proposition 2 applies the standard step of choosing each pointwise bound by the sign of the weight to this learner-matched contrast, and uses the resulting set to judge an observable learner reversal.

**Positive-unlabeled learning.** Writing *Y* = *D*·*S* makes every unverified malignancy an unlabeled positive, the structure of positive-unlabeled learning [@elkan; @bekker], including feature-dependent labeling propensities [@bekkersar]. These methods use a labeling propensity to learn a classifier for the latent label. We do not learn the disease risk. We bound a contrast of it under a floor on the labeling propensity of malignant lesions.

**Verification bias, missing data and shortcuts.** Verification bias in diagnostic accuracy [@begg] can be corrected under ignorable verification [@alonzo]. Graphical criteria address recovery from selection [@bareinboim] and missing data [@mohan], and learning under feature-dependent [@zadrozny] and outcome-dependent selection [@heckman] is classical. Dermatology classifiers exploit skin markings [@winkler] and hidden strata [@oakden], shortcut learning is general [@geirhos], and acquisition shapes what a model can learn [@castro].

**Partial identification and probing.** When a parameter is not point identified, one can report the set of values compatible with the data and stated assumptions [@manski; @tamer], and learn robustly over such sets [@kallus]. Linear probes [@alain] and centered kernel alignment [@kornblith] serve only in supplementary localization diagnostics.

Together, these strands explain how selection can shift an association and how a latent target can be bounded. They leave open how one verification process acts on three objects at once. These are the contrast a learner absorbs from its training data, the effect of changing training selection, and the matching contrast of latent disease. Section III defines the three objects separately and states what identifies each.

## III. Method

The method first fixes what is compared and only then how it is estimated. Section III-A states the causal model and the label convention. Section III-B defines the learned contrast, the effect of a training-selection rule, which the implemented design identifies, and two disease targets. Section III-C gives the Bayes-optimal offset whose predicted direction we compare with the learners. Section III-D treats the marginal disease association, and Section III-E the disease contrast on the learner's scale. Sections III-F and III-G then choose data, learners and inference to match these objects.

### A. Causal model

Let *D* be latent malignancy under the binary rule of the data set (Section III-F) and *X* the lesion's appearance. Concepts are deterministic readings of the image, *C* = *h*(*X*). Diagnosis is anticausal: disease shapes appearance, *D* → *X* → *C*. Let *U* be patient-level risk and surveillance intensity, *G* the acquisition site and *W* clinical information outside the image, such as reported change. A clinician flags lesions of interest (*F*) and biopsies some of them (*S*, with *S* ≤ *F* as indicators). The recorded label is *Y* = *D*·*S*. Fig. 1(a) shows the graph, with latent nodes dashed. The arrow from *G* to *D* stands for differences in case mix between acquisition sites. The graph lets verification depend on disease through *W* and through appearance. Disease can then be missing not at random among unverified lesions, whose recorded label is nonetheless zero. Lesions with *Y* = 0 are recorded negatives, and those with *S* = 1 are biopsied recorded negatives.

**Fig. 1.** Causal model of selective verification (a) and the regions of Lemma 1 (b).

A learner does not see *X* itself but an input *X̃*: tabular measurements or image features. A training regime is a selection *R* of lesions into the training sample. M0 trains on all lesions (*R* ≡ 1). M2 trains on verified lesions (*R* = *S*). The identities below use only the variables they name, so they hold for any graph consistent with *Y* = *D*·*S*. The graph describes how the data arose and is not the basis of identification.

### B. Targets

Split concept *k* at its cohort tertiles into lower (*t* = 0) and upper (*t* = 1) strata. For a learner *m*, the **learned contrast** Δ_{m,k} is the mean uncalibrated logit among upper-stratum lesions minus that among lower-stratum lesions. It is computed on a fixed evaluation population, with each lesion weighted equally, averaged over the randomness of training. The logit is computed from the predicted probability clipped to [10⁻⁷, 1 − 10⁻⁷], which caps it at ±16.12. We call this uncalibrated scale raw. For the two secondary families, which output pre-sigmoid logits directly, no clipping is applied (Supplementary Section S5). The theory of Section III-C concerns Δ^Bayes_{m,k}, the same contrast of the unclipped Bayes-optimal logit of the training distribution of learner *m*; the learned contrasts are its empirical counterparts. It is a marginal tertile contrast: it includes differences in correlated concepts between the strata and is not the effect of concept *k* with the others held fixed. We drop *k* when the concept is clear. Write *p*(*x̃*) = P(*D* = 1 | *x̃*). The **disease target** on the same scale and population is

ψ_k = E[logit *p*(*X̃*) | *t* = 1] − E[logit *p*(*X̃*) | *t* = 0],

assuming both stratum means are finite.

Because *p* depends on the input, ψ_k is specific to a learner family. The tabular-input and image-feature disease targets are different functionals of the same latent disease.

A **learner reversal** is a difference in sign between Δ_{M0,k} and Δ_{M2,k}. It is observable. But M0 and M2 differ in more than one respect, so a learner reversal is not by itself the effect of a selection rule. We define that effect next. A **learner-scale disease-relative reversal** is a difference in sign between Δ_{M2,k} and ψ_k. It can be established only as far as the identified set for ψ_k determines the sign of ψ_k. We also use the marginal disease contrast θ_k = log OR_D between the strata, the classical object of selection bias. A **marginal association reversal** is a positive θ_k with a negative verified-only association log OR_{D|S} (Lemma 1), or the reverse.

**Training-selection effect.** Let η index a rule, set by the analyst, for selecting recorded negatives into the training set only. η does not change the validation set used for early stopping. Let ω collect the exogenous random numbers of the pipeline: inclusion uniforms, the malignant training draw, the validation draw and the optimizer seed. The contrast of a single fit is then a potential outcome Δ_j(η, ω), and the training-selection effect on concept *j* is τ_j(η₁, η₀) = E_ω[Δ_j(η₁, ω) − Δ_j(η₀, ω)]. Each rule is implemented directly with ω held fixed within a replicate, so τ_j is identified by design, conditional on the realized cohort, patient split and evaluation population, with no assumption about clinical verification. It is not the effect of a concept on disease, nor of a clinical biopsy policy, nor a superpopulation effect over repeated patient samples.

### C. The training-selection offset

**Proposition 1 (training-selection offset).** Write *r*_y(*x̃*) = P(*R* = 1 | *Y* = *y*, *x̃*) for the probability that a lesion with recorded label *y* and learner input *x̃* enters training. Then, for every *x̃* with 0 < P(*Y* = 1 | *x̃*) < 1 and *r*₀(*x̃*), *r*₁(*x̃*) > 0,

logit P(*Y* = 1 | *x̃*, *R* = 1) = logit P(*Y* = 1 | *x̃*) + log *r*₁(*x̃*) − log *r*₀(*x̃*).

*Proof sketch.* By Bayes' rule, the odds of *Y* = 1 given *x̃* and *R* = 1 equal P(*Y* = 1 | *x̃*)·*r*₁(*x̃*) divided by P(*Y* = 0 | *x̃*)·*r*₀(*x̃*). Supplementary Section S1 gives the proof. The identity is stated on the learner's own input and uses no disease label. Under biopsy-only selection the offset is −log *g* ≥ 0, so the biopsy-trained logit exceeds the full-cohort logit. For example, *g* = 0.1 at an input raises the Bayes-optimal logit there by log 10.

Suppose *R* is independent of *X̃* given *X* and *Y*, as when selection depends only on appearance and the recorded label. The selection probability then averages over what the input does not show: *r*₀(*x̃*) = E[*r*₀(*X*) | *x̃*, *Y* = 0]. Two cases matter here. *Training on verified lesions.* With *R* = *S*, every malignant training lesion is verified, so *r*₁ = 1. Then *r*₀(*x̃*) = *g*(*x̃*) = P(*S* = 1 | *x̃*, *Y* = 0), the verification propensity among recorded negatives. No further condition is needed. For the Bayes-optimal learners *f*₀ of M0 and *f*₂ of M2, logit *f*₀ − logit *f*₂ = log *g*, and therefore

Δ^Bayes_{M0,k} − Δ^Bayes_{M2,k} = E[log *g*(*X̃*) | *t* = 1] − E[log *g*(*X̃*) | *t* = 0].

Write *G*_k for the right-hand side. Then Δ^Bayes_{M2,k} = Δ^Bayes_{M0,k} − *G*_k, so the Bayes-optimal learners reverse exactly when Δ^Bayes_{M0,k}(Δ^Bayes_{M0,k} − *G*_k) < 0, for either sign of Δ^Bayes_{M0,k}. The stratum means require logit *q* and log *g* to be integrable within each stratum. *A controlled selection dose.* Let each recorded-negative training lesion enter independently with probability ρ_η = min{1, κ_η exp(η*z*_k(*X*))}. Here *z*_k is the standardized concept, and κ_η fixes the expected number of selected recorded negatives. If recorded positives are drawn uniformly, *r*₀ is known by design. When *z*_k is part of the learner's input, *r*₀(*x̃*) = ρ_η(*z*_k). Without the cap, the Bayes contrast of any concept *j* then moves linearly: Δ^Bayes_j(η) = Δ^Bayes_j(0) − η·μ_{kj}. Here μ_{kj} is the difference in mean *z*_k between the strata of concept *j*. The Bayes-optimal training-distribution shift is then −(η₁ − η₀)·μ_{kj}. When the input determines *z*_k only partly, as for image features, *r*₀(*x̃*) = E[ρ_η(*z*_k(*X*)) | *x̃*, *Y* = 0]. We approximate it with a Gaussian model for *z*_k given *x̃* (Supplementary Section S1). Adding a constant to a logit cancels in the contrast.

These statements are exact for Δ^Bayes, the unclipped Bayes-optimal logit of the selected training distribution. A trained network adds approximation, optimization and early-stopping effects, and clipping can change a difference of stratum means. For a trained pipeline they give a predicted direction, which we compare with the observed effect τ_j; they do not guarantee its sign, and agreement in magnitude is descriptive (Section IV).

### D. Classical identities for the disease association

**Lemma 1 (verification selection factor).** For the strata of concept *k*, let π_t^d = P(*S* = 1 | *D* = *d*, *t*), *A* = π₁¹/π₀¹, *B* = π₁⁰/π₀⁰ and δ = log *B* − log *A*. Then log OR_{D|S} = θ − δ. For θ > 0, verification amplifies the association if δ < 0, attenuates it if 0 < δ < θ, and reverses it if δ > θ. For θ < 0 the regions mirror, and if θ = 0 verification alone creates an association equal to −δ.

Fig. 1(b) shows these regions in the plane of θ and δ: reversal occurs above the diagonal for positive θ and below it for negative θ. Lemma 1 is the classical selection bias factor [@kleinbaum; @greenland] with verification ratios, and its reversal region is the known possibility of reversal by selection [@nguyen; @banack]. *B* is not observable, because π⁰ conditions on latent *D*. What is observed is *B*_V = *g*₁/*g*₀ with *g*_t = P(*S* = 1 | *Y* = 0, *t*), the share of biopsied lesions among recorded negatives of stratum *t*. It equals E[*g*(*X̃*) | *Y* = 0, *t*] whenever *t* is a function of *x̃* or *S* is independent of *t* given *x̃* and *Y* = 0, and may differ otherwise (Supplementary Section S1). It differs from *B* by a factor close to one when disease is rare, and Supplementary Section S1 bounds it.

Supplementary Section S1 adds Lemma 2. When the log ratio of malignant to benign verification probabilities is linear in *x* with slope λ, a population-optimal verified-only logistic learner shifts the disease coefficients by λ.

### E. Identification of the disease target

Let *q*(*x̃*) = P(*Y* = 1 | *x̃*), and let σ(*x̃*) = P(*S* = 1 | *x̃*) be the overall verification propensity. Let *s*(*x̃*) = P(*S* = 1 | *D* = 1, *x̃*) be the probability that a malignant lesion with input *x̃* is verified. Assume the pointwise floor *s*(*x̃*) ≥ *s*_min for every *x̃*, with *s*_min ∈ (0, 1]. This is stronger than a floor on the share of malignant lesions that are verified, which can exceed *s*_min while the pointwise floor fails in part of the input space.

Write *a*_k(*x̃*) = P(*t* = 1 | *x̃*)/P(*t* = 1) − P(*t* = 0 | *x̃*)/P(*t* = 0), so that ψ_k = E[*a*_k(*X̃*) logit *p*(*X̃*)].

**Proposition 2 (identified set for the disease target).** Let *p̄*(*x̃*) = min{*q*(*x̃*)/*s*_min, *q*(*x̃*) + 1 − σ(*x̃*)}. Let *L*(*x̃*) = logit *q*(*x̃*) where *a*_k(*x̃*) ≥ 0 and *L*(*x̃*) = logit *p̄*(*x̃*) where *a*_k(*x̃*) < 0. Let *H* swap the two endpoints. Where *a*_k = 0, either endpoint contributes zero. Then ψ_k lies in [ψ_L, ψ_U], with ψ_L = E[*a*_k(*X̃*)·*L*(*X̃*)] and ψ_U = E[*a*_k(*X̃*)·*H*(*X̃*)]. Assume *q* and *p̄* lie in (0, 1) and that logit *q* and logit *p̄* are integrable within each outer tertile, so that ψ_k, ψ_L and ψ_U are finite. With the true *q*, σ and *a*_k this interval is sharp relative to the nonparametric model of the observed data that satisfies *Y* = *D*·*S* and the pointwise floor. Further restrictions on verification, such as structural, smoothness, monotonicity or cross-input constraints that the graph of Fig. 1(a) might motivate, could narrow it. The sign of ψ_k is identified as positive when ψ_L > 0 and as negative when ψ_U < 0.

**Corollary (stratum formula).** When *t* is a function of *x̃*, ψ_L = E[logit *q* | *t* = 1] − E[logit *p̄* | *t* = 0] and ψ_U = E[logit *p̄* | *t* = 1] − E[logit *q* | *t* = 0]. For any other input these stratum formulas are valid outer bounds.

*Proof sketch.* Pointwise, *p* = *q*/*s* lies between *q* and *p̄*, and both endpoints are attained. Its extremes combine the endpoints by the sign of *a*_k; assigning them by observed stratum can only widen the interval (Supplementary Section S1). Without a floor, the observed law still gives *p* ≤ *q* + 1 − σ. Because fewer than 0.3 percent of lesions are verified, this bound is close to one for most inputs, and the floor is what makes it informative. The population tipping floor is *s*∗ = inf{*s*_min : ψ_L > 0}, computed from the exact bound; a positive sign is identified for every floor above it. With the stratum formula, write Δ_q = E[logit *q* | *t* = 1] − E[logit *q* | *t* = 0] for the true recorded-label risk, and Δ̂_q for its estimate from a calibrated M0. Where *q*/*s*_min is the active branch and small, ψ_L ≈ Δ_q + log *s*_min, so *s*∗ ≈ exp(−Δ_q). This approximation only interprets *s*∗ and can misstate the sign near it (Supplementary Section S1). Above *s*∗, a biopsy-trained learner with a negative contrast is reversed relative to disease.

The floor enters ψ_L only through *p̄*, where *a*_k < 0. A positive sign thus needs the floor only there, and with the stratum formula only on inputs that occur in the lower stratum. We state the floor for every *x̃* for simplicity. The tabular input contains the five concepts, so the stratum formula is sharp for the tabular-input target. Image features determine the concept only partly, so for the image-feature target we report the stratum formula as an outer set, which needs no estimate of *a*_k. Its tipping floor is then a sufficient floor and can exceed that of the sharp set.

Proposition 2 assumes *q* and σ known. On ISIC-2024 we replace *q* by M0 on the learner's own input, calibrated on the validation set. We replace σ by one out-of-fold gradient-boosting model on the tabular features, shared by both families and held fixed in every bootstrap replicate. {{N:nuis_sentence}} Probabilities are clipped to [10⁻⁷, 1 − 10⁻⁷] before the logit, and isotonic estimates to [10⁻⁶, 1 − 10⁻⁶]. The result is a plug-in estimate of the identified set, without the sharpness of the population set. The primary plug-in uses Platt calibration [@platt]. No estimator of *q* is privileged by theory, so we also report isotonic calibration [@zadroznyelkan] and gradient boosting [@friedman] under one bootstrap. The smallest floor on a grid of step 0.01 at which the plug-in lower endpoint is positive is the plug-in tipping floor. The smallest floor at which it is positive in at least 95 percent of bootstrap replicates is the bootstrap stability threshold. Both describe the plug-in procedure under one nuisance specification, not a floor identified from the data.

### F. Data and learners

**Cohorts.** ISIC-2024 contains 401,059 lesion image crops, one per lesion, from 1,042 patients [@kurtansky]. A lesion identifier marks 22,058 clinician-tagged lesions (*F*). Of these, 1,068 carry histopathology (*S*): 393 malignant (*Y*), 561 benign and 114 indeterminate (75 melanocytic and 39 epidermal proliferations). The binary target counts indeterminate lesions as not malignant. We keep this rule, so *D* is malignancy under it, and biopsied recorded negatives include the indeterminate lesions. The learners are trained on the released label. {{N:indet_sentence}} The disease-side count analyses are repeated with indeterminate lesions coded as unverified or as malignant (Supplementary Section S7). Every malignant label is histopathologically confirmed, and lesions never linked to pathology were recorded as benign [@kurtansky], so the release follows *Y* = *D*·*S*. PAD-UFES-20 contains 2,298 smartphone images of 1,641 lesions from 1,373 patients [@pacheco]. Basal cell carcinoma, squamous cell carcinoma and melanoma form the malignant class, and actinic keratosis, nevus and seborrheic keratosis the non-malignant class. Placing actinic keratosis, a premalignant lesion, in the non-malignant class is a binary mapping chosen for this analysis. Every image recorded as malignant was biopsied, and non-malignant lesions that were not biopsied carry a clinical diagnosis.

**Concepts.** We use five concepts measured by the total-body photography system [@kurtansky]: color variegation, size, lesion-skin contrast, asymmetry and border irregularity. Color variation, asymmetry and border are aspects of the ABCD rule of dermoscopy [@nachbar]. The system measures them on total-body photographs, not on dermoscopic images. The analysis lock made color variegation, size and lesion-skin contrast primary and the other two secondary.

**Split and regimes.** Patients were split 60/20/20 into train, validation and test, stratified by whether a patient had any malignant label. The three splits hold 624, 209 and 209 patients with 222, 92 and 79 recorded malignancies. The test population holds 78,625 lesions. M2 is trained on 622 verified training lesions, of which 222 are malignant. Every malignant training lesion is verified. The two regimes therefore differ in the size of the training set and the source of its recorded negatives. For the two primary perceptron families, they also differ in the validation lesions used for early stopping and calibration: all validation lesions for M0, verified ones for M2. The two secondary families run for a fixed number of steps, select the checkpoint by validation AUROC and are not calibrated (Supplementary Section S5).

**Learner families.** (i) A two-layer perceptron on 47 tabular features: 29 measurements of the total-body photography system and 18 columns for age, sex, body site, acquisition site and skin tone. No identifier, flag, biopsy or diagnosis field enters the input. Features are z-scored, and tertile cut points set, once on the whole cohort without labels, and every analysis conditions on this preprocessing (Supplementary Section S5). (ii) The same perceptron on frozen ResNet-50 features with ImageNet weights (torchvision IMAGENET1K_V2) [@he]. (iii) A linear probe on the frozen ResNet-50. (iv) ResNet-50 fine-tuned end to end. M0 and M2 share every other setting. The two frozen-feature families are primary. Supplementary Section S5 lists every setting and input column.

### G. Analyses and inference

**Primary inference.** The primary family is the three primary concepts in the two frozen-feature families, six comparisons. Learned contrasts are on this raw scale; calibration enters only the plug-in nuisances of Proposition 2 and the gap diagnostics. Point estimates average three seeds of the original fits. In each of {{N:B_primary}} joint bootstrap replicates per family, we resample training patients, refit M0 and M2 with three seeds and calibrate them on the validation set. We then resample test patients and average the three seeds, so the bootstrap statistic is the point estimator itself, given the fixed preprocessing and validation set. This inference conditions on the realized validation set. It treats the observed patients as a sample from a population like the observed sites, so it does not cover a new site; for the fixed test population it is a sensitivity analysis to patient resampling (Supplementary Section S6). Checks use {{N:B19}} one-seed replicates, {{N:B30}} replicates that also resample validation patients, and a descriptive half-sampling check without replacement. A learner reversal is called Bonferroni sign-separated when the Bonferroni-adjusted percentile bootstrap intervals [@efron] of Δ_M0 and Δ_M2, at level 1 − 0.05/6, lie on opposite sides of zero. The claim concerns one cell. A false claim in either direction needs a one-sided miss of probability at most 0.05/12, so the error is at most 0.05/6 per cell and 0.05 over six cells if the bootstrap tails are nominal. The intervals are not simultaneous intervals for the twelve contrasts. The label refers to the tertile definition and, unless stated otherwise, to the fixed validation set. Secondary concepts are reported with 95 percent intervals and are not part of the multiplicity family. {{N:mc_rule}} All other joint intervals refit one seed per replicate, except the refitted three-seed checks of Supplementary Section S6. For the linear-probe and fine-tuned families, intervals resample test patients only.

**Status of analyses.** An analysis lock fixed the primary and secondary outcomes after the primary comparison had first been inspected and before the analyses that test alternative explanations. The lock is dated 25 September 2026 and is part of the first public commit of the repository. It is a structured post-inspection plan, not a preregistration, and the primary analysis is partly data-informed. The Bonferroni adjustment controls multiplicity within this locked six-comparison family only, so it does not make the primary inference confirmatory. After inspecting the primary results, we changed the primary bootstrap statistic from a one-seed refit to the three-seed average used by the point estimator. We added the Monte Carlo resolution rule at the same stage, which made color variegation with tabular features unresolved. Alternative definitions and families reuse the same lesions and are not independent tests. Apart from the size-matched control, every analysis in Sections IV-B to IV-G, including the dose experiment, the marginal analysis and the choice of color variegation as the worked example, was added after the primary results had been seen.

**Checks and simulation.** Symbolic and unit-test checks verify the algebra and the code, and a phase diagram checks Lemmas 1 and 2 on 648 populations (Supplementary Sections S2 and S3). Simulations test the plug-in set of Proposition 2 in synthetic populations and in an adversarial design with an overstated floor. Semi-synthetic designs on the real ISIC-2024 inputs include one where the floor fails where verification is rarest.

Each result in Section IV is read under its own condition. The learner reversal compares two fitted pipelines on one test population. The training-selection effect is identified by design, conditional on the cohort and split. The gap prediction is exact only for Bayes-optimal learners. On the learner's scale, statements about disease depend on the floor and on the estimates of *q* and σ. The marginal analysis depends on a floor and on counts only.

## IV. Results

Section IV-A reports the learner reversal that the rest of the section examines. Sections IV-B to IV-D test it against training-set size, training selection, the verification propensity and support. Section IV-E checks the attenuating case on PAD-UFES-20. Sections IV-F and IV-G turn to disease, first on the learner's scale and then as a marginal association.

### A. Learner reversal on ISIC-2024

Table 1 reports the learned contrasts. {{N:jb_sentence}}

{{N:resample_sentence}}

{{N:clip_sentence}}

Fig. 2 shows all four families. The linear-probe and fine-tuned families also reversed color variegation, with no test bootstrap resample of the same sign, conditional on the fitted models. {{N:size_defs_sentence}} {{N:balanced_sentence}}

In Table 1, contrasts are uncalibrated logit contrasts on the ISIC-2024 test population, and M2 is early-stopped on verified validation lesions. The primary rows show Bonferroni-adjusted percentile intervals at level 1 − 0.05/6 from 5,000 three-seed joint resamples of training and test patients, conditional on the realized validation set. The secondary rows show 95 percent intervals; they lie outside the six-comparison family and carry no Bonferroni label. "Unresolved" marks a sign-separation label of the full replicate set that was reproduced in fewer than 95 percent of Monte Carlo resamples of the stored replicates. The same-sign count is descriptive, not a test.

**TABLE 1. Learned contrasts of the frozen-feature learners on the ISIC-2024 test population.**

{{TABLE:main_reversal}}

In Fig. 2, M0 is shown as circles and M2 as triangles on panel-specific scales. The frozen-feature panels use clipped logits on all 78,625 test lesions; the linear-probe and fine-tuned panels use pre-sigmoid logits on a uniform sample of 30,000 test lesions from 207 patients, with checkpoints chosen by validation AUROC (Supplementary Section S5). Its bars are test-patient bootstrap intervals conditional on the fitted models and therefore differ from the joint training-and-test bootstrap intervals of the primary inference in Table 1.

**Fig. 2.** Learned contrasts of M0 and M2 in the four learner families.

### B. Controlled selection of recorded negatives can induce the reversal

Supplementary Section S6 summarizes the tests of alternative explanations in one table and reports each in full.

A size-matched control combined the same malignant lesions with 320 recorded negatives per replicate. For color variegation and size, in both families, the contrast was positive in 20 of 20 replicates when the recorded negatives were drawn at random. It was negative in 20 of 20 when they were biopsied ones. For color variegation it was also negative in 20 of 20 when they were flagged but never biopsied. Training-set size alone therefore does not explain these two reversals. Each arm also drew its validation set from its own pool, so the arms differ in their development data, not only in the source of training negatives. For lesion-skin contrast with image features the random arm was itself close to zero, so the control does not separate size from selection there. {{N:arms_sentence}}

A controlled selection dose changed only which recorded negatives entered training (Fig. 3(a)). In each of 20 replicates, a validation set was drawn. It was shared by all five doses, together with the malignant training lesions and the seed. Each recorded-negative training lesion received one uniform variable that decided its inclusion at every dose. Each replicate therefore fixes ω, and contrasts between doses are paired potential outcomes. The outcome is the raw-logit contrast, because Platt calibration on the unselected validation set would target the unselected distribution (Supplementary Section S6).

{{N:dose_sentence}}

In Fig. 3(a), points are means of the raw logit contrast of color variegation over the 20 paired replicates with 320 expected recorded negatives, with 2.5 to 97.5 percentile ranges. Its dashed lines are the predicted shift of the Bayes-optimal logit contrast given each learner's input, anchored at η = 0; their comparison with the observed points is a diagnostic. For image features, they use a ridge estimate of the conditional concept mean. In Fig. 3(b), points are calibrated medians over 200 one-seed joint resamples with 95 percent percentile intervals, against the identity line, and labels name the concept of each point in its family's color.

**Fig. 3.** Controlled selection dose (a) and observed against predicted learner gap (b).


### C. The learner gap follows the recorded-negative verification propensity

{{N:bridge_sentence}} Fig. 3(b) plots the observed against the predicted concept-level gap. Agreement between observed and predicted gaps also reflects approximation error, calibration, estimation of *g* and optimization. It is a diagnostic, not a test of the identity; the symbolic and unit tests verify the identity itself (Supplementary Section S2).

Verification of recorded negatives depended on appearance mainly within patients. A one standard deviation increase in color variegation relative to the patient's own lesions raised the log odds of verification by 0.93 (SE 0.04). For the patient's mean, the increase was 0.10 (SE 0.15). This describes a verification pattern, consistent with clinicians biopsying a patient's atypical-looking lesions, and does not identify the clinical mechanism. {{N:site_sentence}}

### D. Overlap

{{N:overlap_paragraph}}

### E. External consistency check on PAD-UFES-20

On PAD-UFES-20, every image recorded as malignant was biopsied, so *A* = 1 under the recorded-diagnosis operationalization and Lemma 1 then reduces to log OR_{D|S} = θ − log *B*. Verification attenuated ten of eleven association features, amplified one and reversed none (Fig. 4). The learner contrasts showed the same qualitative attenuation. With patients split 60/20/20, M2 trained on the 814 biopsied training images had a smaller contrast than M0 for all six symptoms. For image color variegation, the contrast fell from +2.06 [1.47, 2.84] to +0.70 [0.22, 1.23]. The cohorts differ in modality, population, prevalence and workflow. PAD-UFES-20 illustrates the attenuating regime of Lemma 1 and validates neither the mechanism on ISIC-2024 nor Lemma 1 for latent disease. If some clinically diagnosed lesions were malignant, *A* could differ from one. The direction depends on where the missed malignancies fall. Missed cases only in the lower tertile would raise *A* above one, and only in the upper tertile would lower it below one.

In Fig. 4, the PAD-UFES-20 features are placed at *A* = 1 under the recorded-diagnosis operationalization. Each ISIC-2024 path computes θ and δ exactly from whole-cohort counts for π₀¹ = 0.5 and *A* from 0.5 to 2. Every point satisfies θ − δ = log OR_{D|S}, so each path has slope one, keeps the verified-only sign and crosses θ = 0 at *A* = RR_Y. Open circles mark *A* = 1, which moves by less than 0.01 for π₀¹ from 0.25 to 1. The paths are sensitivity paths, not confidence intervals.

**Fig. 4.** ISIC-2024 and PAD-UFES-20 in the plane of Lemma 1.

### F. Learner-scale disease target: an assumption-indexed sensitivity analysis

{{N:psi_primary_sentence}} {{N:sharp_sentence}}

{{N:psi_est_sentence}} {{N:tail_sentence}}

In Table 2, Δ̂_q is the difference in mean logit *q̂* between the outer tertiles of the test population for the original fits. The point floor uses the exact bound on the seed-averaged curve of ψ_L, and the Monte Carlo range of the stability threshold comes from 500 resamples of the replicates. Image-feature floors use the common tabular σ̂. Ranges across estimators of *q* summarize specification sensitivity and are not statistical confidence intervals. Joint counts are resampling frequencies under the stated plug-in procedure, not probabilities that a disease-relative reversal holds. Log-loss is on the test population for the original fits.

**TABLE 2. Plug-in tipping floor for color variegation under three estimates of *q*.**

{{TABLE:psi_est}}


{{N:psi_sim_sentence}}

### G. Separate post-inspection marginal sensitivity analysis

{{N:marg_paragraph}}

Taken together, the results separate three kinds of evidence. The designed selection rule changed the learned contrast of a fitted pipeline. The observed verification propensity predicted the direction of the gap between the two regimes, with a size that depended on the propensity model. Statements about disease depended on an assumed floor and, on the learner's scale, also on the estimator of *q*. One limit applies to the learner reversal itself: on common-support subsets the joint intervals no longer separated the color signs.

## V. Discussion

The results support claims about the training pipeline and only conditional claims about disease. We state each, then the threats to validity and the implications for reading a classifier's concept contrasts.

**What the evidence supports.** A designed change in which recorded negatives entered training moved what a classifier learned in the predicted direction. With 320 expected recorded negatives, the mean learned contrast also changed sign across the designed dose range. The observed learner reversal between M0 and M2 persisted when training-set size was matched, with each arm using its own validation set. On the calibrated scale and the full test population, the gap between the two learners had the sign predicted from the recorded-negative verification propensity in all ten cells, a descriptive agreement over correlated concepts. The observed reversal itself is not a randomized contrast. Relative to disease, the evidence comes from the marginal analysis of Section IV-G. For color variegation, crude pooled counts indicated a reversal under an unchecked lower-tertile floor, in the whole cohort and, at a higher floor, in the test population; the fixed six-site standardization did not reach the resampling criterion for color variegation but did for size. The analysis does not identify or validate transport to a new site.

**What the evidence does not support.** We do not claim that verification always reverses learned relationships; in the phase diagram no population reversed when benign verification did not depend on appearance. We do not claim a Bonferroni sign-separated learner reversal for every concept, or a learner reversal of color variegation within better-supported regions, where the M0 contrast fell and could turn negative. We do not claim a learner-scale disease-relative reversal. That would need a floor on malignant verification where *a*_k < 0, or on inputs of the lower tertile for the stratum formula, which this cohort cannot check, and a better estimate of recorded-label risk in the rare-event tail. A floor that fails where verification is rarest is one violation we simulated. The controlled experiment intervenes on training selection, not on clinical biopsy decisions, and we do not claim that its doses match the strength of clinical verification. Localization diagnostics on the fine-tuned encoders found no evidence of a general representation-level reversal (Supplementary Section S5).

**Threats to validity.** *Concepts.* The concepts are measurements from total-body photography, and the results apply to them as measured. *Inference.* The test population holds 209 patients, and site-level heterogeneity in verification was substantial. Fine-tuning under M0 was unstable across seeds, which is why that family is secondary. *Estimation.* The width of the identified set under the floor does not shrink with more data. {{N:local_sentence}}

**Implications.** Concept-level readings of a classifier trained on biopsied lesions should not be read as statements about disease. This holds for the stratum contrasts studied here and, plausibly, for directional derivatives along concept vectors such as TCAV [@kim], which we did not study. Marginal concept contrasts should not be read as isolated concept effects. Comparisons across data sets should record how each was verified. Estimating the selection propensity needs the full cohort of candidate lesions with a biopsy indicator. Recording the clinical reasons for biopsy could support point identification of disease risk. This would require exchangeability of verification given that information, positivity and correct measurement.

In short, two results concern the pipeline and need no assumption about disease. The designed selection rule shows that changing training negatives changes the learned contrast, and the observed verification propensity predicts the direction of the gap between the two regimes. Conclusions about disease need an assumption on malignant verification that the cohort cannot check. The two disease analyses can disagree for three reasons: they target different quantities, a ratio of mean risks against a contrast of mean log odds; they use different populations, the whole cohort against the test population; and they assume different floors, an average within the lower tertile against a floor on inputs.

## VI. Conclusion

We asked whether selective verification changes what a classifier learns. The training-selection offset is exact for Bayes-optimal learners and observable for biopsy-based training, and in trained learners its predicted direction held. On ISIC-2024, the two regimes gave color variegation and size contrasts of opposite sign on the full test population. Size with image features kept Bonferroni sign separation under every bootstrap scheme, but only for the tertile contrast. Color kept opposite point-estimate signs under every concept definition, but its joint intervals no longer separated the signs on common-support subsets. The designed change in training selection has an effect identified by design, conditional on the cohort. Each selected contrast moved in the predicted direction and, with 320 expected recorded negatives, its mean changed sign across the designed dose range. The ISIC-2024 analysis did not establish a learner-scale disease-relative reversal, and the plug-in tipping floor depended on the estimator of recorded-label risk. In the crude pooled cohort, the marginal analysis indicated a reversal for color variegation only under an assumed lower-tertile verification floor, and the fixed six-site analysis did not reach the resampling criterion. Changing by experiment which recorded negatives entered training altered the learned concept contrast. Training selection can therefore shape the statistical relationships a classifier learns, not only filter how it is evaluated.

## Statements

**Data availability.** Both data sets are public. ISIC-2024 is distributed through the ISIC Archive [@kurtansky] and PAD-UFES-20 through Mendeley Data [@pacheco]. No new patient data were collected.

**Code availability.** The scripts that produce every number, table and figure in this paper are available at https://github.com/dqtoan87/CausalReversal, together with the symbolic checks, the unit tests and the analysis lock. The version reported here is release v1.11-submission. Supplementary Section S8 maps each result to its script.

**Ethics.** This is a secondary analysis of two de-identified public data sets released under the approvals documented by their providers. No further approval was required.

**Author contributions.** Quang Toan Dao designed the study, derived the theory, implemented the pipeline, ran the analyses and drafted the manuscript. Viet Anh Nguyen supervised the study and revised the manuscript.

**Funding.** This work received no specific grant from any funding agency.

**Conflicts of interest.** The authors declare no competing interests.

<!-- REFERENCES -->
