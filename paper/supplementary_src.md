# Supplementary Material

**Causal Analysis of Learning Reversal under Selective Verification in Skin Cancer Classification**

## S1. Notation and proofs

Table {{T:notation}} fixes the notation. The code uses the same names, so each symbol can be traced to its implementation.

**TABLE {{T:notation}}. Notation used in the paper and in the code.**

| Symbol | Meaning | Name in code |
| --- | --- | --- |
| *D* | latent malignancy under the binary rule of the release, in which Indeterminate histopathology is not malignant; observed only where *S* = 1 | `D` |
| *X*, *C* = *h*(*X*) | lesion appearance and a concept read from it | `x`, concept columns |
| *U*, *G*, *W* | patient-level risk, acquisition site, clinical information outside the image | not modeled directly |
| *F* | lesion flagged by a clinician as a lesion of interest | `F` |
| *S* | lesion has a linked histopathology diagnosis, Malignant, Benign or Indeterminate; *S* ≤ *F* as indicators | `S` |
| *Y* = *D*·*S* | recorded malignant label | `Y` |
| *t* | lower (0) or upper (1) cohort tertile of a concept | `tertile` |
| π¹(*x*), π⁰(*x*) | P(*S* = 1 \| *D* = 1, *x*) and P(*S* = 1 \| *D* = 0, *x*) | `pi1`, `pi0` |
| *g*(*x*) | P(*S* = 1 \| *x*, *Y* = 0), verification propensity among recorded negatives, observable | `g` |
| *A*, *B* | π₁¹/π₀¹ and π₁⁰/π₀⁰, not observable | `logA`, `logB` |
| *g*_t, *B*_V | P(*S* = 1 \| *Y* = 0, *t*), the share of biopsied lesions among recorded negatives of tertile *t*; *B*_V = *g*₁/*g*₀ | `g_t`, `B_V` |
| θ_k, δ | disease contrast log OR_D and log *B* − log *A* | `theta`, `delta` |
| *q*(*x*), σ(*x*) | P(*Y* = 1 \| *x*) and P(*S* = 1 \| *x*), the overall verification propensity | `q`, `sigma` |
| *s*(*x*), *s*_min | P(*S* = 1 \| *D* = 1, *x*) and its assumed pointwise floor | `s`, `s_min` |
| *p̄*(*x̃*) | upper bound min{*q*/*s*_min, *q* + 1 − σ} on disease risk in Proposition 2 | `U` |
| M0, M2 | learner on all lesions with *Y*; learner on verified lesions only | `erm_y`, `erm_verified` |
| Δ_{m,k} | learned contrast of learner *m* for concept *k* | `vr9_closing` |
| η, ρ_η, *u*_i | selection dose, inclusion probability of a recorded-negative training lesion, and the uniform variable that decides inclusion of lesion *i* | `vr26_dose_poisson`, `vr21_dose_calibrated` |
| ω | exogenous random numbers of the pipeline: inclusion uniforms, malignant training draw, validation-set draw and optimizer seed | `vr46_dose_fixed_val` |
| τ_j(η₁, η₀) | training-selection effect, E_ω[Δ_j(η₁, ω) − Δ_j(η₀, ω)] | `vr46_dose_fixed_val` |

**Positivity and integrability.** Throughout, P(*D* = 1 | *x*) ∈ (0, 1), π¹, π⁰ ∈ (0, 1], *r*₀, *r*₁ > 0 and *s*_min ∈ (0, 1]. Stratum means of logits and log propensities are assumed finite. In the code, probabilities are clipped to [10⁻⁷, 1 − 10⁻⁷] before a logit. If a verification probability is zero, the verified-only odds ratio is undefined. A unit test checks that the code does not return a finite value in that case.

**Proof of Proposition 1.** Fix an input value *x̃* and write *r*_y(*x̃*) = P(*R* = 1 | *Y* = *y*, *x̃*). By Bayes' rule, P(*Y* = 1, *R* = 1 | *x̃*) = P(*Y* = 1 | *x̃*)·*r*₁(*x̃*) and P(*Y* = 0, *R* = 1 | *x̃*) = P(*Y* = 0 | *x̃*)·*r*₀(*x̃*). Their ratio is the odds of *Y* = 1 given *x̃* and *R* = 1, and taking logarithms gives the statement. No disease label, and no assumption on how *D* enters *Y*, is used.

*Selection that depends on appearance.* Suppose *R* is independent of *X̃* given *X* and *Y*, and write *r*_y(*x*) = P(*R* = 1 | *Y* = *y*, *x*). Then *r*_y(*x̃*) = E[*r*_y(*X*) | *x̃*, *Y* = *y*]. Without this condition, for example if clinicians used age or site beyond what the image shows, the statement still holds with *r*_y defined on the input. The averaging step, however, does not.

*Training on verified lesions.* With *R* = *S*, a lesion with *Y* = 1 has *S* = 1, so *r*₁(*x̃*) = 1. On the input itself, *r*₀(*x̃*) = P(*S* = 1 | *x̃*, *Y* = 0) = *g*(*x̃*). The Bayes-optimal M0 predicts *f*₀ = P(*Y* = 1 | *x̃*). The Bayes-optimal M2 predicts *f*₂ = P(*Y* = 1 | *x̃*, *S* = 1), which equals P(*D* = 1 | *x̃*, *S* = 1). Hence logit *f*₀ − logit *f*₂ = log *g*(*x̃*), and the offset of biopsy-only selection is logit *f*₂ − logit *f*₀ = −log *g* ≥ 0. Write *G*_k for the stratum contrast of log *g*. Taking the mean of these unclipped logits over each stratum of the evaluation population and subtracting gives Δ^Bayes_{M2,k} = Δ^Bayes_{M0,k} − *G*_k, so the two Bayes-optimal contrasts have opposite signs exactly when Δ^Bayes_{M0,k}(Δ^Bayes_{M0,k} − *G*_k) < 0. If a logit is shifted by a constant, for example because positives were resampled, the shift is the same in both strata and cancels in the contrast. These statements concern unclipped logits. The learned contrasts clip the predicted probability before the logit, which can change a difference of stratum means, so they are compared with Δ^Bayes only as a diagnostic.

*Controlled selection dose.* Recorded-positive training lesions are drawn uniformly, so *r*₁ is constant. Each recorded-negative training lesion enters independently with probability ρ_η(*z*) = min{1, κ_η exp(η*z*)}, with *z* = *z*_k(*X*) and κ_η chosen so that the expected number equals the target. Selection depends on the lesion only through *z*_k and *Y*, so the averaging step applies by construction. If *z*_k is a function of *x̃*, then *r*₀(*x̃*) = ρ_η(*z*_k(*x̃*)) exactly, and the predicted change of Δ^Bayes_j is minus the stratum contrast of log ρ_η(*z*_k) on the evaluation population. Without the cap this is −η·μ_{kj}, with μ_{kj} = E[*z*_k | *t*_j = 1] − E[*z*_k | *t*_j = 0]. If the input determines *z*_k only partly, *r*₀(*x̃*) = E[ρ_η(*z*_k(*X*)) | *x̃*, *Y* = 0]. Ignoring the cap and treating *z*_k(*X*) given *x̃* and *Y* = 0 as normal with mean *m*_k(*x̃*) and variance *v*_k(*x̃*), log *r*₀(*x̃*) = log κ_η + η*m*_k(*x̃*) + η²*v*_k(*x̃*)/2. The first-order prediction uses *m*_k only, which is exact when *v*_k does not depend on *x̃*. The second-order prediction adds the stratum contrast of *v*_k. Both are approximations. We estimate *m*_k by ridge regression of *z*_k on the input among recorded-negative training lesions, and *v*_k by ridge regression of the squared residual on a held-out half.

*Sampling without replacement.* A second design drew exactly *n* recorded negatives by the Gumbel top-*k* construction (Kool et al., 2019) with weights *w* = exp(η*z*). Its inclusion probabilities are not proportional to *w*. They are well approximated by 1 − exp(−λ*w*), with λ fixed by the expected count, which is proportional to *w* only while λ*w* is small. Section S6 reports the size of the deviation.

**Proof of Lemma 1.** Among verified lesions in stratum *t*, P(*D* = 1, *S* = 1 | *t*) = *p*ₜπₜ¹ and P(*D* = 0, *S* = 1 | *t*) = (1 − *p*ₜ)πₜ⁰. The odds of disease among verified lesions are therefore odds(*p*ₜ)·πₜ¹/πₜ⁰. Taking the ratio across strata gives OR_{D|S} = OR_D·*A*/*B*, the classical multiplicative selection bias factor with the four selection probabilities grouped into a malignant and a benign ratio. Logarithms give log OR_{D|S} = θ − δ. For θ > 0, the association keeps its sign exactly when δ < θ. It is weaker when 0 < δ < θ and stronger when δ < 0. It reverses when δ > θ, which is equivalent to *B*/*A* > OR_D. For θ < 0 the same argument gives reversal for δ < θ, attenuation for θ < δ < 0 and amplification for δ > 0. If *A* = *B*, then OR_{D|S} = OR_D. If θ = 0, then log OR_{D|S} = −δ.

**Relation between *g*_t and *g*(*x̃*).** By iterated expectations, *g*_t = E[P(*S* = 1 | *X̃*, *Y* = 0, *t*) | *Y* = 0, *t*]. This equals E[*g*(*X̃*) | *Y* = 0, *t*] when *t* is a function of *x̃*, as for the tabular input, or when *S* is independent of *t* given *X̃* and *Y* = 0. Otherwise the two can differ. For example, if *X̃* carries no information and verification rates among recorded negatives are 0.10, 0.20 and 0.30 in the three tertiles, then *g*(*X̃*) = 0.20 in every tertile while *B*_V = 3. *B*_V is also a ratio of means, whereas *G*_k of Proposition 1 contrasts means of log *g* over the evaluation population. We therefore use *B*_V only for Lemma 1 and the marginal analysis, and *G*_k only for the learner gap.

**Relation between *B* and *B*_V.** *g*ₜ = P(*S* = 1 | *Y* = 0, *t*) = (1 − *p*ₜ)πₜ⁰ / (1 − *p*ₜπₜ¹). Hence *B*_V = *B*·[(1 − *p*₁)(1 − *p*₀π₀¹)] / [(1 − *p*₀)(1 − *p*₁π₁¹)]. When disease is rare, each factor is close to one. Without the rare-disease approximation, *B* is partially identified. Write *V* = *S*(1 − *Y*), which is observed, and *v*ₜ = P(*V* = 1 | *t*) = (1 − *p*ₜ)πₜ⁰. Then *B* = [*v*₁/(1 − *p*₁)] / [*v*₀/(1 − *p*₀)], which is decreasing in *p*₀ and increasing in *p*₁. With each *p*ₜ in a known interval, the extremes of *B* are attained at the corners.

**Marginal risk ratio.** Because *Y* = *D*·*S*, P(*Y* = 1 | *t*) = *p*ₜπₜ¹, and hence RR_Y = RR_D·*A* exactly. The disease risk ratio exceeds one exactly when *A* < RR_Y, which is the most direct sensitivity parameter. A floor is one way to bound *A*. Because π₁¹ ≤ 1, the one-sided condition π₀¹ ≥ *s*_min on the lower stratum alone gives *A* ≤ 1/*s*_min. RR_D > 1 is then identified when RR_Y·*s*_min > 1. If the floor holds in both strata, the observed law bounds each disease risk by *p*_t ∈ [*q*_t, *p̄*_t], with *q*_t = P(*Y* = 1 | *t*), σ_t = P(*S* = 1 | *t*) and *p̄*_t = min{*q*_t/*s*_min, *q*_t + 1 − σ_t}. The second term holds because unverified malignant lesions cannot outnumber unverified lesions. For *q*₀, *q*₁ > 0 the sharp interval is RR_D ∈ [*q*₁/*p̄*₀, *p̄*₁/*q*₀]. When the floor branch *q*_t/*s*_min is active in both strata, it equals [*s*_min·RR_Y, RR_Y/*s*_min], that is *A* ∈ [*s*_min, 1/*s*_min]. Otherwise [*s*_min·RR_Y, RR_Y/*s*_min] is only an outer interval. For example, with *q*_t = 0.10 and σ_t = 0.95 in both strata and *s*_min = 0.5, the sharp interval is [2/3, 1.5], against [0.5, 2]. On ISIC-2024 the floor branch was active in both outer tertiles of every primary concept at every floor above 0.002 (script `vr47_counts_audit.py`), so the two intervals coincide there. RR_D > 1 needs only *q*₁ > *p̄*₀, which holds when RR_Y·*s*_min > 1 and uses only the lower-stratum floor. For two strata, log RR_D and θ = log OR_D have the same sign, because both are positive exactly when *p*₁ > *p*₀. The condition concerns the average malignant verification within a stratum. When the concept is a function of the learner's input, the pointwise floor of Proposition 2 implies it, but not conversely, so the marginal result rests on a weaker assumption. With covariate-adjusted in place of crude risk ratios, the same argument holds within each covariate cell, so we report the adjusted version only as a sensitivity analysis.

**Lemma 2 (logistic learners).** If logit *p*(*x*) = α + βᵀ*x* and π¹(*x*)/π⁰(*x*) = exp(ℓ₀ + λᵀ*x*) for a coefficient vector λ, the population-optimal verified-only logistic learner converges to β + λ. When verification depends on the outcome only, λ = 0. The converse does not hold: a constant malignant-to-benign verification ratio is sufficient for λ = 0.

**Proof of Lemma 2.** By Bayes' rule, odds(*D* = 1 | *x*, *S* = 1) = odds(*D* = 1 | *x*)·π¹(*x*)/π⁰(*x*). Under the log-linear likelihood ratio, logit P(*D* = 1 | *x*, *S* = 1) = α + ℓ₀ + (β + λ)ᵀ*x*. A logistic learner fitted on the verified population is then correctly specified, so its population optimum is β + λ. When λ = 0, the ratio π¹/π⁰ is constant and the slope is unchanged. Outcome-only verification is one such case, the setting of Prentice and Pyke for case-control sampling. A constant ratio does not imply outcome-only verification. For example, π⁰(*x*) = 0.1 + 0.1*x* and π¹ = 2π⁰ on *x* ∈ {0, 1} give λ = 0, although both depend on *x*. Outside the log-linear model none of these equalities is guaranteed.

**Proof of Proposition 2.** Because *Y* = *D*·*S*, *q*(*x̃*) = *p*(*x̃*)·*s*(*x̃*), so *p* = *q*/*s*. With *s* ∈ [*s*_min, 1], *p* ∈ [*q*, *q*/*s*_min]. In addition, P(*D* = 1, *S* = 0 | *x̃*) ≤ P(*S* = 0 | *x̃*), so *p* ≤ *q* + 1 − σ. Hence *p*(*x̃*) ∈ [*q*(*x̃*), *p̄*(*x̃*)]. The upper limit *q* + 1 − σ holds without any floor; the floor tightens it to *q*/*s*_min. Setting *s* = 1 attains the lower endpoint. Setting *s* = *s*_min attains the upper endpoint when *q*/*s*_min binds, and assigning every unverified lesion with input *x̃* to disease attains it otherwise. Both choices are compatible with the observed distribution of (*X̃*, *t*, *S*, *Y*). The choice can differ across values of *x̃*, but not across lesions that share *x̃*, because *p* is a function of *x̃*. By iterated expectations, E[*f*(*X̃*) | *t* = *j*] = E[*f*(*X̃*)·P(*t* = *j* | *X̃*)]/P(*t* = *j*), so ψ_k = E[*a*_k(*X̃*) logit *p*(*X̃*)] with *a*_k as in the main text. This is increasing in logit *p*(*x̃*) where *a*_k(*x̃*) > 0 and decreasing where *a*_k(*x̃*) < 0. Its smallest value, ψ_L, takes the lower endpoint where *a*_k > 0 and the upper endpoint where *a*_k < 0. Its largest value does the reverse. Where *a*_k(*x̃*) = 0 the endpoint chosen does not affect ψ_k. Every value between the two extremes is attained by a continuous choice of *s*. The interval is thus sharp relative to the nonparametric model of the observed data that satisfies *Y* = *D*·*S* and the pointwise floor. Any further restriction on *s*(*x̃*), such as smoothness, monotonicity or links across inputs, can only narrow it. If *t* is a function of *x̃*, then *a*_k > 0 exactly on the upper stratum and *a*_k < 0 exactly on the lower one, which gives the stratum formulas. In general, the stratum formula for the lower endpoint equals E[*a*_k⁺ logit *q* − *a*_k⁻ logit *p̄*] minus a nonnegative term. It assigns logit *p̄* ≥ logit *q* with weight P(*t* = 0 | *x̃*)/P(*t* = 0) also where *a*_k > 0, and logit *q* with weight P(*t* = 1 | *x̃*)/P(*t* = 1) also where *a*_k < 0. It is therefore a valid lower bound, and the same argument covers the upper endpoint. Where *q*/*s*_min binds, logit *p̄* = logit *q* − log *s*_min + log[(1 − *q*)/(1 − *q*/*s*_min)], and the last term is positive. For small *q* it is close to zero, which gives ψ_L ≈ Δ_q + log *s*_min for the stratum formula and *s*∗ ≈ exp(−Δ_q). Because the dropped term is positive, the approximation overstates ψ_L and can understate *s*∗. With *q*₀ = 0.01, *q*₁ = 0.02, σ = 0.1 and *s*_min = 0.497, the approximate floor 0.495 is exceeded, yet ψ_L = logit 0.02 − logit 0.02012 < 0; the exact floor is *q*₀/*q*₁ = 0.5. Every floor in the paper is computed from the exact bound. The proposition assumes that *q* and *p̄* lie in (0, 1) and that logit *q* and logit *p̄* are integrable within each outer tertile, so that ψ_k, ψ_L and ψ_U are finite. The population tipping floor is *s*∗ = inf{*s* : ψ_L(*s*) > 0}, with *s*∗ undefined when no floor up to one gives ψ_L > 0. Because ψ_L increases in *s*, a positive sign is identified for every floor above *s*∗; at *s*∗ itself ψ_L can equal zero, as in the example. On ISIC-2024 we report the smallest point of a grid of step 0.01 with ψ_L > 0, which lies within 0.01 above *s*∗ of the plug-in curve. For image features the outer set gives a sufficient floor, which can exceed the floor of the sharp set. The lower endpoint uses *p̄* only where *a*_k < 0, or, for the stratum formula, only on inputs in the lower stratum. Without a floor elsewhere, the smallest value of logit *p* there is still logit *q*, attained at *s* = 1. Identifying a positive sign thus needs the floor only on that part of the input space. The same holds for ψ_U with the roles reversed.

**Estimated weights.** The sharp set needs the sign of *a*_k, which for image features is a further nuisance. A plug-in with an estimate â_k whose sign is wrong on part of the input space no longer bounds ψ_k from below, because it assigns logit *q* where the true weight is negative. We have not established the finite-sample validity of such a plug-in. Sharpness is a property of the population set with the true *a*_k. The stratum formula needs no such estimate and remains a valid outer bound, so it is the one we report for the image-feature target. Script `vr42_sharp_weighted.py` checks both statements on a known population and compares the two on ISIC-2024 (Section S7). Table {{T:theory42}} uses a one-dimensional input on a grid and a concept equal to the input plus Gaussian noise of the stated scale. Disease and malignant verification are known functions of the input. With no noise the two sets coincide. With noise the stratum formula is wider, both sets contain the true ψ, and the sharp set is attained. Reversing the sign of the weight moves the lower endpoint above the true ψ.

**TABLE {{T:theory42}}. Stratum formula and sharp set on a known population.**

{{TABLE:theory42}}

## S2. Validation of the theory

Table {{T:theory}} summarizes the checks.

In Table {{T:theory}}, identity errors are maximum absolute errors on the log odds scale. Agreement is the fraction of cases in which the observed region or reversal matches the statement.

**TABLE {{T:theory}}. Validation of the identities.**

{{TABLE:theory}}

Script `vr10_symbolic.py` checks each statement with sympy (Meurer et al., 2017) on a declared positive domain. Table {{T:symbolic}} lists the statements. All were verified. For the attenuation sets, sympy does not reduce the intersection when θ is symbolic, so they were confirmed at θ ∈ {0.1, 1, 7/3, 50}.

**TABLE {{T:symbolic}}. Symbolic checks.**

| Check | Statement | Result |
| --- | --- | --- |
| P1.1 | logit *f*₀ − logit *f*₂ = log *g* | verified |
| P1.2 | a constant added to both strata cancels in Δ^Bayes | verified |
| P1.3 | *r*₀ ∝ exp(η*z*) shifts logit *f*₂ by −η*z* plus a constant | verified |
| P1.4 | with *X* only partly observed through *x̃*, odds(*Y* \| *x̃*, *R* = 1) = odds(*Y* \| *x̃*)·*r*₁ / E[*r*₀(*X*) \| *x̃*, *Y* = 0] | verified |
| L1.1 | OR_{D\|S} = OR_D·*A*/*B* | verified |
| L1.2 | log form of L1.1 | verified |
| L1.3 | θ > 0: reversal set is δ ∈ (θ, ∞) | verified |
| L1.4 | θ > 0: attenuation set is δ ∈ (0, θ) | verified at four values of θ |
| L1.5 | θ > 0: amplification set is δ ∈ (−∞, 0) | verified |
| L1.6 | δ > θ if and only if *B*/*A* > OR_D | verified |
| L1.7 | θ < 0: reversal set is δ ∈ (−∞, θ) | verified |
| L1.8 | θ < 0: attenuation set is δ ∈ (θ, 0) | verified at four values of θ |
| L1.9 | θ < 0: amplification set is δ ∈ (0, ∞) | verified |
| L1.10 | *A* = *B* gives no distortion | verified |
| L1.11 | θ = 0 gives OR_{D\|S} = *A*/*B* | verified |
| L1.12 | RR_Y = RR_D·*A* | verified |
| L2.1 | odds(*D* \| *x*, *S* = 1) = odds(*D* \| *x*)·π¹/π⁰ | verified |
| L2.2 | verified-only slope equals β + λ under the log-linear ratio | verified |
| L2.3 | logit *P* − log *P* = −log(1 − *P*) | verified |
| L2.4 | −log(1 − *P*) ≤ *P*/(1 − *P*) | verified |

Thirty unit tests in `test_reversal_theory.py` pass. They cover Proposition 1 on 5,000 random models, its general form with appearance only partly observed on 2,000 random models, its stratum form, and the selection-dose shift. They cover every region and boundary of Lemma 1 for both signs of θ, and the identity on 5,000 random models. They also cover the cases *A* = *B*, θ = 0 and *A* = 1. They confirm that a zero verification probability is not reported silently. They check Lemma 2 at four settings, the interval for *B* and the bootstrap statistics. They also confirm that the identified set of Proposition 2 contains ψ and that its lower endpoint is attained.

**Simulation of Proposition 2.** Script `vr23_psi_sim.py` draws populations of 300,000 lesions with prevalence 0.5 percent, five concepts and three noise features. Malignant verification is π¹(*x*) = *s*_true + (1 − *s*_true)·expit(*w*₁ᵀ*c*), with true floors 0.5, 0.7 and 0.9. Benign lesions are flagged and verified with probabilities that depend on appearance, with a mean verification rate of 0.3 percent. Concepts are independent or have pairwise correlation 0.5. Nuisance models are fitted on 200,000 lesions and the sets are evaluated on the other 100,000, at assumed floors 0.3, 0.5, 0.7 and 0.9. The oracle set uses the true *q* and σ, and the plug-in set uses gradient-boosting estimates of both. Table {{T:psisim}} reports how often each set contained the true ψ and how often it certified a sign. A set certifies a sign when it excludes zero; the term describes the set, not a statistical guarantee. No false-sign certification occurred in this design. Containment fell when the assumed floor exceeded the true one, as the proposition implies.

In Table {{T:psisim}}, entries are fractions over settings and concepts.

**TABLE {{T:psisim}}. Proposition 2 in simulation.**

{{TABLE:psisim}}

**Stress test of Proposition 2.** Script `vr28_psi_stress.py` searches for the failure that matters most, a certified sign opposite to the true one. It uses true floors of 0.1, 0.3, 0.5 and 0.9 and assumed floors from 0.1 to 0.9, so that many settings overstate the floor. Besides the design above, an adversarial design sets the disease effects of four concepts to 0, −0.3, 0.05 and −0.2. Malignant verification rises steeply with these concepts, so the recorded label has a positive contrast while ψ is zero or negative. Three constructions are compared: the oracle set, a gradient-boosting plug-in, and a two-layer perceptron trained for 30 epochs and Platt-calibrated on a held-out validation split, which mirrors the analysis of ISIC-2024. Table {{T:psistress}} reports the results. With a valid floor, no construction certified the wrong sign in either design. With an overstated floor in the adversarial design, all three did in some settings, mostly when the true floor was 0.1 and the assumed one was 0.5 or more. Several of these settings have ψ close to zero, and the others have ψ of −0.44 and −0.65. A valid floor is therefore necessary for a learner-scale disease-relative conclusion, and the simulation offers no protection against an overstated one.

In Table {{T:psistress}}, entries are fractions over settings and concepts, and ψ ≤ 0 counts settings whose true target is not positive.

**TABLE {{T:psistress}}. Stress test of Proposition 2.**

{{TABLE:psistress}}

**Semi-synthetic test on the ISIC-2024 inputs.** The simulations above use eight synthetic features and do not reproduce the rare-event and support structure of ISIC-2024. Script `vr34_semisynth.py` keeps the real learner inputs and the real patient split. It also keeps three functions of the input, estimated once from the real data: a calibrated risk logit, a ridge prediction of color variegation, and the verification propensity among recorded negatives. The risk logit comes from the perceptron M0 in one design and from gradient boosting in a second, so that neither estimator of *q* is favored by the data-generating process. Disease is drawn with logit *p*(*x̃*) = α + ℓ(*x̃*) + *b*·*m*(*x̃*). Here ℓ is the standardized risk logit and *m* the standardized color prediction. The intercept α sets the prevalence to 0.2 or 0.5 percent, and *b* sets the true ψ for color variegation to about 0.6, 0 or −0.3. Malignant verification is *s*(*x̃*) = *s*_true + (1 − *s*_true)·expit(−1 + 1.5*m*(*x̃*)), with *s*_true of 0.5 or 0.8, so it rises with the concept, and benign verification is the real propensity. Recorded labels follow *Y* = *D*·*S*. For each of 24 settings per family and design, the same three estimates of *q* as in Section S6 were fitted to the simulated labels. The plug-in set was then evaluated at the true floor against the true ψ. The set uses the stratum formula, which is sharp for the tabular input and an outer set for image features. The containment rate is the share of settings in which the plug-in set contained the true ψ. It describes the plug-in set, not the coverage of a confidence procedure. Low containment means that the plug-in set was often displaced from the true target, mostly downward. No wrong sign was observed under the stated criterion. The shift was not always downward: in five cells with a negative target the lower endpoint was 0.10 to 0.23 above the oracle one, with containment of 0.50 to 0.75, and no sign was reported in those cells. M2 and ĝ, a perceptron or gradient boosting, were also fitted. We recorded the lesion-level slope of the calibrated gap on log ĝ, which the population identity fixes at one. Table {{T:semi34}} reports the results.

In Table {{T:semi34}}, disease risk is built from the perceptron or from gradient boosting on the ISIC-2024 inputs. A wrong sign counts a certified sign opposite to a true ψ beyond ±0.05.

**TABLE {{T:semi34}}. Semi-synthetic test of the plug-in set at the true floor.**

{{TABLE:semi34}}

{{N:semi_bridge}}

In Table {{T:semibridge34}}, the gap is calibrated and regressed on log ĝ in designs where Proposition 1 holds exactly.

**TABLE {{T:semibridge34}}. Lesion-level slope of the learner gap in the semi-synthetic designs.**

{{TABLE:semibridge34}}

**Local violation of the floor.** The designs above satisfy the assumed floor everywhere. Running `vr34_semisynth.py` with the argument `local` keeps the perceptron design with one change. Malignant verification is set to 0.20 among the 30 percent of lesions with the lowest true overall verification propensity. This is below the assumed floor of 0.5 or 0.8, which still holds elsewhere. This mimics a floor that looks reasonable on average but fails where verification is rarest. The oracle set uses the true *q* and σ at the assumed floor, so its errors come from the assumption alone. Table {{T:local34}} reports containment and certification over the 24 settings per family.

In Table {{T:local34}}, sets are evaluated at the assumed floor. A wrong sign counts a certified sign opposite to a true ψ beyond ±0.05.

**TABLE {{T:local34}}. Semi-synthetic test with the floor violated where verification is rarest.**

{{TABLE:local34}}

## S3. Phase diagram design

The concept *c* is evaluated on a grid of 161 points on [−4, 4] with normal weights, and the hidden severity *h* on 16 Gauss-Hermite nodes. Every quantity is an exact population expectation. The disease model is logit P(*D* | *c*, *h*) = α + β*c* + γ*h*, with α solved for the target prevalence. Benign verification is π⁰(*c*) = expit(*b*₀ + *b*′*c*), with *b*₀ solved for a mean of 0.3 percent. Malignant verification is π¹(*c*, *h*) = *s*_lo + (1 − *s*_lo)·expit(*ac* + γ*h*). The grid crosses β ∈ {0.25, 0.5, 1}, *b*′ ∈ {0, 0.5, 1, 1.5, 2, 3}, *s*_lo ∈ {0.5, 0.7, 0.9}, *a* ∈ {−1, 0, 1}, prevalence ∈ {0.002, 0.02} and γ ∈ {0, 1}. Tertile cut points are ±0.4307. Population-optimal logistic learners are fitted by Newton iterations on the weighted grid.

With no appearance-driven biopsy of benign lesions (*b*′ = 0), no population reversed. At *b*′ = 1, 84 percent did, and at *b*′ of 1.5 or more, all did. The recorded-label learner reversed in none of the 648 populations. The finite-sample check draws 300,000 lesions at each of 24 random grid points and trains M0 and M2 perceptrons on *c* plus two noise features. In the 15 populations predicted to reverse, the M2 contrast ranged from −4.60 to −0.23 and the M0 contrast from +0.51 to +2.85. In the 9 predicted not to reverse, both contrasts were positive, with M2 at least +0.11.

Whether the population-optimal verified-only learner reversed agreed with Lemma 1 in 647 of 648 populations, and the remaining population sits on the boundary (Fig. S1).

In Fig. S1, each point is one of 648 simulated populations, shaded by the regions of Lemma 1. The circled point is the single population where the learner disagrees with the region. The figure file is `figures/figS1_phase_diagram.png`.

**Fig. S1.** Reversal boundary in simulation.

## S4. Sensitivity to the concept definition

Seven definitions were applied to the same predictions. Four split the concept: tertile (primary), quartile, quintile and median splits. Three are slopes of the logit: an ordinary least squares slope on the standardized concept, the same slope after removing each patient's mean, and a slope on the concept's cohort percentile rank. Table {{T:definitions}} reports the three primary concepts. Color variegation and lesion-skin contrast keep opposite point-estimate signs under every definition in every family; for color the within-patient slope shrinks the verified-only contrast but keeps its sign. These are point estimates, and the fine-tuned M0 contrast of lesion-skin contrast has a 95 percent interval that includes zero. Size keeps the reversal under every split-based definition, but not under slope-based definitions. With tabular and image features its verified-only slope is between −0.06 and +0.01, and it is positive within patients with image features. In the linear-probe and fine-tuned families it is positive. So the size reversal is carried by the outer tertiles. This is a sensitivity of the estimand, not a failure of the primary inference, which is defined on tertiles.

In Table {{T:definitions}}, each cell gives Δ_M0 / Δ_M2, and values condition on the fitted models.

**TABLE {{T:definitions}}. Learned contrasts under seven concept definitions.**

{{TABLE:definitions}}

## S5. Learner settings and localization of the reversal

Under common-readout and head-swap diagnostics on the fine-tuned encoders, the training regime of the decision head explained a larger change in the learned contrast than the origin of the encoder. The diagnostics did not support a general representation-level reversal, and they cannot exclude representation-level contributions.

Table {{T:settings}} lists the training settings.

In Table {{T:settings}}, M0 and M2 share every setting except the lesions: M0 trains, early-stops and is calibrated on all training and validation lesions, M2 on the verified ones only.

**TABLE {{T:settings}}. Training settings of the learner families.**

| Setting | Frozen-feature perceptrons | Linear probe | Fine-tuned ResNet-50 |
| --- | --- | --- | --- |
| Input | 47 tabular features, or 2,048 frozen ImageNet features | 128 × 128 tile | 128 × 128 tile |
| Architecture | two hidden layers of 128 (tabular) or 256 (image), GELU, dropout 0.1 | frozen ResNet-50, linear head | ResNet-50, linear head |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.0001 | AdamW, head 0.001 | AdamW, backbone 0.0001, head 0.001, one-cycle |
| Budget | up to 30 epochs, early stopping on validation loss | 2,000 steps of 64 | 4,000 steps of 64 |
| Validation set | all validation lesions for M0; verified validation lesions for M2 | M0: all validation malignant lesions and 10,000 sampled recorded negatives; M2: verified validation lesions | same as linear probe |
| Checkpoint | lowest validation loss | highest validation AUROC, evaluated every 500 steps | same as linear probe |
| Output for Δ | logit of the predicted probability clipped to [10⁻⁷, 1 − 10⁻⁷] | pre-sigmoid logit, no clipping | same as linear probe |
| Evaluation set | all 78,625 test lesions, 209 patients | uniform sample of 30,000 test lesions, 207 patients | same as linear probe |
| Calibration | Platt on the validation set, for plug-in and gap analyses only | none | none |
| Sampling | natural | positive fraction at least 10 percent | positive fraction at least 10 percent |
| Augmentation | none | flips, 90-degree rotations, crops; no color change | same |
| Seeds | 0, 1, 2 | 0, 1, 2 | 0, 1, 2; seeds 3 and 4 added for the head swap |

**Inputs and units.** Table {{T:manifest}} lists the inputs. Each ISIC-2024 image crop is one lesion and has equal weight in every contrast. Patients are the unit of every split and resample. Tertile cut points are computed once on the whole cohort, with the lower tertile at or below the 1/3 quantile and the upper at or above the 2/3 quantile, and stay fixed in every resample. Each outer tertile holds 133,687 lesions, and no concept value is missing. The three splits hold 624, 209 and 209 patients, 241,485, 80,949 and 78,625 lesions, 622, 231 and 215 verified lesions, and 222, 92 and 79 recorded malignancies. The analyses ran with PyTorch 2.10, torchvision 0.25, scikit-learn 1.7.2, NumPy 2.3.5 and pandas 2.3.3. The frozen-feature perceptrons use batches of 4,096 lesions, or one twentieth of a smaller training set with at least 64, early stopping with patience 5, and single precision.

**TABLE {{T:manifest}}. Inputs of the learners.**

| Input | Columns | Processing |
| --- | --- | --- |
| Concepts | color variegation `tbp_lv_norm_color`, size `tbp_lv_areaMM2` (mm²), lesion-skin contrast `tbp_lv_deltaLBnorm`, asymmetry `tbp_lv_symm_2axis`, border irregularity `tbp_lv_norm_border` | z-scored on the whole cohort; part of the tabular input |
| Other measurements of the total-body photography system | `tbp_lv_A`, `Aext`, `B`, `Bext`, `C`, `Cext`, `H`, `Hext`, `L`, `Lext`, `area_perim_ratio`, `color_std_mean`, `deltaA`, `deltaB`, `deltaL`, `deltaLB`, `eccentricity`, `minorAxisMM`, `nevi_confidence`, `perimeterMM`, `radial_color_std_max`, `stdL`, `stdLExt`, `symm_2axis_angle` (each with the prefix `tbp_lv_`) | z-scored on the whole cohort without labels; missing values set to 0 |
| Covariates | age, sex, anatomical site, acquisition site, skin-tone quartile of the individual typology angle | age centered at 50 and scaled by 10, median-imputed, with a missing indicator; categories one-hot with the first level dropped; 18 columns |
| Excluded | patient and lesion identifiers, the lesion identifier that defines *F*, 3D coordinates, location strings, the lesion-detection confidence, and every diagnosis or histopathology field | not used by any learner |
| Image features | 2,048 pooled features of ResNet-50 with ImageNet weights (torchvision IMAGENET1K_V2) | crop resized bilinearly to 224 × 224, normalized with ImageNet means, stored in half precision, each feature z-scored on the whole cohort |

Test AUROC for the recorded label was 0.941 under M0 and 0.708 under M2 with tabular features. With frozen image features it was 0.815 and 0.689. With fine-tuning it was 0.912 to 0.918 and 0.731 to 0.752 for seeds 0 to 2.

**Common readout.** On each encoder's penultimate features for the shared probe set, we fitted a ridge regression to a target that is the same for both encoders. The first target is the logit of the recorded-label risk *q*(*x*), from a tabular gradient-boosting model fitted out of fold. The second is the logit of verified-lesion risk *q*/σ. We also fitted ridge directions for each concept. The alignment is the cosine between the readout direction and the concept direction. A representation reversal would appear as opposite signs of this alignment between the M0 and M2 encoders. Table {{T:repr}} reports these diagnostics for seeds 0 to 2. For the layer-3 shift, a ridge regression of each standardized concept on the spatially pooled layer-3 activations gives a concept direction. The activations are shifted by plus and minus one standard deviation of the concept along that direction at every spatial location, and the change in the output logit is passed through layer 4 and the head.

In Table {{T:repr}}, values are averaged over seeds 0 to 2, and the last column counts the seeds in which M0 and M2 have opposite signs.

**TABLE {{T:repr}}. Representation diagnostics after fine-tuning.**

{{TABLE:repr}}

Concept decodability was the same under both regimes. Cross-validated R² ranged from 0.44 to 0.48 for color variegation, 0.73 to 0.77 for lesion-skin contrast, 0.33 to 0.36 for asymmetry and 0.35 to 0.38 for border irregularity. Linear centered kernel alignment between M0 and M2 features was 0.79, 0.22 and 0.74 across seeds 0 to 2. Alignment between two seeds of M0 was 0.39, against 0.98 for two seeds of M2. Fine-tuning under M0 is thus much less stable across seeds, and we treat alignment measures only as secondary diagnostics. Fig. S2 plots them per seed.

**Head swap.** For each of five seeds, both fine-tuned encoders were frozen. New linear heads were trained on their penultimate features under both regimes, with the linear-probe recipe of Table {{T:settings}}, and all four pairings were evaluated on the same probe set. The head effect is the mean contrast under M2 heads minus that under M0 heads, averaged over the two encoders. The encoder effect is the mean contrast on the M2 encoder minus that on the M0 encoder, averaged over the two heads. Retrained heads are new models, so these effects summarize the diagnostic and are not a unique causal decomposition of the original networks. Table {{T:headswap}} reports the mean contrast of each pairing, both effects with their range over seeds, and the number of seeds in which the sign followed the head. Heads retrained under M0 learned contrasts close to zero, so the sign criterion is fragile, and the effect sizes are the informative summary.

In Table {{T:headswap}}, contrasts and effects are means over five seeds, and brackets give the range over seeds.

**TABLE {{T:headswap}}. Head swap on fine-tuned encoders.**

{{TABLE:headswap}}

In Fig. S2, the panels show the decision contrast, the layer-3 intervention and the two common readouts. The figure file is `figures/figS2_finetune.png`.

**Fig. S2.** Fine-tuned diagnostics per seed.

## S6. Joint bootstrap and tests of alternative explanations

**Joint bootstrap.** For the two primary families, each of {{N:B19}} replicates resampled training patients with replacement and refitted M0 and M2. Early stopping used the validation loss on all validation lesions for M0 and on the verified validation lesions for M2, the same filter that defines each training set. Each replicate then Platt-calibrated M0 on the full validation set and M2 on the verified validation lesions, resampled test patients and recomputed Δ, the plug-in set of Proposition 2 and the support-restricted contrasts. Calibrated contrasts serve only the plug-in analyses and the diagnostics; every robustness label uses raw contrasts. Replicate *b* resampled patients with generator seed 50,000 + *b* and trained both learners with seed *b*. Optimizer randomness thus varies independently across replicates, and the bootstrap distribution mixes patient resampling with training randomness. These {{N:B19}} replicates alone describe the randomized one-seed training procedure and serve as a check. Point estimates average the three original fits with seeds 0 to 2. They target the same expectation over seeds with less variance but do not have the sampling distribution of the one-seed statistic. Script `vr39_seed_variance.py` therefore refitted each of the {{N:B_primary}} replicates per family with two further seeds, *b* + 100,000 and *b* + 200,000, on the same resampled patients. Script `vr43_three_seed_primary.py` then bootstraps the three-seed average, which is the statistic of the point estimate. These three-seed replicates give the primary intervals and labels of Table 1. Table {{T:seed39}} splits the variance of the primary contrasts and compares the Bonferroni intervals and labels of the one-seed and three-seed statistics on the same replicates. Table {{T:joint19}} gives 95 percent percentile intervals and Bonferroni-adjusted percentile intervals at level 1 − 0.05/6, for raw and calibrated contrasts. {{N:platt}} Calibration rescales each learner's logit by a positive factor, so it does not change the sign of Δ.

**Output scale.** Every learned contrast of the two frozen-feature families uses the logit of the predicted probability after clipping to [10⁻⁷, 1 − 10⁻⁷], so logits are capped at ±16.12. The linear-probe and fine-tuned families use pre-sigmoid logits (Table {{T:settings}}). Script `vr50_logit_clip.py` refitted M0 and M2 with seeds 0 to 2, which reproduced the original point estimates, and compared this measure with the direct pre-sigmoid output (Table {{T:clip50}}). The cap bound mainly for M0 with image features, whose direct logits reached about −51; with tabular features it bound for 0.08 percent of test lesions. There it raised the lowest logits, more often in the lower tertile of the three primary concepts, which shrank every contrast toward zero without changing its sign. Script `vr51_direct_indeterminate.py` then refitted the primary three-seed bootstrap on the same 5,000 replicates: the same patient resamples and seeds as Table 1, which it reproduced to within 1.4 × 10⁻⁵ over every stored contrast of both families. With direct logits, every Bonferroni label was unchanged and reproduced in every Monte Carlo resample where it was positive (Table {{T:vr51}}). The intervals of Δ_M0 widened, for color variegation with image features to [0.15, 3.43] against [0.26, 2.33], and stayed above zero. The plug-in *q̂* of Proposition 2 is a Platt transform of the same clipped logits. The dose learners train on about 36 percent malignant lesions, like M2, whose outputs never reached the cap; their lesion-level outputs were not stored, so this was not checked directly.

In Table {{T:clip50}}, values average the refits with seeds 0 to 2 on the test population.

**TABLE {{T:clip50}}. Learned contrasts with the clipped and the direct output logit.**

{{TABLE:clip50}}

In Table {{T:vr51}}, intervals come from the 5,000 three-seed replicates of Table 1, refitted; the direct version is computed for image features only, where the cap binds, and reproduction is over 2,000 resamples of the replicates.

**TABLE {{T:vr51}}. Primary bootstrap with direct logits and with M2 trained without indeterminate biopsies.**

{{TABLE:vr51}}

**Populations and intervals.** Every analysis conditions on unsupervised preprocessing fitted once on the whole cohort: the z-scores and median imputation of Table {{T:manifest}} and the tertile cut points. No label enters them. The principal components used by the image gradient-boosting model are fitted once on original training patients. In each bootstrap replicate *q̂* is refitted, while σ̂ is the fixed out-of-fold estimate. Results therefore describe this cohort-level protocol, not a pipeline that refits preprocessing on training patients only. Table {{T:targets}} lists, for each quantity, the population, the randomness that its interval resamples or averages, what is held fixed, and how the interval should be read.

**TABLE {{T:targets}}. Populations, sources of randomness and meaning of the intervals.**

| Quantity | Population | Resampled or averaged | Held fixed | Meaning of the interval |
| --- | --- | --- | --- | --- |
| Δ_M0, Δ_M2, Table 1 | ISIC-2024 test population, 209 patients | training and test patients; three seeds | validation set; patient split | sensitivity to patient resampling and training randomness; a population interval only if patients are a sample from a population like the observed sites |
| Δ_M0, Δ_M2, Fig. 2 | test population | test patients | fitted learners | conditional on the fitted learners |
| Gap and lesion-level slope, Fig. 3(b) | test population | training and test patients; one seed | validation set | diagnostic of Proposition 1, not a test of it |
| τ_j, dose experiment | test population | ω: inclusion uniforms, malignant draw, validation draw and seed, 20 replicates | cohort, patient split, test population | effect of the implemented rule on this pipeline; no patient sampling |
| Plug-in ψ_L and *s*∗, Table 2 | test population | training, validation and test patients; one seed | estimator of *q*; σ̂ | sampling variability of the plug-in for one nuisance specification; not a confidence bound |
| RR_Y and verified-only log OR, Section IV-G | whole cohort | patient clusters, not stratified by site | tertile cut points; standardization weights *w*_g | resampling stability of the marginal sign under an assumed floor |

In Table {{T:seed39}}, each of the {{N:B_primary}} joint replicates per family has three seeds, and labels follow the primary rule on these replicates.

**TABLE {{T:seed39}}. Seed and patient-resampling variability of the primary contrasts.**

{{TABLE:seed39}}

In Table {{T:joint19}}, the {{N:B19}} replicates per family target the randomized one-seed training procedure, the primary three-seed intervals are in Table 1, and robustness is judged on the raw contrasts.

**TABLE {{T:joint19}}. One-seed joint bootstrap intervals for the frozen-feature families.**

{{TABLE:joint19}}

**Validation resampled, three estimates of *q*.** Early stopping and Platt calibration use the validation set, and M2 is calibrated on its verified lesions only. Script `vr30_q_bootstrap.py` repeated the analysis in {{N:B30s}} replicates per family that resampled training, validation and test patients independently within each split. In every replicate it computed the plug-in ψ_L for three estimates of *q*. Two are the perceptron M0 with Platt or with isotonic calibration. The third is gradient boosting of *Y* on the tabular features, or on the first 64 principal components of the image features, with Platt calibration. Table {{T:full30}} compares the robustness labels. For each estimate, it also reports the plug-in tipping floor *s*∗ on the seed-averaged curve, its 95th percentile over replicates and the bootstrap stability threshold. The last columns give *s*∗ within the supported region and the number of replicates in which ψ_L(*s*) > 0 and Δ_M2 < 0 held together. The bootstrap stability threshold is the smallest floor at which ψ_L(*s*) > 0 in at least 95 percent of replicates. Because ψ_L increases in s in every replicate, this threshold equals the 95th percentile of *s*∗ when replicates in which no floor up to one makes ψ_L positive are counted as above one. Both are computed this way. It describes sampling variability for one specification of *q*. It is not a confidence bound for ψ or for the identified set, and it does not account for the choice among estimates of *q*. Its Monte Carlo range is the 2.5th to 97.5th percentile of the same statistic over 500 resamples of the stored replicates.

In Table {{T:full30}}, training, validation and test patients are resampled for three estimates of *q*. In the point-floor columns, "none" means that the lower endpoint of the seed-averaged curve was not positive at any floor up to one. In the threshold column, it means that no floor reached the 95 percent rule, although some replicates can still have a positive lower endpoint.

**TABLE {{T:full30}}. Robustness and plug-in tipping floor with all three splits resampled.**

{{TABLE:full30}}

**Monte Carlo resolution with all splits resampled.** With 1,000 replicates, each Bonferroni tail holds about four replicates. Script `vr47_counts_audit.py` therefore resampled the stored replicates 2,000 times (Table {{T:mc47}}). Against the one-seed check with validation fixed, only the label of color variegation with image features changed, from Bonferroni to 95 percent only. Its lower Bonferroni endpoint of Δ_M0 was −0.01, and the new label was reproduced in 57 percent of resamples, so the change is unresolved. Each other label was reproduced in at least 99 percent. The labels of Table {{T:full30}} thus reflect both validation variability and the limited resolution of the tails.

In Table {{T:mc47}}, labels use one seed per replicate and 1,000 replicates per family, and reproduction is over 2,000 resamples of the stored replicates.

**TABLE {{T:mc47}}. Monte Carlo resolution of the labels with all three splits resampled.**

{{TABLE:mc47}}

**Subsampling.** Resampling patients with replacement and refitting networks with early stopping may not behave as the percentile bootstrap assumes. Script `vr36_subsample.py` drew half of the training patients and half of the test patients without replacement in 300 replicates per family, and refitted M0 and M2. It then formed descriptive percentile intervals of the m-out-of-n form (Politis et al., 1999), [θ̂ − Q_{1−a/2}, θ̂ − Q_{a/2}] for a = 0.05 and a = 0.05/6. Here θ̂ is the point estimate of Table 1, θ*_m the estimate from a half-sample, and Q_a the a-quantile of √(m/n)(θ*_m − θ̂) with m/n = 1/2. The rescaling assumes that the learned contrast converges at the root-n rate, which fitted networks with early stopping need not satisfy. It also omits the finite-sampling factor: for a sample mean, half-sampling without replacement has conditional variance (1 − m/n)S²/m, so the rescaled spread is half of S²/n even at the root-n rate. These intervals therefore carry no nominal 95 percent or Bonferroni coverage; Table 1 remains the inferential basis. A learner trained on half of the patients may also target a different contrast. We therefore use subsampling only to check whether cells that the bootstrap separated stay separated. It also separated tabular size and image lesion-skin contrast, which we do not count as robust. Table {{T:sub36}} reports the intervals.

In Table {{T:sub36}}, intervals are the descriptive rescaled half-sampling intervals defined above, from 300 half-samples per family.

**TABLE {{T:sub36}}. Primary comparisons under half-sampling of patients, descriptive intervals.**

{{TABLE:sub36}}

**Monte Carlo error.** The Bonferroni endpoints are the 0.42 and 99.58 percentiles of the replicates, so they rest on few replicates in each tail. For the primary three-seed bootstrap, script `vr43_three_seed_primary.py` resampled the {{N:B_primary}} stored three-seed replicates per family 2,000 times and recomputed the Bonferroni endpoints and the label in each resample. It reports the Monte Carlo standard error of each endpoint and the share of resamples that reproduce the label of Table 1. A label is called resolved at the Monte Carlo resolution when this share is at least 0.95. Table {{T:mc43}} gives the results. For the one-seed check, script `vr29_bootstrap_audit.py` applied the same procedure to the {{N:B19}} stored one-seed replicates. It also reports the quantiles of *s*∗ and the joint event above. Table {{T:mc29}} gives the results.

In Table {{T:mc43}}, the audit covers the replicates of Table 1 with 2,000 resamples of the stored replicates.

**TABLE {{T:mc43}}. Monte Carlo audit of the primary three-seed bootstrap.**

{{TABLE:mc43}}

In Table {{T:mc29}}, the table also gives the distribution of the plug-in tipping floor over the one-seed replicates.

**TABLE {{T:mc29}}. Monte Carlo error of the one-seed check.**

{{TABLE:mc29}}

**Summary of the tests of alternative explanations.** Table {{T:alt}} collects the tests of alternative explanations for the learner reversal; the paragraphs below report each in full.

In Table {{T:alt}}, size-matched intervals are 2.5 and 97.5 percentiles over 20 replicates, and each arm drew its own validation set. Dose rows use 320 expected recorded negatives and paired-replicate bootstrap intervals, and their ratios are descriptive. Predicted-shift rows use 10 replicates per arm. Lesion-level rows use 200 one-seed joint resamples, or 100 for the errors-in-variables diagnostic. Overlap rows are point estimates of the original fits, and the PAD-UFES-20 row uses 200 joint resamples.

**TABLE {{T:alt}}. Tests of alternative explanations for the learner reversal.**

{{TABLE:alt}}

**Size-matched control.** Every malignant training lesion is verified, so M0 and M2 differ in the size of the training set and in the source of the recorded negatives. In each of 20 replicates we drew 178 of the 222 malignant training lesions and 320 recorded negatives. The negatives came from one of three pools: all 241,263 recorded-negative training lesions, the 12,271 flagged but unverified ones, or the 400 biopsied ones, which include indeterminate histopathology. The malignant lesions were shared by the three arms within a replicate, and each arm drew its validation lesions from its own pool by the same rule. The arms therefore differ in the validation data used for early stopping as well as in the training negatives. Training used the same code, batch rule and early stopping as M2. Table {{T:size_matched}} reports the mean learned contrast on the test population, the 2.5 and 97.5 percentiles over replicates and the number of positive replicates.

**TABLE {{T:size_matched}}. Size-matched control, 20 replicates per arm.**

{{TABLE:size_matched}}

**Quantitative prediction for the arms.** Relative to the random arm, the flagged and verified arms select recorded negatives with *r*₀(*x*) proportional to the probability that a recorded negative with appearance *x* belongs to the arm's pool. Proposition 1 then predicts the shift in Δ from the random arm as minus the stratum contrast of log ĝ_arm. Here ĝ_arm(*x̃*) is that probability, estimated with the same perceptron and input as the learner. Script `vr21_dose_calibrated.py` reran the arms with 10 replicates and calibrated learners for this comparison and computed both shifts. Table {{T:arms21}} compares observed and predicted shifts for all five concepts.

**TABLE {{T:arms21}}. Observed and predicted shift of the learned contrast from the random arm, calibrated learners.**

{{TABLE:arms21}}

**Controlled selection dose.** Script `vr46_dose_fixed_val.py` gives the primary version. For each of the three primary concepts and both frozen-feature families, 20 replicates drew 178 malignant training lesions and a replicate-specific validation set. The validation set held malignant lesions and uniformly drawn recorded negatives. Both draws were shared by all doses of the replicate and were not selected by η. Each recorded-negative training lesion *i* received one uniform variable *u*_i per replicate. It entered training at dose η when *u*_i < ρ_η,i = min{1, κ_η exp(η*z*_k,i)}, for η ∈ {−1, −0.5, 0, 0.5, 1}, with κ_η set so that 320 are expected. These common random numbers couple the draws at different doses. Realized counts per fit ranged from 282 to 360 at 320 expected, from 933 to 1,068 at 1,000 and from 2,911 to 3,120 at 3,000. The optimizer seed was also shared, so a replicate fixes ω of Section III-B and the five fits are paired potential outcomes. Color variegation was repeated with 1,000 and 3,000 expected recorded negatives. Early stopping used the replicate-specific validation set. The outcome is the raw logit contrast. Platt calibration on the replicate-specific validation set targets the unselected distribution rather than the selected training distribution of Proposition 1. At η = 1, and at no other dose, its fitted slope was negative in 29 of 200 fits, which reverses the sign of a calibrated contrast. We report calibrated contrasts for completeness only. The prediction for the tabular learner is minus the stratum contrast of log ρ_η(*z*_k) on the test population, including the cap, which is the exact Bayes-optimal shift of the training distribution. For the image learner the prediction is the first-order input-aware form of Section S1. The mean contrast of the selected concept decreased at each step between the five doses in all six combinations, and single replicates did so in 50 to 90 percent of cases. The conditional mean *m*_k is estimated by ridge regression on recorded negatives of resampled training patients in each replicate. The estimate τ̂_k(1, −1) is the mean over replicates of the paired difference Δ_k(1, ω) − Δ_k(−1, ω), with a percentile bootstrap over the 20 replicates, 10,000 resamples; ratio intervals use 2,000. These intervals are conditional on the realized cohort, patient split and test population and summarize the paired pipeline randomness across replicates, including the validation-set draw used for early stopping. Table {{T:dose46}} reports all cells, and Table {{T:spill46}} compares observed and predicted slopes for all five concepts.

In Table {{T:dose46}}, the replicate-specific validation set is shared across doses, and replicates are paired. Slopes and τ̂(1, −1) are for the raw logit contrast of the selected concept. Ranges are 2.5 and 97.5 percentiles over replicates, and bracketed intervals are bootstrap intervals over the 20 replicates, conditional on the cohort, patient split and test population. The prediction is the Bayes-optimal training-distribution shift. Raw finite-network logits need not equal the Bayes-optimal log-odds of the selected training distribution, so the observed-over-predicted ratio is descriptive. It does not test the magnitude predicted by Proposition 1.

**TABLE {{T:dose46}}. Selection dose on training lesions only.**

{{TABLE:dose46}}

In Table {{T:spill46}}, entries are observed / predicted slopes of raw logit contrasts with 320 expected recorded negatives and the validation set shared across doses within each replicate.

**TABLE {{T:spill46}}. Dose slopes of all five concepts under the training-only dose.**

{{TABLE:spill46}}

**Joint training-and-validation dose.** Script `vr32_dose_nuisance.py` ran a second design in which validation negatives were drawn under the same rule as training negatives, without common random numbers. Each learner was Platt-calibrated on its own validation lesions. The dose then acts on training, early stopping and calibration together. This design estimates the effect of a joint selection rule for development data, not the training-selection effect, and its agreement in magnitude with Proposition 1 is only diagnostic. Its calibrated contrast crossed zero in the same six combinations, and script `vr44_dose_estimand.py` gives its paired effects. Table {{T:dose32}} reports it, Table {{T:tau44}} gives the paired effects and Table {{T:spill32}} the slopes of all five concepts. Script `vr26_dose_poisson.py` ran the same joint design with 10 replicates and the ridge prediction only, with the same pattern.

In Table {{T:dose32}}, contrasts are calibrated, there are 20 replicates, and ranges are 2.5 and 97.5 percentiles over replicates.

**TABLE {{T:dose32}}. Joint training-and-validation dose.**

{{TABLE:dose32}}

In Table {{T:tau44}}, the outcome is the calibrated contrast of the selected concept with 320 expected recorded negatives. Intervals are bootstrap intervals over the 20 replicates, conditional on the fixed cohort.

**TABLE {{T:tau44}}. Effect of the joint training-and-validation dose.**

{{TABLE:tau44}}

In Table {{T:spill32}}, entries are observed / predicted slopes with 320 expected recorded negatives.

**TABLE {{T:spill32}}. Dose slopes of all five concepts under the joint dose.**

{{TABLE:spill32}}

**Sampling without replacement.** Script `vr21_dose_calibrated.py` ran a joint design with exactly 320 recorded negatives drawn without replacement by the Gumbel top-*k* construction, for η up to 3. Its inclusion probabilities are not proportional to *w* (Section S1). Script `vr26_dose_poisson.py` computed Table {{T:gumbel26}}, which compares the nominal and actual expected number of upper-tertile lesions, checked by 300 Monte Carlo draws, and the predicted change in the contrast. For size, a few lesions with extreme area would have nominal probabilities above one, so the composition of the sample differs from the nominal one at η = 0.5. The predicted contrast changes by less than one percent in every case, because it averages the log inclusion probability of this design, written π^G, over the test population. Table {{T:dose21}} gives the results of this design, which also drew validation lesions by the same rule and agrees with Table {{T:dose32}}. Table {{T:ess21}} gives the effective sample size of its selection weights, which fell steeply for η > 1.

**TABLE {{T:gumbel26}}. Inclusion probabilities under sampling without replacement, recorded-negative training pool of 241,263 lesions, 320 draws.**

{{TABLE:gumbel26}}

In Table {{T:dose21}}, ranges are 2.5 and 97.5 percentiles over replicates.

**TABLE {{T:dose21}}. Selection dose under sampling without replacement.**

{{TABLE:dose21}}

In Table {{T:ess21}}, the effective sample size is Kish's (Kish, 1965), as a share of the 241,263 recorded-negative training lesions. Shares are the selection weight falling in the lower and upper tertiles.

**TABLE {{T:ess21}}. Support of the selection dose for color variegation.**

{{TABLE:ess21}}

**Proposition 1 at the lesion level.** Script `vr22_pointwise_bridge.py` ran 200 joint bootstrap replicates for each frozen-feature family. Each replicate resampled training patients and trained M0, M2 and a propensity model ĝ with the same perceptron and input. Each model was Platt-calibrated on validation lesions from its own population: all validation lesions for M0, verified ones for M2, and recorded negatives for ĝ, with *S* as the label. After resampling test patients, the lesion-level calibrated gap logit *f*₀ − logit *f*₂ was regressed on log ĝ, for which Proposition 1 predicts slope one. The concept-level observed gap was compared with the stratum contrast of log ĝ. Table {{T:pointwise22}} reports medians and 95 percent percentile intervals over replicates.

**TABLE {{T:pointwise22}}. Observed and predicted learner gap Δ_M0 − Δ_M2, calibrated learners, 200 joint resamples.**

{{TABLE:pointwise22}}

**Split-sample errors-in-variables diagnostic.** The regressor log ĝ is estimated. Under classical measurement error, a least-squares slope can be attenuated toward zero. Script `vr27_bridge_eiv.py` ran 100 joint replicates per family in which two propensity models, ĝ_A and ĝ_B, were fitted on disjoint halves of the resampled training patients. The slope of the calibrated gap on log ĝ_A was instrumented by log ĝ_B, averaged over both orders, The two halves split the unique patients of the resampled training set, so no patient enters both. The instrumented slope is consistent only if the estimation errors of the two halves are independent of each other, of the true log *g* and of the residual of the gap. Both halves share the validation set used for early stopping. M0, M2 and ĝ also use the same resampled patients, and biopsied recorded negatives enter both ĝ and M2, which can make their errors dependent in an unknown direction. The correlation of log ĝ_A and log ĝ_B estimates the reliability of a single estimate. Table {{T:eiv27}} reports the results. The instrumented point estimates exceed one. The lesion-level regression is an association diagnostic rather than a test that the slope equals one.

In Table {{T:eiv27}}, the slope is that of the learner gap on log ĝ, with 100 joint replicates per family.

**TABLE {{T:eiv27}}. Split-sample errors-in-variables diagnostic for the lesion-level slope.**

{{TABLE:eiv27}}

Models of the same class can also share systematic error. The instrumented slope is therefore a sensitivity diagnostic, not a correction that recovers the true slope. Script `vr33_bridge_alt.py` repeated the lesion-level analysis in 100 joint replicates per family, with a second model class for ĝ and on the supported test population. The second class is gradient boosting of *S* among recorded negatives. Table {{T:bridge33}} reports the results. Section S2 gives the slope that the same trained learners attain in a semi-synthetic design where the population identity holds exactly.

In Table {{T:bridge33}}, the gap is calibrated, slopes are shown on the full and supported test populations, and there are 100 joint replicates per family.

**TABLE {{T:bridge33}}. Lesion-level slope of the learner gap for two model classes of ĝ.**

{{TABLE:bridge33}}

**Verification pattern.** Among recorded negatives, verification was regressed on each concept split into its deviation from the patient's mean and the patient's mean, with sex, anatomical site, acquisition site and skin tone. Standard errors are clustered by patient. Table {{T:within}} reports the coefficients per standard deviation, to three decimals. The between-patient coefficients of asymmetry and border irregularity nearly coincide because the two measurements are highly correlated (Spearman 0.94).

**TABLE {{T:within}}. Within-patient and between-patient dependence of verification on appearance among recorded negatives.**

{{TABLE:within}}

**Acquisition sites.** Table {{T:sites}} reports, per acquisition site, the adjusted log odds ratio of verification among recorded negatives between the upper and lower tertiles. It comes from a logistic model with sex, anatomical site and skin tone. It approximates log *B*_V when verification is rare, but it is a logit coefficient and not the crude ratio *B*_V of Table {{T:marginal35}}. The table covers sites with at least ten biopsied recorded negatives in the two outer tertiles. Site 3 had 15 biopsied recorded negatives, all in the upper tertile, so its contrast is not estimable. Site 7 had 7 biopsied recorded negatives and falls below the threshold. Sites are numbered alphabetically by name in Tables {{T:sites}} and {{T:loso}}. Values are given to three decimals. Color variegation and lesion-skin contrast give similar values at Sites 4 and 5 because the two measurements are correlated (Spearman 0.65) and not because a value was reused. The fits use different lesions, for example 230 and 221 biopsied recorded negatives at Site 5. Table {{T:sitehet}} pools the estimable sites with a DerSimonian-Laird random-effects model (DerSimonian & Laird, 1986). τ² is the DerSimonian-Laird between-site variance. Table {{T:loso}} retrains M0 and M2, three seeds each, after removing one site's patients from training, validation and test. Both color variegation contrasts kept their signs for every left-out site in both families. Lesion-skin contrast with image features did not: its M0 contrast was −0.08 without Site 2 or Site 6.

**TABLE {{T:sites}}. Adjusted log odds ratio of verification among recorded negatives by acquisition site, with standard errors.**

{{TABLE:sites}}

**TABLE {{T:sitehet}}. Heterogeneity of the adjusted log odds ratio of verification across acquisition sites.**

{{TABLE:sitehet}}

**TABLE {{T:loso}}. Learned contrasts with one acquisition site left out, mean over seeds 0 to 2.**

{{TABLE:loso}}

**Overlap.** The common verification propensity σ̂ is a gradient-boosting model on the tabular features, used as one clinical support definition for both families and estimated without using the test fold. The thresholds are the 1st, 5th and 10th percentiles of σ̂ among verified lesions of the original training set. σ̂ and the thresholds are held fixed in every replicate, and each replicate restricts its resampled test lesions to those with σ̂ at or above a threshold. The intervals therefore condition on this support definition and do not reflect variation in it. Table {{T:support19}} reports the contrasts. Table {{T:overlap}} reports the full, support-restricted and flagged test populations for the original fits.

In Table {{T:support19}}, intervals come from the one-seed joint bootstrap with {{N:B19}} replicates per family, with σ̂ and the support thresholds fixed from the original training set.

**TABLE {{T:support19}}. Learned contrasts within the support of verified training lesions.**

{{TABLE:support19}}

In Table {{T:overlap}}, intervals are patient bootstrap intervals conditional on the fitted models.

**TABLE {{T:overlap}}. Learned contrasts on the full, support-restricted and flagged test populations.**

{{TABLE:overlap}}

**Other estimates of σ.** Support membership depends on the estimate of σ. Proposition 2 defines σ on each learner's input, so the perceptron trained on that input is the theory-matched definition. The tabular gradient-boosting estimate is a common clinical definition for both families and the only one with joint intervals. Script `vr37_overlap_sigma.py` repeated the overlap analysis with two further estimates: perceptrons trained on the tabular or on the image features, Platt-calibrated on validation data and averaged over seeds 0 to 2. It also reran the common estimate under the same procedure, with the original fits of M0 and M2 averaged over seeds 0 to 2 and 2,000 bootstrap resamples of test patients. These intervals condition on the fitted learners and are narrower than the joint intervals of Table {{T:support19}}. Table {{T:sigma37}} reports the share of test lesions retained and the share that changed membership relative to the common estimate.

**TABLE {{T:sigma37}}. Overlap analysis under three estimates of σ, test-patient bootstrap conditional on the fitted learners.**

{{TABLE:sigma37}}

**Composition of the supported population.** Restricting the test population to σ̂ ≥ *c* changes the joint distribution of the concepts within each tertile, so Δ can change even when the prediction function does not. Within each tertile, script `vr40_support_composition.py` reweights the supported lesions by the inverse of their estimated probability of being supported given the five standardized concepts. That probability comes from a logistic model fitted on the lesions of the tertile in the full test population, and the weights are truncated at their 99th percentile. The reweighted contrast compares supported lesions with the concept composition of the full population. Intervals come from 300 test-patient bootstrap resamples, refitting the weighting model, conditional on the fitted learners. Table {{T:comp40}} reports color variegation. The reweighting did not restore the full-population M0 contrast in any setting, and M2 stayed negative. Standardizing the supported lesions on the five measured concepts thus left the attenuation within support in place. This weighting model does not balance unmeasured appearance, patient or site variables. The analysis changes the evaluation population and does not isolate extrapolation by M2.

In Table {{T:comp40}}, reweighting targets the full-population concept composition within each tertile, and original fits are averaged over seeds 0 to 2.

**TABLE {{T:comp40}}. Supported contrasts of color variegation, raw and reweighted.**

{{TABLE:comp40}}

**Marginal and conditional contrasts.** The tertile contrast is marginal over the other concepts. The same script also balanced the two tertiles of each primary concept on the other four concepts by inverse probability weighting on the full test population. This is a different estimand, the contrast with the other concepts held at a common distribution. Table {{T:partial40}} shows that it differs substantially from the marginal contrast because the concepts are correlated. The learned contrasts in this paper are therefore not effects of one concept with the others held fixed.

In Table {{T:partial40}}, original fits are averaged over seeds 0 to 2.

**TABLE {{T:partial40}}. Marginal tertile contrasts and contrasts balanced on the other concepts.**

{{TABLE:partial40}}

**PAD-UFES-20.** The release holds 2,298 images of 1,641 lesions from 1,373 patients, and each learner was trained and evaluated on images. The association analysis of Fig. 4 uses eleven features: six symptoms, an age tertile and four image measures. The learner analysis evaluates ten features, because age and body region enter the clinical learner as inputs and the age tertile is not evaluated separately. Patients were split 60/20/20 with a fixed seed. M0 was trained on all 1,402 training images with the recorded diagnosis, and M2 on the 814 biopsied training images. Both used L2-regularized logistic regression on frozen image features or on clinical features, namely age, body region and six symptoms, and were evaluated on the same 463 test images. Intervals and probabilities come from 200 joint bootstrap resamples of training and test patients, with refitting. Table {{T:pad}} reports each feature. Every image has equal weight, so a lesion with several images contributes several. Categories classify point estimates. For image asymmetry and border irregularity, the M2 interval contains zero and opposite signs occurred in 19 and 39 percent of resamples, so their attenuation is not a stable conclusion.

**TABLE {{T:pad}}. Learned contrasts on PAD-UFES-20.**

{{TABLE:pad}}

**Slope version of the gap.** Before estimating *g*, we also compared the slope of the learned logit gap on each concept, adjusted for covariates, with the logistic slope of the indicator *V* of a biopsied recorded negative. That comparison relies on rare disease, which Proposition 1 does not need. Its correlation across the five concepts was 0.97 in every family. Correlations over five fixed concepts are descriptive and are not inference over a population of concepts.

## S7. Plug-in bounds for the disease target

**Proposition 2 on ISIC-2024.** The nuisance *q* was the Platt-calibrated M0 on the learner's input, and σ an out-of-fold gradient-boosting model on the tabular features for both families, both fitted without the test fold. All sets in this section use the stratum formula, which is sharp for the tabular-input target and an outer set for the image-feature target. With these plug-ins the bounds are plug-in bounds, an estimate of the identified set and not the population set of Proposition 2. Table {{T:psi}} reports them at three floors, together with the plug-in tipping floor *s*∗. This is the smallest floor on a grid of step 0.01 at which ψ_L > 0, computed on the seed-averaged curve. The last entry is the 95th percentile of *s*∗ over the {{N:B19}} one-seed joint replicates of training and test patients, with validation fixed. It is a bootstrap stability threshold, not a floor identified from the data. Table {{T:full30}} gives the bootstrap stability threshold with validation patients also resampled, for three estimates of *q*.

In Table {{T:psi}}, bounds are under the stated malignant-verification floor for the frozen-feature families, averaged over seeds 0 to 2. They replace *q* and σ by estimates and are not the population identified set of Proposition 2.

**TABLE {{T:psi}}. Plug-in bounds for the disease target on the learner's scale.**

{{TABLE:psi}}

**Estimated-weight approximation for the image-feature target.** Script `vr42_sharp_weighted.py` refitted M0 with seeds 0 to 2 and Platt calibration. It estimated P(*t* | *x̃*) by gradient boosting on the tabular features or on 64 principal components of the image features, cross-fitted over five folds of test patients. It then computed the outer set and an estimated-weight approximation to the sharp set on the seed-averaged curve. Stability thresholds use 1,000 test-patient resamples conditional on the fitted learners and weights, so they are not comparable to the joint thresholds of Table {{T:full30}}. Table {{T:sharp42}} reports the results. The agreement column is the share of outer-tertile test lesions whose estimated weight has the sign of their observed stratum. Because the stratum is not a function of the image input, the sign of *a*_k need not match a lesion's observed stratum. This share measures how well the input predicts the stratum; it is not the accuracy of the weight sign. The estimated-weight result is exploratory.

In Table {{T:sharp42}}, M0 is Platt-calibrated, and stability thresholds resample test patients only, with the learners and weights fixed, unlike the refitted thresholds of Table 2; sharpness holds for the population set with the true weight and is not guaranteed after replacing it by an estimate.

**TABLE {{T:sharp42}}. Plug-in tipping floor from the stratum formula and an estimated-weight approximation.**

{{TABLE:sharp42}}

**Inputs, branches and clipping.** Proposition 2 takes *q* and σ on the same input, while the primary plug-in uses one tabular σ̂ for both families. Script `vr48_nuisance_audit.py` refitted M0 with seeds 0 to 2 and checked three points (Table {{T:nuis48}}). First, σ̂ enters ψ_L only through the second branch *q̂* + 1 − σ̂ of *p̄*. At the point floor it was active for at most 0.01 percent of lower-tertile lesions, and replacing σ̂ by a perceptron on the learner's input left the floors unchanged. Second, *q̂* exceeded σ̂ for some lesions, which is impossible in the population because *Y* ≤ *S*. Such lesions only push the second branch above one, so they leave the bound unchanged, but they show that the two nuisances are not jointly coherent. Third, isotonic calibration of the tabular learner put many test lesions at its lower bound of 10⁻⁶, so its point floor depends on that arbitrary bound. In a separate refit without clipping, almost all of these lesions had an isotonic estimate of exactly zero; fewer than 0.01 percent had a positive estimate below 10⁻⁶. The second branch is chosen before clipping, and the minimum is then clipped to [10⁻⁷, 1 − 10⁻⁷]. The σ̂ is the fixed out-of-fold estimate in every bootstrap replicate, so the input-matched check covers the point floor of color variegation only. Platt calibration put no lesion at the clipping bound, and the isotonic fit of the image learner reached neither bound.

In Table {{T:nuis48}}, shares are over test lesions and averaged over seeds 0 to 2, and floors use the seed-averaged curve of the exact bound.

**TABLE {{T:nuis48}}. Nuisance inputs, active branch and clipping of the plug-in bound for color variegation.**

{{TABLE:nuis48}}

**Calibration and nuisance choices.** Script `vr25_psi_sensitivity.py` recomputed *s*∗ on the original fits with six estimates built on the perceptron M0. Four vary the calibration: none, Platt (Platt, 1999), isotonic (Zadrozny & Elkan, 2002) or beta calibration (Kull et al., 2017). The other two use Platt calibration, either with σ replaced by max(σ̂, *q̂*) from a perceptron trained on the learner's input or with logit *q* and logit *p̄* winsorized at the 1st and 99th percentiles. On the logit scale a Platt slope multiplies the concept contrast, so the calibration step moves *s*∗. Table {{T:psisens25}} reports the results, and Table {{T:full30}} gives gradient boosting under the bootstrap. Beta calibration, the alternative σ and winsorizing left the Platt value almost unchanged. Isotonic calibration moved it by about 0.1. The uncalibrated image learner, whose Platt slope was about 0.45, lowered it to below 0.2.

**Fit of the estimates of *q*.** Table {{T:qdiag30}} reports the test log-loss and Brier score of each estimate of *q* for the original fits, averaged over seeds 0 to 2. It also shows calibration by bins of predicted risk: the number of lesions, their mean predicted risk and the number of malignant lesions in each bin. Most test lesions fall below a predicted risk of 0.002, where the bins hold few malignant lesions, so these diagnostics cannot tell the estimates apart where ψ_L is decided. Event counts by tertile in Table {{T:tail38}} are diagnostics, not tests: they ignore patient clustering and nuisance uncertainty. For the tabular input, *t* is a function of *x̃*, so a correct *q* matches the expected count of each tertile in the population. For image features *t* can carry information on *Y* beyond *x̃*, so even a correct *q* need not match it.

In Table {{T:qdiag30}}, original fits are averaged over seeds 0 to 2.

**TABLE {{T:qdiag30}}. Fit of the estimates of *q* on the test population.**

{{TABLE:qdiag30}}

**Why the estimates differ.** Script `vr38_logit_tail.py` works within the outer tertiles of color variegation on the test population. It compares the observed number of malignant lesions, the expected number under each estimate of *q*, the mean of *q* and the mean of logit *q*. Empirical calibration approximately matches aggregate event risk, a sum of *q* over lesions, although it is not an exact constraint for every calibration method or bin. The disease target averages logit *q*, which is steep and concave near zero. Estimates with nearly the same expected events can differ widely in mean logit *q* when most lesions have very small *q*. Table {{T:tail38}} reports the values.

In Table {{T:tail38}}, values are on the test population for the original fits averaged over seeds 0 to 2.

**TABLE {{T:tail38}}. Events and logit-scale mean of *q* in the outer tertiles of color variegation.**

{{TABLE:tail38}}

In Table {{T:psisens25}}, all rows use the perceptron M0 and the seed-averaged curve, and "none" means that no floor up to one made ψ_L positive.

**TABLE {{T:psisens25}}. Plug-in tipping floor under calibration and nuisance choices.**

{{TABLE:psisens25}}

**Two floors.** A single floor on the lower stratum must hold even where verification is rarest. Let the floor be s_low for test lesions whose σ̂ lies below the 1st or 5th percentile threshold of Section S6 and s_high elsewhere. Table {{T:twofloor25}} gives, on a grid of step 0.05, the smallest s_low at which ψ_L > 0 for each s_high. A strong floor outside the low-support region lets a weaker one suffice inside it. This analysis uses the Platt-calibrated M0 only and is exploratory; the dependence on the estimator of *q* shown in Table {{T:full30}} applies to it as well.

In Table {{T:twofloor25}}, the floor elsewhere is given, and M0 is Platt-calibrated.

**TABLE {{T:twofloor25}}. Smallest low-support floor that keeps ψ_L positive.**

{{TABLE:twofloor25}}

**Marginal risk ratio.** The marginal version of Section S1 identifies the disease risk ratio between the outer tertiles as above one exactly when *A* < RR_Y. A floor *s*_min on the average malignant verification in the lower tertile alone implies *A* ≤ 1/*s*_min. This analysis was specified after color variegation had emerged as the central case and is a post-inspection sensitivity analysis. Script `vr35_marginal_rr.py` computes RR_Y directly from counts, with no learner, on the test population and on the whole cohort, with 2,000 patient-cluster bootstrap replicates. RR_D > 1 is identified at lower-tertile floors above 1/RR_Y, and the next column replaces RR_Y by its 5th percentile over resamples. Table {{T:marginal35}} also gives the log odds ratio of the recorded label between the tertiles among verified lesions, the verified-only association of Lemma 1. The 95 percent resampling stability threshold is the smallest floor at which two events held together in at least 95 percent of the same patient-cluster resamples: RR_D > 1 was identified, and the verified-only log odds ratio was negative. This joint event is the marginal association reversal, conditional on that floor. The next column gives the same threshold on the scale of *A*. Both are thresholds of a resampling procedure, not lower confidence bounds for malignant verification, and nothing controls error for the post-inspection choice of concept and analysis. The last column is the observed benign ratio *B*_V = *g*₁/*g*₀, a plausibility benchmark for *A* rather than an estimate of it. Malignant and benign lesions may reach biopsy through different routes. The whole cohort is used because the estimand needs no fitted learner; the test population, on which the learner analyses are evaluated, is reported alongside. It is a different estimand from ψ on a different scale and population, so it is neither an estimator nor a check of ψ and does not validate Proposition 2. The site-standardized row describes the observed mixture of sites under a common within-site floor; it is a mixture estimand and does not show a reversal in each site.

**Indeterminate histopathology.** The release records 114 biopsied lesions as Indeterminate, and its binary target counts them as not malignant (Table {{T:ind49}}). Script `vr49_indeterminate.py` repeats the whole-cohort count analysis under three codings. The first is the binary rule of the release, used throughout. The second treats indeterminate lesions as unverified for disease, so the verified set holds only Malignant and Benign histopathology. The third counts them as malignant. Table {{T:ind49b}} reports the results with 2,000 patient-cluster resamples. Every coding kept RR above one and the verified-only log odds ratio below zero for the three primary concepts, and the thresholds changed by at most 0.10. The learners are trained on the released label, and this table recomputes only the count analyses. Of the 400 biopsied recorded negatives that train M2, 70 are indeterminate. Recoding them changes the counts, for example *B*_V for color variegation from 4.96 to 6.03. Script `vr51_direct_indeterminate.py` therefore retrained M2 without the indeterminate lesions in training and early stopping, on the 5,000 three-seed replicates of Table 1. Every Bonferroni label was unchanged (Table {{T:vr51}}), with color variegation for tabular features still unresolved. Per replicate, the three-seed M2 contrast changed sign only for size with tabular features, in 3 of 5,000 replicates; with tabular features the M2 contrasts became more negative, for example color variegation from −0.95 to −1.19 in the original fits, while image features barely moved. Counting the indeterminate lesions as malignant and retraining both learners kept the sign of every point estimate of the original fits (Table {{T:vr51p}}).

In Table {{T:vr51p}}, entries are tertile contrasts of the original fits averaged over seeds 0 to 2 on the test population.

**TABLE {{T:vr51p}}. Learned contrasts under three codings of indeterminate histopathology.**

{{TABLE:vr51p}}

In Proposition 2, excluding indeterminate lesions from verification changes only σ, which enters the second branch of *p̄*; that branch was active for at most 0.01 percent of lower-tertile lesions in the point calculation of Table {{T:nuis48}}, which used the release coding.

In Table {{T:ind49}}, counts are lesions with a linked histopathology diagnosis, by patient split and by outer tertile over the whole cohort.

**TABLE {{T:ind49}}. Histopathology diagnoses of biopsied lesions.**

{{TABLE:ind49}}

In Table {{T:ind49b}}, values come from whole-cohort counts with patient-cluster bootstrap intervals, and the threshold lies on a grid of step 0.05.

**TABLE {{T:ind49b}}. Marginal analysis under three codings of indeterminate histopathology.**

{{TABLE:ind49b}}

**Marginal risk ratio and the learner-scale target.** RR_D is a ratio of mean risks between the outer tertiles, whereas ψ contrasts mean logits of *p*(*x̃*) on the test population. For the tabular input, *t* is a function of *x̃*, so E[*p*(*X̃*) | *t*] = P(*D* = 1 | *t*) and the ratio of these means is RR_D on the same population. For image features this needs P(*D* = 1 | *x̃*, *t*) = P(*D* = 1 | *x̃*). If the input carries no information, for example, the ratio of mean *p*(*x̃*) is one while RR_D can be three. The marginal analysis also uses the whole cohort, whereas Table 2 uses the test population.

**Sites and the scale of the benchmark.** Script `vr41_marginal_site.py` repeats the marginal analysis by acquisition site, standardized across sites and with one site left out at a time. The standardized risk ratio is Σ_g *w*_g P(*Y* = 1 | *t* = 1, *g*) / Σ_g *w*_g P(*Y* = 1 | *t* = 0, *g*), with one reference distribution *w*_g, the share of outer-tertile lesions in site *g*, used for both tertiles. Suppose the lower-tertile floor π₀¹(*g*) ≥ *s* holds in every site *g*. Then the standardized disease risk ratio is at least *s* times the standardized RR_Y. Each site's upper-tertile disease risk is at least its recorded risk, its lower-tertile disease risk is at most its recorded risk divided by *s*, and the common weights carry both inequalities to the sums. The same one-sided argument applies with site-specific verification. The result describes the observed mixture of sites and does not by itself identify the corresponding result at a new site. The verified-only association is pooled with the Mantel-Haenszel odds ratio, so the two sides use different weights. Table {{T:site47}} therefore also standardizes the verified-only side with the same weights *w*_g, as a risk ratio among verified lesions. It uses the six sites with verified lesions in both outer tertiles. Site 3 has none in the lower tertile and is excluded from both sides. In a resample where another site loses verified lesions in one tertile, that site is dropped from both sides. Script `vr47_counts_audit.py` computes this version. The threshold in Table {{T:site47}} is thus for an adaptive procedure whose site set can change between resamples. If a resample that drops a site counts as undetermined, so that the six-site mixture stays fixed, color variegation never reaches 95 percent, and size reaches it at a higher floor (Table {{T:site47}}). Table {{T:site41}} reports the results for color variegation and size. For color variegation the crude pooled threshold was 0.65; site standardization with Mantel-Haenszel pooling gave 0.62, the common-weight version 0.63 only with resamples that drop sites, and the fixed six-site mixture did not reach the criterion. The association also differed across sites. For color variegation, the two sites with the most malignant lesions would need floors near 0.8 on their own.

The script also compares two ways of transporting the appearance dependence of benign verification to malignant verification. On the ratio scale, *A* = *B*_V is impossible once π₀¹ > 1/*B*_V. On the logit scale, a common gradient γ = log *B*_V gives *A*(π₀¹, γ) = expit(logit π₀¹ + γ)/π₀¹, which falls below the largest *A* of Table {{T:marginal35}} once π₀¹ exceeds the value marked in Fig. S3. Neither scale is identified, and clinicians may use information that differs between malignant and benign lesions, so both are benchmarks.

In Fig. S3, the contour lies in the plane of lower-tertile malignant verification π₀¹ and its logit gradient γ, for color variegation and size. To the right of each curve *A* is below the threshold and the positive disease sign is identified. As γ grows the curve approaches π₀¹ = 1/*A*, the lower-tertile floor of Table {{T:marginal35}}. Points mark γ = log *B*_V. The figure file is `figures/figS3_logit_benchmark.png`.

**Fig. S3.** Largest *A* compatible with the marginal association reversal.

In Table {{T:site47}}, weights are shares of outer-tertile lesions renormalized over the six sites, and the threshold uses 2,000 patient-cluster resamples on a grid of step 0.01.

**TABLE {{T:site47}}. Site standardization with common weights on both sides.**

{{TABLE:site47}}

In Table {{T:site41}}, floors are point values; Site 3 has no malignant lesion in the lower tertile, so its RR_Y and floor are not defined; the 95 percent resampling stability threshold without a site uses 500 patient-cluster resamples and the standardized row 2,000.

**TABLE {{T:site41}}. Marginal analysis by acquisition site.**

{{TABLE:site41}}

In Table {{T:marginal35}}, values come from counts with patient-cluster bootstrap intervals, and floors refer to the average malignant verification in the lower tertile. The stability threshold and the largest *A* are thresholds of the resampling procedure, not confidence bounds. The threshold lies on a grid of step 0.05 and the largest *A* on a grid of step 0.01, so they are not exact reciprocals. The observed benign ratio *B*_V is a plausibility benchmark for *A*, not an estimate of it.

**TABLE {{T:marginal35}}. Marginal risk ratio of the recorded label between the outer tertiles.**

{{TABLE:marginal35}}

## S8. Reproducibility

Table {{T:repro}} maps every reported result to the script that produces it. The analysis lock, which fixed primary and secondary outcomes before the tests of alternative explanations, is in `analysis_lock.md`. It is dated 25 September 2026 and is part of the first public commit of the repository; its later sections log each post-inspection change with its date.

**Rebuilding the main results.** The driver `Code/run_vr_pipeline.sh` and the `Code/run_round*.sh` scripts record the order and arguments of every run. A minimal chain for Table 1 and Table {{T:dose46}} is as follows. `vl0_embed.py` computes the image features from the ISIC-2024 archive. `vl1_identification.py` fits the nuisances, and `vl_common.py` freezes the patient split. `vr19_primary_bootstrap.py --family tabular --start -3 --end 0` fits the three original models (seeds 0 to 2) that give the point estimates, and `--start 0 --end 5000` the one-seed replicates; the same for `image`, then `vr19_primary_bootstrap.py --summarize`. Running this chain from the data regenerates every replicate; building the text from the stored JSON needs only the last step below. `vr39_seed_variance.py` adds two seeds per replicate, and `vr43_three_seed_primary.py` gives the intervals and labels of Table 1. `vr46_dose_fixed_val.py --family tabular --cells color_variegation:320 size:320 lesion_skin_contrast:320`, the same for `image`, then `--summarize`, give Table {{T:dose46}}. `paper/build_paper.py` and `paper/build_supplementary.py` rebuild the text. Embeddings, nuisance arrays and checkpoints are not distributed because of their size, but these scripts regenerate them from the public data. On other hardware, refits can differ slightly; the labels of Table 1 are stable to the Monte Carlo resolution reported in Section S6.

In Table {{T:repro}}, scripts are in `Code/` and outputs in `Result/` of the repository.

**TABLE {{T:repro}}. Where each result comes from.**

| Result | Script | Output |
| --- | --- | --- |
| Lemma 1 and 2 checks; slope identity | `vr1_reversal_theory.py` | `vr1_reversal_theory.json` |
| Table {{T:symbolic}} | `vr10_symbolic.py` | `vr10_symbolic.json` |
| Unit tests | `test_reversal_theory.py` | 30 tests |
| Table {{T:psisim}} | `vr23_psi_sim.py` | `vr23_psi_sim.json` |
| Table {{T:psistress}} | `vr28_psi_stress.py` | `vr28_psi_stress.json` |
| Fig. S1 | `vr2_phase_diagram.py` | `vr2_phase_diagram.json` |
| Fig. 2; Table {{T:definitions}} | `vr9_closing.py` | `vr9_closing.json`; `closing_heads.npz`, not distributed |
| Table 1 point estimates (original fits, seeds 0 to 2); one-seed check in Tables {{T:joint19}}, {{T:support19}} and {{T:psi}} | `vr19_primary_bootstrap.py` | `vr19_primary_bootstrap.json`, `vr19/` |
| Plug-in point tipping floor in Table {{T:psi}} | `vr24_full_bootstrap.py` | `vr24_full_bootstrap.json` |
| Tables {{T:arms21}}, {{T:dose21}} and {{T:ess21}} | `vr21_dose_calibrated.py` | `vr21_dose_calibrated.json` |
| Table {{T:gumbel26}} | `vr26_dose_poisson.py` | `vr26_dose_poisson.json` |
| Table {{T:tau44}} | `vr44_dose_estimand.py` | `vr44_dose_estimand.json` |
| Share of test lesions retained at each support threshold, Section IV-D | `vr45_support_retention.py` | `vr45_support_retention.json` |
| Table {{T:eiv27}} | `vr27_bridge_eiv.py` | `vr27_bridge_eiv.json` |
| Table 2; Tables {{T:full30}} and {{T:qdiag30}} | `vr30_q_bootstrap.py` | `vr30_q_bootstrap.json`, `vr30/` |
| Table {{T:sub36}} | `vr36_subsample.py` | `vr36_subsample.json`, `vr36/` |
| Fig. 3(a); dose results of Section IV-B; Tables {{T:dose46}} and {{T:spill46}} | `vr46_dose_fixed_val.py` | `vr46_dose_fixed_val.json`, `vr46/` |
| Tables {{T:dose32}} and {{T:spill32}} (joint training-and-validation dose) | `vr32_dose_nuisance.py` | `vr32_dose_nuisance.json` |
| Table {{T:bridge33}} | `vr33_bridge_alt.py` | `vr33_bridge_alt.json` |
| Tables {{T:semi34}} and {{T:semibridge34}} | `vr34_semisynth.py` (argument `mlp` or `gbm`) | `vr34_semisynth.json`, `vr34_semisynth_gbm.json` |
| Table {{T:sigma37}} | `vr37_overlap_sigma.py` | `vr37_overlap_sigma.json` |
| Table {{T:tail38}} | `vr38_logit_tail.py` | `vr38_logit_tail.json` |
| Two further seeds per replicate for the three-seed bootstrap; Table {{T:seed39}} | `vr39_seed_variance.py` | `vr39/`, `vr39_seed_variance.json` |
| Table {{T:local34}} | `vr34_semisynth.py` (argument `local`) | `vr34_semisynth_local.json` |
| Tables {{T:comp40}} and {{T:partial40}} | `vr40_support_composition.py` | `vr40_support_composition.json`; `vr40_sigma.npz`, not distributed |
| Table {{T:site41}}, Fig. S3 | `vr41_marginal_site.py` | `vr41_marginal_site.json` |
| Tables {{T:theory42}} and {{T:sharp42}} | `vr42_sharp_weighted.py` | `vr42_sharp_weighted.json` |
| Table 1 intervals and labels (primary three-seed bootstrap, from `vr19/` and `vr39/`); Table {{T:mc43}} | `vr43_three_seed_primary.py` | `vr43_three_seed_primary.json` |
| Table {{T:marginal35}} | `vr35_marginal_rr.py` | `vr35_marginal_rr.json` |
| Table {{T:mc29}} (Monte Carlo audit of the one-seed check) | `vr29_bootstrap_audit.py` | `vr29_bootstrap_audit.json` |
| Tables {{T:psisens25}} and {{T:twofloor25}} | `vr25_psi_sensitivity.py` | `vr25_psi_sensitivity.json` |
| Fig. 3(b); Table {{T:pointwise22}} | `vr22_pointwise_bridge.py` | `vr22_pointwise_bridge.json` |
| Tables {{T:within}}, {{T:sites}} and {{T:overlap}} | `vr16_mechanism.py` | `vr16_mechanism.json` |
| Tables {{T:sitehet}} and {{T:loso}}; PAD-UFES-20 counts | `vr20_audit.py` | `vr20_audit.json` |
| Tables {{T:size_matched}} and {{T:pad}} | `vr12_claim_validation.py` | `vr12_claim_validation.json` |
| Slope version of the gap | `vr12_claim_validation.py --parts b` | `vr12_quantitative.json` |
| Fine-tuned and linear-probe models | `vr4_finetune.py` | `vr4_finetune.json`; `finetune/*.npz` and checkpoints, not distributed |
| Table {{T:repr}}; Fig. S2 | `vr5_representation.py` | `vr5_representation.json` |
| Table {{T:headswap}} | `vr13_headswap.py` | `vr13_headswap.json` |
| Fig. 4; PAD-UFES-20 associations | `vr6_pad_boundary.py`; ISIC-2024 paths from `vr47_counts_audit.py` | `vr6_pad_boundary.json`, `vr47_counts_audit.json` |
| Tables {{T:mc47}} and {{T:site47}}; sharp marginal interval | `vr47_counts_audit.py` | `vr47_counts_audit.json` |
| Table {{T:clip50}} | `vr50_logit_clip.py` | `vr50_logit_clip.json` |
| Tables {{T:vr51}} and {{T:vr51p}} | `vr51_direct_indeterminate.py`, driven by `supervise_vr51.sh` | `vr51_direct_indeterminate.json`, `vr51/` |
| Table {{T:nuis48}}; isotonic zeros (argument `--iso-zero`) | `vr48_nuisance_audit.py` | `vr48_nuisance_audit.json` |
| Tables {{T:ind49}} and {{T:ind49b}} | `vr49_indeterminate.py` | `vr49_indeterminate.json` |
| Patient split; nuisance estimates of *q* and σ | `vl_common.py`, `vl1_identification.py`, `vl3_learning.py` | `vl_split.json`; `vl_nuisance.npz` and `vl_nuisance_image.npz`, not distributed |
| Test AUROC of the tabular and frozen-image learners | `vl3_learning.py` | `vl3_learning.json` |
| Frozen ResNet-50 embeddings | `vl0_embed.py` | `embeddings/`, not distributed |
| Interval for *B* on ISIC-2024 | `vl1_identification.py` | `vl1_identification.json` |
| All figures | `vr11_paper_figures.py` | `paper/figures/` |
| Tables and result sentences | `paper/build_paper.py`, `paper/build_supplementary.py`, `paper/paper_numbers.py` | generated from the files above |

The patient split is frozen in `Result/vl_split.json`. Rebuilding it without retraining every model breaks the patient-disjoint property.

## Supplementary references

DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials. *Controlled Clinical Trials*, *7*(3), 177–188.

Kish, L. (1965). *Survey Sampling*. Wiley.

Kool, W., van Hoof, H., & Welling, M. (2019). Stochastic beams and where to find them: The Gumbel-top-k trick for sampling sequences without replacement. In *Proceedings of the 36th International Conference on Machine Learning* (pp. 3499–3508).

Kull, M., Silva Filho, T., & Flach, P. (2017). Beta calibration: A well-founded and easily implemented improvement on logistic calibration for binary classifiers. In *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics*, PMLR 54 (pp. 623–631).

Meurer, A., Smith, C. P., Paprocki, M., Čertík, O., Kirpichev, S. B., Rocklin, M., … & Scopatz, A. (2017). SymPy: Symbolic computing in Python. *PeerJ Computer Science*, *3*, e103.

Platt, J. C. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In A. J. Smola, P. Bartlett, B. Schölkopf, & D. Schuurmans (Eds.), *Advances in Large Margin Classifiers* (pp. 61–74). MIT Press.

Politis, D. N., Romano, J. P., & Wolf, M. (1999). *Subsampling*. Springer.

Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. In *Proceedings of the 8th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 694–699).
