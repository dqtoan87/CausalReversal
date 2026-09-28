# Supplementary Material

**Causal Analysis of Learning Reversal under Selective Verification in Skin Cancer Classification**

## S1. Notation and proofs

Table S1 fixes the notation. The code uses the same names, so each symbol can be traced to its implementation.

**TABLE S1. Notation used in the paper and in the code.**

| Symbol | Meaning | Name in code |
| --- | --- | --- |
| *D* | latent disease state, observed only where *S* = 1 | `D` |
| *X*, *C* = *h*(*X*) | lesion appearance and a concept read from it | `x`, concept columns |
| *U*, *G*, *W* | patient-level risk, acquisition site, clinical information outside the image | not modeled directly |
| *F* | lesion flagged by a clinician as a lesion of interest | `F` |
| *S* | lesion received histopathology; *S* ⊂ *F* | `S` |
| *Y* = *D*·*S* | recorded malignant label | `Y` |
| *t* | lower (0) or upper (1) cohort tertile of a concept | `tertile` |
| π¹(*x*), π⁰(*x*) | P(*S* = 1 \| *D* = 1, *x*) and P(*S* = 1 \| *D* = 0, *x*) | `pi1`, `pi0` |
| *g*(*x*) | P(*S* = 1 \| *x*, *Y* = 0), verification propensity among recorded negatives, observable | `g` |
| *A*, *B* | π₁¹/π₀¹ and π₁⁰/π₀⁰, not observable | `logA`, `logB` |
| *B*_V | *g*₁/*g*₀, observed verified-benign contrast | `V` slope, `log_BV` |
| θ_k, δ | disease contrast log OR_D and log *B* − log *A* | `theta`, `delta` |
| *q*(*x*), σ(*x*) | P(*Y* = 1 \| *x*) and P(*S* = 1 \| *x*), the overall verification propensity | `q`, `sigma` |
| *s*(*x*), *s*_min | P(*S* = 1 \| *D* = 1, *x*) and its assumed pointwise floor | `s`, `s_min` |
| M0, M2 | learner on all lesions with *Y*; learner on verified lesions only | `erm_y`, `erm_verified` |
| Δ_{m,k} | learned contrast of learner *m* for concept *k* | `vr9_closing` |
| η | selection dose on recorded-negative training lesions | `vr26_dose_poisson`, `vr21_dose_calibrated` |
| ω | exogenous random numbers of the pipeline: inclusion uniforms, malignant training draw, validation-set draw and optimizer seed | `vr46_dose_fixed_val` |
| τ_j(η₁, η₀) | training-selection effect, E_ω[Δ_j(η₁, ω) − Δ_j(η₀, ω)] | `vr46_dose_fixed_val` |

**Positivity.** Throughout, P(*D* = 1 | *x*) ∈ (0, 1), π¹, π⁰ ∈ (0, 1] and *r*₁ > 0. If a verification probability is zero, the verified-only odds ratio is undefined. A unit test checks that the code does not return a finite value in that case.

**Proof of Proposition 1.** Fix an input value *x̃* and write *r*_y(*x̃*) = P(*R* = 1 | *Y* = *y*, *x̃*). By Bayes' rule, P(*Y* = 1, *R* = 1 | *x̃*) = P(*Y* = 1 | *x̃*)·*r*₁(*x̃*) and P(*Y* = 0, *R* = 1 | *x̃*) = P(*Y* = 0 | *x̃*)·*r*₀(*x̃*). Their ratio is the odds of *Y* = 1 given *x̃* and *R* = 1, and taking logarithms gives the statement. No disease label, and no assumption on how *D* enters *Y*, is used.

*Selection that depends on appearance.* Suppose *R* is independent of *X̃* given *X* and *Y*, and write *r*_y(*x*) = P(*R* = 1 | *Y* = *y*, *x*). Then *r*_y(*x̃*) = E[*r*_y(*X*) | *x̃*, *Y* = *y*]. Without this condition, for example if clinicians used age or site beyond what the image shows, the statement still holds with *r*_y defined on the input. The averaging step, however, does not.

*Training on verified lesions.* With *R* = *S*, a lesion with *Y* = 1 has *S* = 1, so *r*₁(*x̃*) = 1. On the input itself, *r*₀(*x̃*) = P(*S* = 1 | *x̃*, *Y* = 0) = *g*(*x̃*). The Bayes-optimal M0 predicts *f*₀ = P(*Y* = 1 | *x̃*). The Bayes-optimal M2 predicts *f*₂ = P(*Y* = 1 | *x̃*, *S* = 1), which equals P(*D* = 1 | *x̃*, *S* = 1). Hence logit *f*₀ − logit *f*₂ = log *g*(*x̃*). Taking the mean over each stratum of the evaluation population and subtracting gives the statement for Δ. If a learner's logit is shifted by a constant, for example because positives were resampled, the shift is the same in both strata and cancels in Δ.

*Controlled selection dose.* Recorded-positive training lesions are drawn uniformly, so *r*₁ is constant. Each recorded-negative training lesion enters independently with probability π_η(*z*) = min{1, κ_η exp(η*z*)}, with *z* = *z*_k(*X*) and κ_η chosen so that the expected number equals the target. Selection depends on the lesion only through *z*_k and *Y*, so the averaging step applies by construction. If *z*_k is a function of *x̃*, then *r*₀(*x̃*) = π_η(*z*_k(*x̃*)) exactly, and the predicted change of Δ_j is minus the stratum contrast of log π_η(*z*_k) on the evaluation population. Without the cap this is −η·μ_{kj}, with μ_{kj} = E[*z*_k | *t*_j = 1] − E[*z*_k | *t*_j = 0]. If the input determines *z*_k only partly, *r*₀(*x̃*) = E[π_η(*Z*) | *x̃*, *Y* = 0]. Ignoring the cap and treating *Z* given *x̃* and *Y* = 0 as normal with mean *m*_k(*x̃*) and variance *v*_k(*x̃*), log *r*₀(*x̃*) = log κ_η + η*m*_k(*x̃*) + η²*v*_k(*x̃*)/2. The first-order prediction uses *m*_k only, which is exact when *v*_k does not depend on *x̃*. The second-order prediction adds the stratum contrast of *v*_k. Both are approximations. We estimate *m*_k by ridge regression of *z*_k on the input among recorded-negative training lesions, and *v*_k by ridge regression of the squared residual on a held-out half.

*Sampling without replacement.* A second design drew exactly *n* recorded negatives by the Gumbel top-*k* construction (Kool et al., 2019) with weights *w* = exp(η*z*). Its inclusion probabilities are not proportional to *w*. They are well approximated by 1 − exp(−λ*w*), with λ fixed by the expected count, which is proportional to *w* only while λ*w* is small. Section S6 reports the size of the deviation.

**Proof of Lemma 1.** Among verified lesions in stratum *t*, P(*D* = 1, *S* = 1 | *t*) = *p*ₜπₜ¹ and P(*D* = 0, *S* = 1 | *t*) = (1 − *p*ₜ)πₜ⁰. The odds of disease among verified lesions are therefore odds(*p*ₜ)·πₜ¹/πₜ⁰. Taking the ratio across strata gives OR_{D|S} = OR_D·*A*/*B*, the classical multiplicative selection bias factor with the four selection probabilities grouped into a malignant and a benign ratio. Logarithms give log OR_{D|S} = θ − δ. For θ > 0, the association keeps its sign exactly when δ < θ. It is weaker when 0 < δ < θ and stronger when δ < 0. It reverses when δ > θ, which is equivalent to *B*/*A* > OR_D. For θ < 0 the same argument gives reversal for δ < θ, attenuation for θ < δ < 0 and amplification for δ > 0. If *A* = *B*, then OR_{D|S} = OR_D. If θ = 0, then log OR_{D|S} = −δ.

**Relation between *B* and *B*_V.** *g*ₜ = P(*S* = 1 | *Y* = 0, *t*) = (1 − *p*ₜ)πₜ⁰ / (1 − *p*ₜπₜ¹). Hence *B*_V = *B*·[(1 − *p*₁)(1 − *p*₀π₀¹)] / [(1 − *p*₀)(1 − *p*₁π₁¹)]. When disease is rare, each factor is close to one. Without the rare-disease approximation, *B* is partially identified. Write *V* = *S*(1 − *Y*), which is observed, and *v*ₜ = P(*V* = 1 | *t*) = (1 − *p*ₜ)πₜ⁰. Then *B* = [*v*₁/(1 − *p*₁)] / [*v*₀/(1 − *p*₀)], which is decreasing in *p*₀ and increasing in *p*₁. With each *p*ₜ in a known interval, the extremes of *B* are attained at the corners.

**Marginal risk ratio.** Because *Y* = *D*·*S*, P(*Y* = 1 | *t*) = *p*ₜπₜ¹, and hence RR_Y = RR_D·*A* exactly. The disease risk ratio exceeds one exactly when *A* < RR_Y, which is the most direct sensitivity parameter. A floor is one way to bound *A*. Because π₁¹ ≤ 1, the one-sided condition π₀¹ ≥ *s*_min on the lower stratum alone gives *A* ≤ 1/*s*_min. The sign of RR_D is then identified as positive when RR_Y·*s*_min > 1. If the floor holds in both strata, *A* ∈ [*s*_min, 1/*s*_min], and every value in that range is attained by some pair (π₀¹, π₁¹). The sharp interval for log RR_D is then [log RR_Y + log *s*_min, log RR_Y − log *s*_min]. The lower endpoint, and hence the positive sign, uses only the lower-stratum floor. For two strata, log RR_D and θ = log OR_D have the same sign, because both are positive exactly when *p*₁ > *p*₀. The condition concerns the average malignant verification within a stratum. When the concept is a function of the learner's input, the pointwise floor of Proposition 2 implies it, but not conversely, so the marginal result rests on a weaker assumption. With covariate-adjusted in place of crude risk ratios, the same argument holds within each covariate cell, so we report the adjusted version only as a sensitivity analysis.

**Lemma 2 (logistic learners).** If logit *p*(*x*) = α + βᵀ*x* and π¹(*x*)/π⁰(*x*) = exp(ℓ₀ + λᵀ*x*) for a coefficient vector λ, the population-optimal verified-only logistic learner converges to β + λ. When verification depends on the outcome only, λ = 0.

**Proof of Lemma 2.** By Bayes' rule, odds(*D* = 1 | *x*, *S* = 1) = odds(*D* = 1 | *x*)·π¹(*x*)/π⁰(*x*). Under the log-linear likelihood ratio, logit P(*D* = 1 | *x*, *S* = 1) = α + ℓ₀ + (β + λ)ᵀ*x*. A logistic learner fitted on the verified population is then correctly specified, so its population optimum is β + λ. When λ = 0, verification depends on the outcome only and the slope is unchanged, which is the result of Prentice and Pyke for case-control sampling. Outside the log-linear model none of these equalities is guaranteed.

**Proof of Proposition 2.** Because *Y* = *D*·*S*, *q*(*x̃*) = *p*(*x̃*)·*s*(*x̃*), so *p* = *q*/*s*. With *s* ∈ [*s*_min, 1], *p* ∈ [*q*, *q*/*s*_min]. In addition, P(*D* = 1, *S* = 0 | *x̃*) ≤ P(*S* = 0 | *x̃*), so *p* ≤ *q* + 1 − σ. Hence *p*(*x̃*) ∈ [*q*(*x̃*), *U*(*x̃*)]. Setting *s* = 1 attains the lower endpoint. Setting *s* = *s*_min attains the upper endpoint when *q*/*s*_min binds, and assigning every unverified lesion with input *x̃* to disease attains it otherwise. Both choices are compatible with the observed distribution of (*X̃*, *t*, *S*, *Y*). The choice can differ across values of *x̃*, but not across lesions that share *x̃*, because *p* is a function of *x̃*. By iterated expectations, E[*f*(*X̃*) | *t* = *j*] = E[*f*(*X̃*)·P(*t* = *j* | *X̃*)]/P(*t* = *j*), so ψ_k = E[*a*_k(*X̃*) logit *p*(*X̃*)] with *a*_k as in the main text. This is increasing in logit *p*(*x̃*) where *a*_k(*x̃*) > 0 and decreasing where *a*_k(*x̃*) < 0. Its smallest value, ψ_L, takes the lower endpoint where *a*_k > 0 and the upper endpoint where *a*_k < 0. Its largest value does the reverse. Where *a*_k(*x̃*) = 0 the endpoint chosen does not affect ψ_k. Every value between the two extremes is attained by a continuous choice of *s*. The interval is thus sharp relative to the nonparametric model of the observed data that satisfies *Y* = *D*·*S* and the pointwise floor. Any further restriction on *s*(*x̃*), such as smoothness, monotonicity or links across inputs, can only narrow it. If *t* is a function of *x̃*, then *a*_k > 0 exactly on the upper stratum and *a*_k < 0 exactly on the lower one, which gives the stratum formulas. In general, the stratum formula for the lower endpoint equals E[*a*_k⁺ logit *q* − *a*_k⁻ logit *U*] minus a nonnegative term. It assigns logit *U* ≥ logit *q* with weight P(*t* = 0 | *x̃*)/P(*t* = 0) also where *a*_k > 0, and logit *q* with weight P(*t* = 1 | *x̃*)/P(*t* = 1) also where *a*_k < 0. It is therefore a valid lower bound, and the same argument covers the upper endpoint. When *q* is small, logit *U* ≈ logit *q* − log *s*_min wherever *q*/*s*_min binds, which gives ψ_L ≈ Δ_q + log *s*_min for the stratum formula. The sign is then identified as positive when *s*_min > exp(−Δ_q). The lower endpoint uses *U* only where *a*_k < 0, or, for the stratum formula, only on inputs in the lower stratum. Without a floor elsewhere, the smallest value of logit *p* there is still logit *q*, attained at *s* = 1. Certifying a positive sign thus needs the floor only on that part of the input space. The same holds for ψ_U with the roles reversed.

**Estimated weights.** The sharp set needs the sign of *a*_k, which for image features is a further nuisance. A plug-in with an estimate â_k whose sign is wrong on part of the input space no longer bounds ψ_k from below, because it assigns logit *q* where the true weight is negative. We have not established the finite-sample validity of such a plug-in. Sharpness is a property of the population set with the true *a*_k. The stratum formula needs no such estimate and remains a valid outer bound, so it is the one we report for the image-feature target. Script `vr42_sharp_weighted.py` checks both statements on a known population and compares the two on ISIC-2024 (Section S7). Table S2 uses a one-dimensional input on a grid and a concept equal to the input plus Gaussian noise of the stated scale. Disease and malignant verification are known functions of the input. With no noise the two sets coincide. With noise the stratum formula is wider, both sets contain the true ψ, and the sharp set is attained. Reversing the sign of the weight moves the lower endpoint above the true ψ.

**TABLE S2. Stratum formula and sharp set on a known population.**

| Noise of the concept given the input | Floor | True ψ | Stratum formula [ψ_L, ψ_U] | Sharp set [ψ_L, ψ_U] | Lower endpoint with the weight sign reversed |
| ---: | ---: | ---: | --- | --- | ---: |
| 0.0 | 0.5 | +0.65 | [0.22, 1.63] | [0.22, 1.63] | +1.63 |
| 0.0 | 0.7 | +0.00 | [−0.22, 0.51] | [−0.22, 0.51] | +0.51 |
| 0.0 | 0.9 | −0.43 | [−0.50, −0.28] | [−0.50, −0.28] | −0.28 |
| 0.5 | 0.5 | +0.58 | [0.11, 1.53] | [0.19, 1.46] | +1.46 |
| 0.5 | 0.7 | −0.00 | [−0.24, 0.49] | [−0.20, 0.45] | +0.45 |
| 0.5 | 0.9 | −0.39 | [−0.46, −0.24] | [−0.45, −0.25] | −0.25 |
| 1.0 | 0.5 | +0.46 | [−0.06, 1.35] | [0.17, 1.12] | +1.12 |
| 1.0 | 0.7 | +0.00 | [−0.27, 0.46] | [−0.14, 0.34] | +0.34 |
| 1.0 | 0.9 | −0.30 | [−0.38, −0.17] | [−0.35, −0.20] | −0.20 |
| 2.0 | 0.5 | +0.29 | [−0.30, 1.11] | [0.12, 0.69] | +0.69 |
| 2.0 | 0.7 | +0.00 | [−0.30, 0.42] | [−0.08, 0.21] | +0.21 |
| 2.0 | 0.9 | −0.19 | [−0.28, −0.07] | [−0.22, −0.13] | −0.13 |

**Lesion-level interval.** The first step of the proof gives, for each lesion, P(*D* = 1 | *x̃*) ∈ [*q*(*x̃*), *U*(*x̃*)]. Section S7 uses this interval for a constrained learner.

## S2. Validation of the theory

Table S3 summarizes the checks.

In Table S3, identity errors are maximum absolute errors on the log odds scale. Agreement is the fraction of cases in which the observed region or reversal matches the statement.

**TABLE S3. Validation of the identities.**

| Check | Setting | Result |
| --- | --- | --- |
| Proposition 1, learner gap and general offset | 5,000 random models; symbolic | exact to machine precision; verified symbolically |
| Lemma 1, identity | 20,000 random models | max error 1.8 × 10⁻¹⁵; region agreement 1.000 |
| Proposition 1, Lemmas 1 and 2, symbolic | sympy, positive domain | all 20 statements verified |
| Lemma 2, log-linear likelihood ratio | 48 population learners | max coefficient error 1.1 × 10⁻¹⁶; reversal agreement 1.000 |
| Lemma 2, verification floor 0.5 | 48 population learners | max coefficient error 0.022; reversal agreement 1.000 |
| Phase diagram, population learner | 648 populations | 469 reversals; agreement 0.998 |
| Phase diagram, prediction from observables | 648 populations | sign agreement 0.995 |
| Phase diagram, finite-sample perceptron | 24 populations, 300,000 lesions each | sign agreement 1.000 |
| ISIC-2024 slope identity, joint model | five concepts | residual from −0.15 to +0.13 |

Script `vr10_symbolic.py` checks each statement with sympy (Meurer et al., 2017) on a declared positive domain. Table S4 lists the statements. All were verified. For the attenuation sets, sympy does not reduce the intersection when θ is symbolic, so they were confirmed at θ ∈ {0.1, 1, 7/3, 50}.

**TABLE S4. Symbolic checks.**

| Check | Statement | Result |
| --- | --- | --- |
| P1.1 | logit *f*₀ − logit *f*₂ = log *g* | verified |
| P1.2 | a constant added to both strata cancels in Δ | verified |
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

**Simulation of Proposition 2.** Script `vr23_psi_sim.py` draws populations of 300,000 lesions with prevalence 0.5 percent, five concepts and three noise features. Malignant verification is π¹(*x*) = *s*_true + (1 − *s*_true)·expit(*w*₁ᵀ*c*), with true floors 0.5, 0.7 and 0.9. Benign lesions are flagged and verified with probabilities that depend on appearance, with a mean verification rate of 0.3 percent. Concepts are independent or have pairwise correlation 0.5. Nuisance models are fitted on 200,000 lesions and the sets are evaluated on the other 100,000, at assumed floors 0.3, 0.5, 0.7 and 0.9. The oracle set uses the true *q* and σ, and the plug-in set uses gradient-boosting estimates of both. Table S5 reports how often each set contained the true ψ and how often it certified a sign. No false-sign certification occurred in this design. Containment fell when the assumed floor exceeded the true one, as the proposition implies.

In Table S5, entries are fractions over settings and concepts.

**TABLE S5. Proposition 2 in simulation.**

| Setting | Oracle set contains ψ | Plug-in set contains ψ | Sign certified, oracle / plug-in | Wrong sign certified |
| --- | ---: | ---: | --- | ---: |
| assumed floor at most the true one | 1.00 | 0.93 | 0.86 / 0.74 | 0.00 |
| assumed floor at most the true one, correlated concepts | 1.00 | 0.96 | 1.00 / 1.00 | 0.00 |
| assumed floor above the true one | 0.57 | 0.63 | 1.00 / 1.00 | 0.00 |

**Stress test of Proposition 2.** Script `vr28_psi_stress.py` searches for the failure that matters most, a certified sign opposite to the true one. It uses true floors of 0.1, 0.3, 0.5 and 0.9 and assumed floors from 0.1 to 0.9, so that many settings overstate the floor. Besides the design above, an adversarial design sets the disease effects of four concepts to 0, −0.3, 0.05 and −0.2. Malignant verification rises steeply with these concepts, so the recorded label has a positive contrast while ψ is zero or negative. Three constructions are compared: the oracle set, a gradient-boosting plug-in, and a two-layer perceptron trained for 30 epochs and Platt-calibrated on a held-out validation split, which mirrors the analysis of ISIC-2024. Table S6 reports the results. With a valid floor, no construction certified the wrong sign in either design. With an overstated floor in the adversarial design, all three did in some settings, mostly when the true floor was 0.1 and the assumed one was 0.5 or more. Several of these settings have ψ close to zero, and the others have ψ of −0.44 and −0.65. A valid floor is therefore necessary for a learner-scale disease-relative conclusion, and the simulation offers no protection against an overstated one.

In Table S6, entries are fractions over settings and concepts, and ψ ≤ 0 counts settings whose true target is not positive.

**TABLE S6. Stress test of Proposition 2.**

| Design | Floor | Settings | ψ ≤ 0 | Contains ψ: oracle / GBM / perceptron | Certifies a sign: oracle / GBM / perceptron | Wrong sign: oracle / GBM / perceptron |
| --- | --- | ---: | ---: | --- | --- | --- |
| positive disease effects | valid | 70 | 0 | 1.00 / 0.96 / 0.97 | 0.31 / 0.23 / 0.31 | 0.00 / 0.00 / 0.00 |
| positive disease effects | overstated | 50 | 0 | 0.52 / 0.72 / 0.58 | 0.80 / 0.70 / 0.84 | 0.00 / 0.00 / 0.00 |
| adversarial | valid | 70 | 42 | 1.00 / 0.96 / 0.99 | 0.14 / 0.10 / 0.11 | 0.00 / 0.00 / 0.00 |
| adversarial | overstated | 50 | 30 | 0.44 / 0.56 / 0.50 | 0.54 / 0.46 / 0.48 | 0.20 / 0.14 / 0.14 |

**Semi-synthetic test on the ISIC-2024 inputs.** The simulations above use eight synthetic features and do not reproduce the rare-event and support structure of ISIC-2024. Script `vr34_semisynth.py` keeps the real learner inputs and the real patient split. It also keeps three functions of the input, estimated once from the real data: a calibrated risk logit, a ridge prediction of color variegation, and the verification propensity among recorded negatives. The risk logit comes from the perceptron M0 in one design and from gradient boosting in a second, so that neither estimator of *q* is favored by the data-generating process. Disease is drawn with logit *p*(*x̃*) = α + ℓ(*x̃*) + *b*·*m*(*x̃*). Here ℓ is the standardized risk logit and *m* the standardized color prediction. The intercept α sets the prevalence to 0.2 or 0.5 percent, and *b* sets the true ψ for color variegation to about 0.6, 0 or −0.3. Malignant verification is *s*(*x̃*) = *s*_true + (1 − *s*_true)·expit(−1 + 1.5*m*(*x̃*)), with *s*_true of 0.5 or 0.8, so it rises with the concept, and benign verification is the real propensity. Recorded labels follow *Y* = *D*·*S*. For each of 24 settings per family and design, the same three estimates of *q* as in Section S6 were fitted to the simulated labels. The plug-in set was then evaluated at the true floor against the true ψ. The set uses the stratum formula, which is sharp for the tabular input and an outer set for image features. The containment rate is the share of settings in which the plug-in set contained the true ψ. It describes the plug-in set, not the coverage of a confidence procedure. Low containment with no wrong-sign certification means that the plug-in set was often displaced from the true target, mostly downward, which made certification conservative rather than accurate. M2 and ĝ, a perceptron or gradient boosting, were also fitted. We recorded the lesion-level slope of the calibrated gap on log ĝ, which the population identity fixes at one. Table S7 reports the results.

In Table S7, disease risk is built from the perceptron or from gradient boosting on the ISIC-2024 inputs. A wrong sign counts a certified sign opposite to a true ψ beyond ±0.05.

**TABLE S7. Semi-synthetic test of the plug-in set at the true floor.**

| Disease risk built from | Features | Estimate of *q* | Scenario | Mean ψ_L minus oracle | Containment rate | Sign certified | Wrong sign certified |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| perceptron | tabular | perceptron, Platt | positive target | −0.17 | 0.88 | 0.88 | 0.00 |
| perceptron | tabular | perceptron, Platt | target near zero | −0.15 | 1.00 | 0.00 | 0.00 |
| perceptron | tabular | perceptron, Platt | negative target | −0.06 | 1.00 | 0.38 | 0.00 |
| perceptron | tabular | perceptron, isotonic | positive target | +0.06 | 0.62 | 1.00 | 0.00 |
| perceptron | tabular | perceptron, isotonic | target near zero | −0.19 | 0.88 | 0.12 | 0.00 |
| perceptron | tabular | perceptron, isotonic | negative target | −0.06 | 0.88 | 0.38 | 0.00 |
| perceptron | tabular | gradient boosting, Platt | positive target | −0.31 | 1.00 | 0.62 | 0.00 |
| perceptron | tabular | gradient boosting, Platt | target near zero | −0.13 | 1.00 | 0.00 | 0.00 |
| perceptron | tabular | gradient boosting, Platt | negative target | −0.00 | 0.88 | 0.12 | 0.00 |
| perceptron | image | perceptron, Platt | positive target | −0.23 | 0.88 | 0.50 | 0.00 |
| perceptron | image | perceptron, Platt | target near zero | −0.08 | 1.00 | 0.00 | 0.00 |
| perceptron | image | perceptron, Platt | negative target | +0.02 | 0.88 | 0.25 | 0.00 |
| perceptron | image | perceptron, isotonic | positive target | −0.16 | 0.75 | 0.62 | 0.00 |
| perceptron | image | perceptron, isotonic | target near zero | −0.10 | 1.00 | 0.00 | 0.00 |
| perceptron | image | perceptron, isotonic | negative target | −0.01 | 0.88 | 0.12 | 0.00 |
| perceptron | image | gradient boosting, Platt | positive target | −0.33 | 1.00 | 0.50 | 0.00 |
| perceptron | image | gradient boosting, Platt | target near zero | −0.03 | 1.00 | 0.00 | 0.00 |
| perceptron | image | gradient boosting, Platt | negative target | +0.13 | 0.62 | 0.00 | 0.00 |
| gradient boosting | tabular | perceptron, Platt | positive target | −0.32 | 0.88 | 0.38 | 0.00 |
| gradient boosting | tabular | perceptron, Platt | target near zero | −0.20 | 0.75 | 0.25 | 0.00 |
| gradient boosting | tabular | perceptron, Platt | negative target | −0.14 | 0.75 | 0.12 | 0.00 |
| gradient boosting | tabular | perceptron, isotonic | positive target | −0.42 | 0.88 | 0.38 | 0.00 |
| gradient boosting | tabular | perceptron, isotonic | target near zero | −0.16 | 0.75 | 0.25 | 0.00 |
| gradient boosting | tabular | perceptron, isotonic | negative target | −0.06 | 0.75 | 0.00 | 0.00 |
| gradient boosting | tabular | gradient boosting, Platt | positive target | −0.51 | 0.75 | 0.38 | 0.00 |
| gradient boosting | tabular | gradient boosting, Platt | target near zero | −0.01 | 0.62 | 0.38 | 0.00 |
| gradient boosting | tabular | gradient boosting, Platt | negative target | +0.18 | 0.50 | 0.00 | 0.00 |
| gradient boosting | image | perceptron, Platt | positive target | −0.09 | 1.00 | 0.75 | 0.00 |
| gradient boosting | image | perceptron, Platt | target near zero | +0.02 | 1.00 | 0.00 | 0.00 |
| gradient boosting | image | perceptron, Platt | negative target | +0.10 | 0.75 | 0.00 | 0.00 |
| gradient boosting | image | perceptron, isotonic | positive target | −0.30 | 0.88 | 0.50 | 0.00 |
| gradient boosting | image | perceptron, isotonic | target near zero | −0.01 | 1.00 | 0.00 | 0.00 |
| gradient boosting | image | perceptron, isotonic | negative target | +0.17 | 0.50 | 0.00 | 0.00 |
| gradient boosting | image | gradient boosting, Platt | positive target | −0.36 | 0.75 | 0.50 | 0.00 |
| gradient boosting | image | gradient boosting, Platt | target near zero | +0.01 | 0.75 | 0.25 | 0.00 |
| gradient boosting | image | gradient boosting, Platt | negative target | +0.23 | 0.50 | 0.00 | 0.00 |

Table S8 reports the lesion-level slope of the calibrated gap on log ĝ in the same settings, where the population identity holds exactly. Finite trained learners can depart from slope one even then, and the model class of ĝ moves the slope. We therefore read the slopes on ISIC-2024 as agreement in direction and approximate size only.

In Table S8, the gap is calibrated and regressed on log ĝ in designs where Proposition 1 holds exactly.

**TABLE S8. Lesion-level slope of the learner gap in the semi-synthetic designs.**

| Disease risk built from | Features | Slope on log ĝ, perceptron: median [range] | Gradient boosting: median [range] | Slope on true log g: median |
| --- | --- | --- | --- | ---: |
| perceptron | tabular | 0.83 [0.60, 1.00] | 1.35 [1.03, 1.96] | 0.80 |
| perceptron | image | 0.97 [0.67, 1.40] | 0.98 [0.66, 1.46] | 0.67 |
| gradient boosting | tabular | 0.76 [0.63, 0.94] | 1.35 [0.96, 1.98] | 0.74 |
| gradient boosting | image | 0.95 [0.60, 1.58] | 0.96 [0.53, 1.68] | 0.68 |

**Local violation of the floor.** The designs above satisfy the assumed floor everywhere. Running `vr34_semisynth.py` with the argument `local` keeps the perceptron design with one change. Malignant verification is set to 0.20 among the 30 percent of lesions with the lowest true overall verification propensity. This is below the assumed floor of 0.5 or 0.8, which still holds elsewhere. This mimics a floor that looks reasonable on average but fails where verification is rarest. The oracle set uses the true *q* and σ at the assumed floor, so its errors come from the assumption alone. Table S9 reports containment and certification over the 24 settings per family.

In Table S9, sets are evaluated at the assumed floor. A wrong sign counts a certified sign opposite to a true ψ beyond ±0.05.

**TABLE S9. Semi-synthetic test with the floor violated where verification is rarest.**

| Features | Construction | Containment rate | Sign certified | Wrong sign certified |
| --- | --- | ---: | ---: | ---: |
| tabular | oracle, true nuisances | 0.75 | 0.42 | 0.00 |
| tabular | perceptron, Platt | 0.92 | 0.46 | 0.00 |
| tabular | perceptron, isotonic | 0.62 | 0.46 | 0.00 |
| tabular | gradient boosting, Platt | 0.92 | 0.25 | 0.00 |
| image | oracle, true nuisances | 0.50 | 0.50 | 0.00 |
| image | perceptron, Platt | 0.75 | 0.33 | 0.00 |
| image | perceptron, isotonic | 0.67 | 0.38 | 0.00 |
| image | gradient boosting, Platt | 0.79 | 0.21 | 0.00 |

## S3. Phase diagram design

The concept *c* is evaluated on a grid of 161 points on [−4, 4] with normal weights, and the hidden severity *h* on 16 Gauss-Hermite nodes. Every quantity is an exact population expectation. The disease model is logit P(*D* | *c*, *h*) = α + β*c* + γ*h*, with α solved for the target prevalence. Benign verification is π⁰(*c*) = expit(*b*₀ + *b*′*c*), with *b*₀ solved for a mean of 0.3 percent. Malignant verification is π¹(*c*, *h*) = *s*_lo + (1 − *s*_lo)·expit(*ac* + γ*h*). The grid crosses β ∈ {0.25, 0.5, 1}, *b*′ ∈ {0, 0.5, 1, 1.5, 2, 3}, *s*_lo ∈ {0.5, 0.7, 0.9}, *a* ∈ {−1, 0, 1}, prevalence ∈ {0.002, 0.02} and γ ∈ {0, 1}. Tertile cut points are ±0.4307. Population-optimal logistic learners are fitted by Newton iterations on the weighted grid.

With no appearance-driven biopsy of benign lesions (*b*′ = 0), no population reversed. At *b*′ = 1, 84 percent did, and at *b*′ of 1.5 or more, all did. The recorded-label learner reversed in none of the 648 populations. The finite-sample check draws 300,000 lesions at each of 24 random grid points and trains M0 and M2 perceptrons on *c* plus two noise features. In the 15 populations predicted to reverse, the M2 contrast ranged from −4.60 to −0.23 and the M0 contrast from +0.51 to +2.85. In the 9 predicted not to reverse, both contrasts were positive, with M2 at least +0.11.

Whether the population-optimal verified-only learner reversed agreed with Lemma 1 in 647 of 648 populations, and the remaining population sits on the boundary (Fig. S1).

In Fig. S1, each point is one of 648 simulated populations, shaded by the regions of Lemma 1. The circled point is the single population where the learner disagrees with the region. The figure file is `figures/figS1_phase_diagram.png`.

**Fig. S1.** Reversal boundary in simulation.

## S4. Sensitivity to the concept definition

Seven definitions were applied to the same predictions. Four split the concept: tertile (primary), quartile, quintile and median splits. Three are slopes of the logit: an ordinary least squares slope on the standardized concept, the same slope after removing each patient's mean, and a slope on the concept's cohort percentile rank. Table S10 reports color variegation and size. Color variegation reverses under every definition in every family; the within-patient slope shrinks its verified-only contrast but keeps its sign. Size keeps the reversal under every split-based definition, but not under slope-based definitions. With tabular and image features its verified-only slope is between −0.06 and +0.01, and it is positive within patients with image features. In the linear-probe and fine-tuned families it is positive. So the size reversal is carried by the outer tertiles. This is a sensitivity of the estimand, not a failure of the primary inference, which is defined on tertiles.

In Table S10, each cell gives Δ_M0 / Δ_M2, and values condition on the fitted models.

**TABLE S10. Learned contrasts under seven concept definitions.**

| Family | Concept | tertile | quartile | quintile | median | slope | within-patient slope | rank slope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tabular | Color variegation | +1.02 / −0.95 | +1.23 / −1.08 | +1.42 / −1.16 | +0.72 / −0.71 | +0.61 / −0.37 | +0.69 / −0.19 | +1.55 / −1.41 |
| Tabular | Size | +0.65 / −0.46 | +0.90 / −0.54 | +1.08 / −0.59 | +0.42 / −0.33 | +0.65 / −0.06 | +0.70 / −0.04 | +1.13 / −0.69 |
| Image | Color variegation | +1.64 / −0.66 | +1.85 / −0.75 | +1.97 / −0.81 | +1.21 / −0.49 | +0.75 / −0.27 | +0.69 / −0.16 | +2.40 / −0.98 |
| Image | Size | +1.56 / −0.32 | +1.92 / −0.37 | +2.14 / −0.40 | +1.12 / −0.23 | +0.78 / −0.02 | +0.66 / +0.01 | +2.47 / −0.47 |
| Linear probe | Color variegation | +0.48 / −0.68 | +0.54 / −0.78 | +0.59 / −0.83 | +0.37 / −0.50 | +0.24 / −0.27 | +0.29 / −0.14 | +0.71 / −1.01 |
| Linear probe | Size | +0.67 / −0.23 | +0.81 / −0.27 | +0.90 / −0.30 | +0.48 / −0.16 | +0.41 / +0.01 | +0.40 / +0.04 | +1.07 / −0.34 |
| Fine-tuned | Color variegation | +1.12 / −2.70 | +1.31 / −3.22 | +1.45 / −3.57 | +0.80 / −1.88 | +0.63 / −1.04 | +0.75 / −0.29 | +1.66 / −4.19 |
| Fine-tuned | Size | +1.57 / −0.23 | +1.94 / −0.27 | +2.22 / −0.33 | +1.11 / −0.10 | +1.06 / +0.62 | +1.02 / +0.73 | +2.55 / −0.30 |

## S5. Learner settings and localization of the reversal

Under common-readout and head-swap diagnostics on the fine-tuned encoders, the training regime of the decision head explained a larger change in the learned contrast than the origin of the encoder. The diagnostics did not support a general representation-level reversal, and they cannot exclude representation-level contributions.

Table S11 lists the training settings.

In Table S11, M0 and M2 share every setting except the training lesions.

**TABLE S11. Training settings of the learner families.**

| Setting | Frozen-feature perceptrons | Linear probe | Fine-tuned ResNet-50 |
| --- | --- | --- | --- |
| Input | 47 tabular features, or 2,048 frozen ImageNet features | 128 × 128 tile | 128 × 128 tile |
| Architecture | two hidden layers of 128 (tabular) or 256 (image), GELU, dropout 0.1 | frozen ResNet-50, linear head | ResNet-50, linear head |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.0001 | AdamW, head 0.001 | AdamW, backbone 0.0001, head 0.001, one-cycle |
| Budget | up to 30 epochs, early stopping on validation loss | 2,000 steps of 64 | 4,000 steps of 64 |
| Sampling | natural | positive fraction at least 10 percent | positive fraction at least 10 percent |
| Augmentation | none | flips, 90-degree rotations, crops; no color change | same |
| Seeds | 0, 1, 2 | 0, 1, 2 | 0, 1, 2; seeds 3 and 4 added for the head swap |

Test AUROC for the recorded label was 0.941 under M0 and 0.708 under M2 with tabular features. With frozen image features it was 0.815 and 0.689. With fine-tuning it was 0.912 to 0.918 and 0.731 to 0.752 for seeds 0 to 2.

**Common readout.** On each encoder's penultimate features for the shared probe set, we fitted a ridge regression to a target that is the same for both encoders. The first target is the logit of the recorded-label risk *q*(*x*), from a tabular gradient-boosting model fitted out of fold. The second is the logit of verified-lesion risk *q*/σ. We also fitted ridge directions for each concept. The alignment is the cosine between the readout direction and the concept direction. A representation reversal would appear as opposite signs of this alignment between the M0 and M2 encoders. Table S12 reports these diagnostics for seeds 0 to 2.

In Table S12, values are averaged over seeds 0 to 2, and the last column counts the seeds in which M0 and M2 have opposite signs.

**TABLE S12. Representation diagnostics after fine-tuning.**

| Diagnostic | Concept | M0 | M2 | Seeds with opposite sign |
| --- | --- | ---: | ---: | ---: |
| Common readout, recorded-label risk | Color variegation | +0.404 | +0.454 | 0 of 3 |
| Common readout, recorded-label risk | Lesion-skin contrast | +0.278 | +0.324 | 0 of 3 |
| Common readout, recorded-label risk | Asymmetry | −0.050 | −0.040 | 0 of 3 |
| Common readout, recorded-label risk | Border irregularity | +0.061 | +0.088 | 0 of 3 |
| Common readout, verified-lesion risk | Color variegation | −0.139 | −0.166 | 0 of 3 |
| Common readout, verified-lesion risk | Lesion-skin contrast | −0.181 | −0.212 | 0 of 3 |
| Common readout, verified-lesion risk | Asymmetry | +0.130 | +0.122 | 0 of 3 |
| Common readout, verified-lesion risk | Border irregularity | +0.141 | +0.124 | 0 of 3 |
| Layer-3 shift, output change | Color variegation | +0.035 | +0.027 | 0 of 3 |
| Layer-3 shift, output change | Lesion-skin contrast | +0.037 | −0.027 | 3 of 3 |
| Layer-3 shift, output change | Asymmetry | −0.003 | +0.037 | 2 of 3 |
| Layer-3 shift, output change | Border irregularity | +0.001 | +0.046 | 2 of 3 |

Concept decodability was the same under both regimes. Cross-validated R² ranged from 0.44 to 0.48 for color variegation, 0.73 to 0.77 for lesion-skin contrast, 0.33 to 0.37 for asymmetry and 0.35 to 0.38 for border irregularity. Linear centered kernel alignment between M0 and M2 features was 0.79, 0.22 and 0.74 across seeds 0 to 2. Alignment between two seeds of M0 was 0.39, against 0.98 for two seeds of M2. Fine-tuning under M0 is thus much less stable across seeds, and we treat alignment measures only as secondary diagnostics. Fig. S2 plots them per seed.

**Head swap.** For each of five seeds, both fine-tuned encoders were frozen. New linear heads were trained on their penultimate features under both regimes, with the linear-probe recipe of Table S11, and all four pairings were evaluated on the same probe set. The head effect is the mean contrast under M2 heads minus that under M0 heads, averaged over the two encoders. The encoder effect is the mean contrast on the M2 encoder minus that on the M0 encoder, averaged over the two heads. Retrained heads are new models, so these effects summarize the diagnostic and are not a unique causal decomposition of the original networks. Table S13 reports the mean contrast of each pairing, both effects with their range over seeds, and the number of seeds in which the sign followed the head. Heads retrained under M0 learned contrasts close to zero, so the sign criterion is fragile, and the effect sizes are the informative summary.

In Table S13, contrasts and effects are means over five seeds, and brackets give the range over seeds.

**TABLE S13. Head swap on fine-tuned encoders.**

| Concept | Encoder M0, head M0 | Encoder M0, head M2 | Encoder M2, head M0 | Encoder M2, head M2 | Head effect [seed range] | Encoder effect [seed range] | Seeds where sign follows the head |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| Color variegation | +0.08 | −3.10 | +0.01 | −2.98 | −3.08 [−3.42, −2.74] | +0.03 [−0.29, 0.40] | 3 of 5 |
| Size | +0.14 | −2.01 | +0.08 | −0.98 | −1.61 [−1.89, −1.34] | +0.48 [0.22, 0.89] | 5 of 5 |
| Lesion-skin contrast | −0.01 | −3.78 | −0.09 | −5.72 | −4.70 [−5.18, −4.17] | −1.01 [−1.42, −0.57] | 1 of 5 |
| Asymmetry | +0.04 | +3.43 | +0.11 | +5.91 | +4.60 [4.17, 5.10] | +1.28 [0.81, 1.64] | 1 of 5 |
| Border irregularity | +0.10 | +3.44 | +0.15 | +6.66 | +4.92 [4.50, 5.41] | +1.64 [1.19, 1.99] | 1 of 5 |

In Fig. S2, the panels show the decision contrast, the layer-3 intervention and the two common readouts. The figure file is `figures/figS2_finetune.png`.

**Fig. S2.** Fine-tuned diagnostics per seed.

## S6. Joint bootstrap and tests of alternative explanations

**Joint bootstrap.** For the two primary families, each of 5,000 replicates resampled training patients with replacement and refitted M0 and M2 with early stopping on the full validation set. Each replicate then Platt-calibrated M0 on the full validation set and M2 on the verified validation lesions, resampled test patients and recomputed Δ, the plug-in set of Proposition 2 and the support-restricted contrasts. Replicate *b* resampled patients with generator seed 50,000 + *b* and trained both learners with seed *b*. Optimizer randomness thus varies independently across replicates, and the bootstrap distribution mixes patient resampling with training randomness. These 5,000 replicates alone describe the randomized one-seed training procedure and serve as a check. Point estimates average the three original fits with seeds 0 to 2. They target the same expectation over seeds with less variance but do not have the sampling distribution of the one-seed statistic. Script `vr39_seed_variance.py` therefore refitted each of the 5,000 replicates per family with two further seeds, *b* + 100,000 and *b* + 200,000, on the same resampled patients. Script `vr43_three_seed_primary.py` then bootstraps the three-seed average, which is the statistic of the point estimate. These three-seed replicates give the primary intervals and labels of Table 1. Table S14 splits the variance of the primary contrasts and compares the Bonferroni intervals and labels of the one-seed and three-seed statistics on the same replicates. Table S15 gives 95 percent percentile intervals and Bonferroni-adjusted percentile intervals at level 1 − 0.05/6, for raw and calibrated contrasts. Platt slopes over seeds 0 to 2 were 0.95 to 0.98 for tabular M0 and 0.86 to 1.12 for tabular M2, so tabular learners were close to calibrated. For image M0 they were 0.44 to 0.45 and for image M2 0.49 to 0.85, so the image learners, and image M0 in particular, were overconfident. Calibration rescales each learner's logit by a positive factor, so it does not change the sign of Δ.

In Table S14, each of the 5,000 joint replicates per family has three seeds, and labels follow the primary rule on these replicates.

**TABLE S14. Seed and patient-resampling variability of the primary contrasts.**

| Features | Concept | Contrast | SD between seeds, same resample | SD of single-seed replicates | Share of variance from seeds | Bonferroni interval, single seed | Bonferroni interval, three-seed average | Label, single / three-seed |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| tabular | Color variegation | Δ_M0 | 0.28 | 0.42 | 0.43 | [−0.15, 2.08] | [0.01, 1.88] | 95 percent only / Bonferroni (unresolved) |
| tabular | Color variegation | Δ_M2 | 0.09 | 0.20 | 0.19 | [−1.54, −0.46] | [−1.50, −0.51] | 95 percent only / Bonferroni (unresolved) |
| tabular | Size | Δ_M0 | 0.23 | 0.36 | 0.41 | [−0.36, 1.53] | [−0.25, 1.38] | no / no |
| tabular | Size | Δ_M2 | 0.07 | 0.15 | 0.19 | [−0.83, −0.03] | [−0.79, −0.07] | no / no |
| tabular | Lesion-skin contrast | Δ_M0 | 0.27 | 0.49 | 0.31 | [−0.92, 1.68] | [−0.76, 1.60] | no / no |
| tabular | Lesion-skin contrast | Δ_M2 | 0.11 | 0.25 | 0.21 | [−2.24, −0.90] | [−2.17, −0.92] | no / no |
| image | Color variegation | Δ_M0 | 0.20 | 0.44 | 0.22 | [0.10, 2.40] | [0.26, 2.33] | Bonferroni / Bonferroni |
| image | Color variegation | Δ_M2 | 0.15 | 0.20 | 0.58 | [−1.37, −0.31] | [−1.24, −0.42] | Bonferroni / Bonferroni |
| image | Size | Δ_M0 | 0.16 | 0.33 | 0.25 | [0.38, 2.14] | [0.55, 2.10] | Bonferroni / Bonferroni |
| image | Size | Δ_M2 | 0.11 | 0.14 | 0.67 | [−0.81, −0.10] | [−0.72, −0.16] | Bonferroni / Bonferroni |
| image | Lesion-skin contrast | Δ_M0 | 0.25 | 0.54 | 0.20 | [−0.79, 2.00] | [−0.63, 1.93] | no / no |
| image | Lesion-skin contrast | Δ_M2 | 0.20 | 0.27 | 0.56 | [−2.01, −0.61] | [−1.88, −0.73] | no / no |

In Table S15, the 5,000 replicates per family target the randomized one-seed training procedure, the primary three-seed intervals are in Table 1, and robustness is judged on the raw contrasts.

**TABLE S15. One-seed joint bootstrap intervals for the frozen-feature families.**

| Features | Concept | Δ_M0: estimate, 95% / Bonferroni | Δ_M2: estimate, 95% / Bonferroni | Calibrated Δ_M0 / Δ_M2, 95% | Robust, 95% / Bonferroni |
| --- | --- | --- | --- | --- | --- |
| tabular | Color variegation | +1.02, [0.11, 1.77] / [−0.15, 2.08] | −0.95, [−1.37, −0.59] / [−1.54, −0.46] | [0.10, 1.58] / [−1.30, −0.59] | yes / no |
| tabular | Size | +0.65, [−0.13, 1.27] / [−0.36, 1.53] | −0.46, [−0.71, −0.13] / [−0.83, −0.03] | [−0.12, 1.11] / [−0.68, −0.13] | no / no |
| tabular | Lesion-skin contrast | +0.52, [−0.56, 1.35] / [−0.92, 1.68] | −1.50, [−2.00, −1.03] / [−2.24, −0.90] | [−0.50, 1.21] / [−1.86, −1.06] | no / no |
| tabular | Asymmetry | −0.29, [−1.35, 0.81] / [−1.71, 1.19] | +1.36, [0.96, 1.95] / [0.83, 2.15] | [−1.18, 0.73] / [0.99, 1.79] | no / no |
| tabular | Border irregularity | −0.19, [−1.33, 0.95] / [−1.72, 1.36] | +1.51, [1.08, 2.14] / [0.94, 2.38] | [−1.19, 0.84] / [1.11, 1.96] | no / no |
| image | Color variegation | +1.64, [0.44, 2.13] / [0.10, 2.40] | −0.66, [−1.21, −0.42] / [−1.37, −0.31] | [0.17, 0.82] / [−0.59, −0.25] | yes / yes |
| image | Size | +1.56, [0.69, 1.97] / [0.38, 2.14] | −0.32, [−0.70, −0.16] / [−0.81, −0.10] | [0.27, 0.77] / [−0.33, −0.10] | yes / yes |
| image | Lesion-skin contrast | +1.01, [−0.40, 1.69] / [−0.79, 2.00] | −1.07, [−1.79, −0.76] / [−2.01, −0.61] | [−0.17, 0.64] / [−0.91, −0.44] | no / no |
| image | Asymmetry | −0.50, [−1.06, 0.69] / [−1.33, 1.05] | +1.09, [0.80, 1.79] / [0.66, 2.00] | [−0.39, 0.28] / [0.47, 0.90] | no / no |
| image | Border irregularity | −0.09, [−0.71, 1.11] / [−0.98, 1.42] | +1.18, [0.87, 1.90] / [0.70, 2.13] | [−0.26, 0.45] / [0.50, 0.96] | no / no |

**Validation resampled, three estimates of *q*.** Early stopping and Platt calibration use the validation set, and M2 is calibrated on its verified lesions only. Script `vr30_q_bootstrap.py` repeated the analysis in 1,000 replicates per family that resampled training, validation and test patients independently within each split. In every replicate it computed the plug-in ψ_L for three estimates of *q*. Two are the perceptron M0 with Platt or with isotonic calibration. The third is gradient boosting of *Y* on the tabular features, or on the first 64 principal components of the image features, with Platt calibration. Table S16 compares the robustness labels. For each estimate, it also reports the plug-in tipping floor s∗ on the seed-averaged curve, its 95th percentile over replicates and the bootstrap stability threshold. The last columns give s∗ within the supported region and the number of replicates in which ψ_L(s) > 0 and Δ_M2 < 0 held together. The bootstrap stability threshold is the smallest floor at which ψ_L(s) > 0 in at least 95 percent of replicates. Because ψ_L increases in s in every replicate, this threshold equals the 95th percentile of s∗ when replicates in which no floor up to one makes ψ_L positive are counted as above one. Both are computed this way. It describes sampling variability for one specification of *q*. It is not a confidence bound for ψ or for the identified set, and it does not account for the choice among estimates of *q*. Its Monte Carlo range is the 2.5th to 97.5th percentile of the same statistic over 500 resamples of the stored replicates.

In Table S16, training, validation and test patients are resampled for three estimates of *q*, and "none" means that no floor up to one made ψ_L positive.

**TABLE S16. Robustness and plug-in tipping floor with all three splits resampled.**

| Features | Concept | Robust, one seed per replicate: train and test / all three splits | Estimate of *q* | s∗: point, 95th percentile | Bootstrap stability threshold | Within support, point s∗ at the 1st / 5th / 10th percentile | Joint count at s = 0.7 / 0.8 / 0.9 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tabular | Color variegation | 95 percent only / 95 percent only | perceptron, Platt | 0.38, 0.82 | 0.82 | 0.72 / none / none | 883 / 940 / 977 of 1000 |
| tabular | Color variegation | 95 percent only / 95 percent only | perceptron, isotonic | 0.28, 0.80 | 0.80 | 0.66 / none / none | 896 / 956 / 974 of 1000 |
| tabular | Color variegation | 95 percent only / 95 percent only | gradient boosting, Platt | 0.75, 0.97 | 0.97 | 0.79 / 0.84 / 0.88 | 83 / 174 / 380 of 1000 |
| tabular | Size | no / no | perceptron, Platt | 0.54, none | none | 0.72 / 0.73 / 0.76 | 664 / 805 / 886 of 1000 |
| tabular | Size | no / no | perceptron, isotonic | 0.43, 1.00 | 1.00 | 0.65 / 0.66 / 0.69 | 728 / 841 / 898 of 1000 |
| tabular | Size | no / no | gradient boosting, Platt | 0.74, 0.97 | 0.97 | 0.75 / 0.76 / 0.79 | 97 / 182 / 405 of 1000 |
| tabular | Lesion-skin contrast | no / no | perceptron, Platt | 0.61, none | none | none / none / none | 541 / 656 / 740 of 1000 |
| tabular | Lesion-skin contrast | no / no | perceptron, isotonic | 0.51, none | none | none / none / none | 541 / 656 / 745 of 1000 |
| tabular | Lesion-skin contrast | no / no | gradient boosting, Platt | 0.85, 0.99 | 0.99 | 0.91 / 0.98 / 1.00 | 19 / 61 / 188 of 1000 |
| image | Color variegation | Bonferroni / 95 percent only | perceptron, Platt | 0.49, 0.82 | 0.82 | 0.71 / 0.95 / 0.86 | 794 / 931 / 981 of 1000 |
| image | Color variegation | Bonferroni / 95 percent only | perceptron, isotonic | 0.58, 0.85 | 0.85 | 0.77 / 0.98 / 0.90 | 559 / 881 / 984 of 1000 |
| image | Color variegation | Bonferroni / 95 percent only | gradient boosting, Platt | 0.84, none | none | 0.92 / 0.97 / 0.93 | 2 / 5 / 107 of 1000 |
| image | Size | Bonferroni / Bonferroni | perceptron, Platt | 0.50, 0.76 | 0.76 | 0.61 / 0.62 / 0.56 | 880 / 974 / 998 of 1000 |
| image | Size | Bonferroni / Bonferroni | perceptron, isotonic | 0.57, 0.79 | 0.79 | 0.65 / 0.67 / 0.61 | 674 / 960 / 999 of 1000 |
| image | Size | Bonferroni / Bonferroni | gradient boosting, Platt | 0.79, 0.99 | 0.99 | 0.82 / 0.83 / 0.80 | 0 / 21 / 227 of 1000 |
| image | Lesion-skin contrast | no / no | perceptron, Platt | 0.64, none | none | 0.97 / none / none | 328 / 544 / 747 of 1000 |
| image | Lesion-skin contrast | no / no | perceptron, isotonic | 0.72, none | none | 1.00 / none / none | 133 / 376 / 673 of 1000 |
| image | Lesion-skin contrast | no / no | gradient boosting, Platt | 0.94, none | none | none / none / none | 0 / 2 / 9 of 1000 |

**Subsampling.** Resampling patients with replacement and refitting networks with early stopping may not behave as the percentile bootstrap assumes. Script `vr36_subsample.py` drew half of the training patients and half of the test patients without replacement in 300 replicates per family, and refitted M0 and M2. It then formed m-out-of-n intervals (Politis et al., 1999) [θ̂ − Q_{1−a/2}, θ̂ − Q_{a/2}] for a = 0.05 and a = 0.05/6. Here θ̂ is the point estimate of Table 1, θ*_m the estimate from a half-sample, and Q_a the a-quantile of √(m/n)(θ*_m − θ̂) with m/n = 1/2. The rescaling assumes that the learned contrast converges at the root-n rate, which fitted networks with early stopping need not satisfy. A learner trained on half of the patients may also target a different contrast. We therefore use subsampling only to check that no bootstrap-robust label is reversed, and not to add robust cells. Table S17 reports the intervals.

In Table S17, intervals are the rescaled half-sampling intervals defined above.

**TABLE S17. Primary comparisons under half-sampling of patients.**

| Features | Concept | Δ_M0, Bonferroni-scaled interval | Δ_M2, Bonferroni-scaled interval | Sign separation of subsampling intervals, 95% / Bonferroni-scaled | Same sign in subsamples |
| --- | --- | --- | --- | --- | ---: |
| tabular | Color variegation | [0.56, 1.93] | [−1.34, −0.50] | yes / yes | 0.03 |
| tabular | Size | [0.32, 1.43] | [−0.75, −0.21] | yes / yes | 0.10 |
| tabular | Lesion-skin contrast | [−0.38, 1.30] | [−1.94, −0.94] | no / no | 0.15 |
| image | Color variegation | [1.11, 2.61] | [−0.91, −0.23] | yes / yes | 0.00 |
| image | Size | [1.09, 2.23] | [−0.43, −0.09] | yes / yes | 0.00 |
| image | Lesion-skin contrast | [0.32, 2.27] | [−1.51, −0.50] | yes / yes | 0.06 |

**Monte Carlo error.** The Bonferroni endpoints are the 0.42 and 99.58 percentiles of the replicates, so they rest on few replicates in each tail. For the primary three-seed bootstrap, script `vr43_three_seed_primary.py` resampled the 5,000 stored three-seed replicates per family 2,000 times and recomputed the Bonferroni endpoints and the label in each resample. It reports the Monte Carlo standard error of each endpoint and the share of resamples that reproduce the label of Table 1. A label is called resolved at the Monte Carlo resolution when this share is at least 0.95. Table S18 gives the results. For the one-seed check, script `vr29_bootstrap_audit.py` applied the same procedure to the 5,000 stored one-seed replicates. It also reports the quantiles of s∗ and the joint event above. Table S19 gives the results.

In Table S18, the audit covers the replicates of Table 1 with 2,000 resamples of the stored replicates.

**TABLE S18. Monte Carlo audit of the primary three-seed bootstrap.**

| Features | Concept | Replicates | Bonferroni sign-separated | Label reproduced | Monte Carlo SE of endpoints, Δ_M0 lower / upper, Δ_M2 lower / upper | Resolved at the Monte Carlo resolution |
| --- | --- | ---: | --- | ---: | --- | --- |
| tabular | Color variegation | 5,000 | yes | 0.72 | 0.018 / 0.032, 0.016 / 0.014 | no |
| tabular | Size | 5,000 | no | 1.00 | 0.018 / 0.025, 0.008 / 0.009 | yes |
| tabular | Lesion-skin contrast | 5,000 | no | 1.00 | 0.049 / 0.051, 0.018 / 0.025 | yes |
| image | Color variegation | 5,000 | yes | 1.00 | 0.026 / 0.024, 0.017 / 0.013 | yes |
| image | Size | 5,000 | yes | 1.00 | 0.019 / 0.018, 0.012 / 0.005 | yes |
| image | Lesion-skin contrast | 5,000 | no | 1.00 | 0.033 / 0.066, 0.017 / 0.014 | yes |

In Table S19, the table also gives the distribution of the plug-in tipping floor over the one-seed replicates.

**TABLE S19. Monte Carlo error of the one-seed check.**

| Features | Concept | Robust | Label reproduced | Monte Carlo SE of endpoints, Δ_M0 lower / upper, Δ_M2 lower / upper | s∗ quantiles 5 / 25 / 50 / 75 / 95 | ψ_L(s) > 0 and Δ_M2 < 0 at s = 0.5 / 0.7 / 0.9 |
| --- | --- | --- | ---: | --- | --- | --- |
| tabular | Color variegation | no | 1.00 | 0.032 / 0.059, 0.018 / 0.013 | 0.24 / 0.34 / 0.44 / 0.57 / 0.81 | 0.64 / 0.90 / 0.97 |
| tabular | Size | no | 1.00 | 0.031 / 0.034, 0.013 / 0.009 | 0.37 / 0.49 / 0.61 / 0.76 / none | 0.27 / 0.68 / 0.89 |
| tabular | Lesion-skin contrast | no | 1.00 | 0.044 / 0.036, 0.022 / 0.015 | 0.34 / 0.51 / 0.70 / 0.92 / none | 0.24 / 0.52 / 0.74 |
| image | Color variegation | yes | 0.99 | 0.042 / 0.026, 0.021 / 0.009 | 0.47 / 0.54 / 0.60 / 0.67 / 0.80 | 0.13 / 0.83 / 0.99 |
| image | Size | yes | 1.00 | 0.041 / 0.010, 0.017 / 0.005 | 0.49 / 0.55 / 0.60 / 0.65 / 0.74 | 0.08 / 0.90 / 1.00 |
| image | Lesion-skin contrast | no | 1.00 | 0.043 / 0.053, 0.024 / 0.017 | 0.57 / 0.67 / 0.77 / 0.89 / none | 0.01 / 0.32 / 0.77 |

**Size-matched control.** Every malignant training lesion is verified, so M0 and M2 differ in the size of the training set and in the source of the recorded negatives. In each of 20 replicates we drew 178 of the 222 malignant training lesions and 320 recorded negatives. The negatives came from one of three pools: all 241,263 recorded-negative training lesions, the 12,271 flagged but unverified ones, or the 400 verified ones. The malignant lesions were shared by the three arms within a replicate, and validation lesions were drawn by the same rule. Training used the same code, batch rule and early stopping as M2. Table S20 reports the mean learned contrast on the test population, the 2.5 and 97.5 percentiles over replicates and the number of positive replicates.

**TABLE S20. Size-matched control, 20 replicates per arm.**

| Features | Concept | Benign from all lesions | Benign from flagged, unverified | Benign from verified (M2) |
| --- | --- | --- | --- | --- |
| tabular | Color variegation | +1.22 [0.60, 2.14], 20 of 20 positive | −0.79 [−1.21, −0.38], 0 of 20 positive | −0.94 [−1.12, −0.70], 0 of 20 positive |
| tabular | Size | +0.95 [0.42, 1.60], 20 of 20 positive | −0.11 [−0.46, 0.21], 6 of 20 positive | −0.41 [−0.55, −0.18], 0 of 20 positive |
| tabular | Lesion-skin contrast | +0.46 [−0.18, 1.11], 18 of 20 positive | −1.95 [−2.45, −1.34], 0 of 20 positive | −1.54 [−1.78, −1.29], 0 of 20 positive |
| tabular | Asymmetry | −0.19 [−1.00, 0.38], 8 of 20 positive | +1.88 [1.35, 2.27], 20 of 20 positive | +1.38 [1.08, 1.72], 20 of 20 positive |
| tabular | Border irregularity | +0.04 [−0.74, 0.68], 13 of 20 positive | +2.14 [1.59, 2.52], 20 of 20 positive | +1.56 [1.26, 1.90], 20 of 20 positive |
| image | Color variegation | +0.59 [0.31, 0.90], 20 of 20 positive | −0.82 [−1.05, −0.58], 0 of 20 positive | −0.63 [−0.81, −0.40], 0 of 20 positive |
| image | Size | +0.78 [0.35, 1.02], 20 of 20 positive | −0.28 [−0.47, −0.11], 0 of 20 positive | −0.34 [−0.46, −0.16], 0 of 20 positive |
| image | Lesion-skin contrast | +0.01 [−0.48, 0.28], 12 of 20 positive | −1.50 [−2.05, −1.01], 0 of 20 positive | −0.98 [−1.32, −0.67], 0 of 20 positive |
| image | Asymmetry | +0.31 [−0.18, 0.83], 17 of 20 positive | +1.58 [1.06, 2.13], 20 of 20 positive | +1.01 [0.71, 1.36], 20 of 20 positive |
| image | Border irregularity | +0.60 [−0.11, 1.18], 18 of 20 positive | +1.76 [1.16, 2.35], 20 of 20 positive | +1.07 [0.78, 1.46], 20 of 20 positive |

**Quantitative prediction for the arms.** Relative to the random arm, the flagged and verified arms select recorded negatives with *r*₀(*x*) proportional to the probability that a recorded negative with appearance *x* belongs to the arm's pool. Proposition 1 then predicts the shift in Δ from the random arm as minus the stratum contrast of log ĝ_arm. Here ĝ_arm(*x̃*) is that probability, estimated with the same perceptron and input as the learner. The arms were rerun with 10 replicates and calibrated learners for this comparison. Table S21 compares observed and predicted shifts for all five concepts.

**TABLE S21. Observed and predicted shift of the learned contrast from the random arm, calibrated learners.**

| Features | Arm | Concept | Observed shift from random arm | Predicted shift |
| --- | --- | --- | ---: | ---: |
| tabular | verified | Color variegation | −2.19 | −1.55 |
| tabular | verified | Size | −1.32 | −0.63 |
| tabular | verified | Lesion-skin contrast | −2.04 | −1.76 |
| tabular | verified | Asymmetry | +1.60 | +0.99 |
| tabular | verified | Border irregularity | +1.60 | +1.13 |
| tabular | flagged, unverified | Color variegation | −2.07 | −2.82 |
| tabular | flagged, unverified | Size | −1.02 | −2.27 |
| tabular | flagged, unverified | Lesion-skin contrast | −2.56 | −2.82 |
| tabular | flagged, unverified | Asymmetry | +2.12 | +1.98 |
| tabular | flagged, unverified | Border irregularity | +2.23 | +1.79 |
| image | verified | Color variegation | −1.01 | −1.57 |
| image | verified | Size | −0.90 | −1.09 |
| image | verified | Lesion-skin contrast | −0.88 | −1.66 |
| image | verified | Asymmetry | +0.66 | +1.44 |
| image | verified | Border irregularity | +0.51 | +1.35 |
| image | flagged, unverified | Color variegation | −1.20 | −1.19 |
| image | flagged, unverified | Size | −0.84 | −0.83 |
| image | flagged, unverified | Lesion-skin contrast | −1.40 | −1.37 |
| image | flagged, unverified | Asymmetry | +1.24 | +1.14 |
| image | flagged, unverified | Border irregularity | +1.20 | +1.09 |

**Controlled selection dose.** Script `vr46_dose_fixed_val.py` gives the primary version. For each of the three primary concepts and both frozen-feature families, 20 replicates drew 178 malignant training lesions and a replicate-specific validation set. The validation set held malignant lesions and uniformly drawn recorded negatives. Both draws were shared by all doses of the replicate and were not selected by η. Each recorded-negative training lesion *i* received one uniform variable *U*_i per replicate. It entered training at dose η when *U*_i < π_η,i = min{1, κ_η exp(η*z*_k,i)}, for η ∈ {−1, −0.5, 0, 0.5, 1}, with κ_η set so that 320 are expected. These common random numbers couple the draws at different doses. The optimizer seed was also shared, so a replicate fixes ω of Section III-B and the five fits are paired potential outcomes. Color variegation was repeated with 1,000 and 3,000 expected recorded negatives. Early stopping used the replicate-specific validation set. The outcome is the raw logit contrast. Platt calibration on the replicate-specific validation set targets the unselected distribution rather than the selected training distribution of Proposition 1. At η = 1, and at no other dose, its fitted slope was negative in 29 of 200 fits, which reverses the sign of a calibrated contrast. We report calibrated contrasts for completeness only. The prediction for the tabular learner is minus the stratum contrast of log π_η(*z*_k) on the test population, the exact Bayes-optimal shift of the training distribution. For the image learner it is the first-order input-aware form of Section S1. The conditional mean *m*_k is estimated by ridge regression on recorded negatives of resampled training patients in each replicate. The estimate τ̂_k(1, −1) is the mean over replicates of the paired difference Δ_k(1, ω) − Δ_k(−1, ω), with a percentile bootstrap over the 20 replicates, 10,000 resamples; ratio intervals use 2,000. These intervals are conditional on the realized cohort, patient split and test population and summarize the paired pipeline randomness across replicates, including the validation-set draw used for early stopping. Table S22 reports all cells, and Table S23 compares observed and predicted slopes for all five concepts.

In Table S22, the replicate-specific validation set is shared across doses, and replicates are paired. Slopes and τ̂(1, −1) are for the raw logit contrast of the selected concept. Ranges are 2.5 and 97.5 percentiles over replicates, and bracketed intervals are bootstrap intervals over the 20 replicates, conditional on the cohort, patient split and test population. The prediction is the Bayes-optimal training-distribution shift. Raw finite-network logits need not equal the Bayes-optimal log-odds of the selected training distribution, so the observed-over-predicted ratio is descriptive. It does not test the magnitude predicted by Proposition 1.

**TABLE S22. Selection dose on training lesions only.**

| Features | Selected concept | Recorded negatives | Observed slope [range] | Predicted slope | Observed over predicted [95% CI] | τ̂(1, −1) [95% CI] | Mean contrast crosses zero | Replicates changing sign | Calibrated τ̂(1, −1) |
| --- | --- | ---: | --- | ---: | --- | --- | --- | ---: | ---: |
| tabular | Color variegation | 320 | −2.06 [−2.39, −1.43] | −2.09 | 0.99 [0.94, 1.04] | −4.10 [−4.35, −3.83] | yes | 20 of 20 | −2.86 |
| tabular | Color variegation | 1,000 | −1.94 [−2.25, −1.62] | −2.09 | 0.93 [0.89, 0.97] | −3.95 [−4.17, −3.72] | yes | 20 of 20 | −2.50 |
| tabular | Color variegation | 3,000 | −1.79 [−2.14, −1.41] | −2.09 | 0.86 [0.81, 0.90] | −3.69 [−3.90, −3.47] | yes | 19 of 20 | −2.27 |
| tabular | Size | 320 | −1.38 [−1.80, −0.94] | −1.38 | 1.00 [0.92, 1.08] | −2.59 [−2.78, −2.40] | yes | 20 of 20 | −1.40 |
| tabular | Lesion-skin contrast | 320 | −2.34 [−2.80, −1.98] | −2.05 | 1.14 [1.09, 1.20] | −4.72 [−4.99, −4.47] | yes | 20 of 20 | −3.33 |
| image | Color variegation | 320 | −1.13 [−1.33, −0.93] | −1.03 | 1.10 [1.05, 1.15] | −2.26 [−2.37, −2.14] | yes | 20 of 20 | −1.81 |
| image | Color variegation | 1,000 | −0.80 [−1.15, −0.54] | −1.03 | 0.78 [0.71, 0.85] | −1.60 [−1.78, −1.44] | yes | 18 of 20 | −1.26 |
| image | Color variegation | 3,000 | −0.93 [−1.38, −0.38] | −1.03 | 0.91 [0.78, 1.03] | −1.79 [−2.05, −1.51] | no | 8 of 20 | −1.09 |
| image | Size | 320 | −0.63 [−0.89, −0.41] | −0.71 | 0.88 [0.79, 0.98] | −1.11 [−1.25, −0.99] | yes | 20 of 20 | −0.90 |
| image | Lesion-skin contrast | 320 | −1.61 [−1.86, −1.38] | −1.46 | 1.10 [1.06, 1.15] | −3.04 [−3.17, −2.91] | yes | 20 of 20 | −2.21 |

In Table S23, entries are observed / predicted slopes of raw logit contrasts with 320 selected recorded negatives and the validation set shared across doses within each replicate.

**TABLE S23. Dose slopes of all five concepts under the training-only dose.**

| Features | Selected concept | Color variegation | Size | Lesion-skin contrast | Asymmetry | Border irregularity |
| --- | --- | --- | --- | --- | --- | --- |
| tabular | Color variegation | −2.06 / −2.09 | −1.38 / −1.45 | −1.44 / −1.40 | +0.75 / +0.82 | +0.60 / +0.65 |
| tabular | Size | −1.18 / −1.06 | −1.38 / −1.38 | −0.56 / −0.61 | +0.24 / +0.31 | −0.11 / −0.05 |
| tabular | Lesion-skin contrast | −1.50 / −1.40 | −0.63 / −0.84 | −2.34 / −2.05 | +1.26 / +1.19 | +1.29 / +1.18 |
| image | Color variegation | −1.13 / −1.03 | −0.63 / −0.67 | −1.32 / −1.08 | +0.98 / +0.64 | +0.98 / +0.59 |
| image | Size | −0.44 / −0.54 | −0.63 / −0.71 | −0.25 / −0.42 | +0.11 / +0.27 | −0.07 / +0.10 |
| image | Lesion-skin contrast | −1.25 / −1.11 | −0.67 / −0.65 | −1.61 / −1.46 | +1.24 / +0.93 | +1.27 / +0.93 |

**Joint training-and-validation dose.** Script `vr32_dose_nuisance.py` ran a second design in which validation negatives were drawn under the same rule as training negatives, without common random numbers. Each learner was Platt-calibrated on its own validation lesions. The dose then acts on training, early stopping and calibration together. This design estimates the effect of a joint selection rule for development data, not the training-selection effect, and its agreement in magnitude with Proposition 1 is only diagnostic. Its calibrated contrast crossed zero in the same six combinations, and script `vr44_dose_estimand.py` gives its paired effects. Table S24 reports it, Table S25 gives the paired effects and Table S26 the slopes of all five concepts. Script `vr26_dose_poisson.py` ran the same joint design with 10 replicates and the ridge prediction only, with the same pattern.

In Table S24, contrasts are calibrated, there are 20 replicates, and ranges are 2.5 and 97.5 percentiles over replicates.

**TABLE S24. Joint training-and-validation dose.**

| Features | Selected concept | Recorded negatives | Observed slope [range] | Prediction | Predicted slope [range] | Observed over predicted [95% CI] |
| --- | --- | ---: | --- | --- | --- | --- |
| tabular | Color variegation | 320 | −2.58 [−3.37, −2.01] | Bayes-optimal shift, concept in input | −2.09 [−2.09, −2.09] | 1.24 [1.16, 1.32] |
| tabular | Size | 320 | −1.80 [−2.26, −1.31] | Bayes-optimal shift, concept in input | −1.38 [−1.38, −1.38] | 1.31 [1.21, 1.40] |
| tabular | Lesion-skin contrast | 320 | −2.65 [−3.45, −2.23] | Bayes-optimal shift, concept in input | −2.05 [−2.05, −2.05] | 1.30 [1.23, 1.37] |
| tabular | Color variegation | 1,000 | −2.38 [−2.80, −1.97] | Bayes-optimal shift, concept in input | −2.09 [−2.09, −2.09] | 1.14 [1.09, 1.19] |
| tabular | Color variegation | 3,000 | −2.10 [−2.63, −1.63] | Bayes-optimal shift, concept in input | −2.09 [−2.09, −2.09] | 1.01 [0.95, 1.06] |
| image | Color variegation | 320 | −1.01 [−1.22, −0.85] | ridge conditional mean | −1.03 [−1.04, −1.02] | 0.99 [0.95, 1.03] |
| image | Color variegation | 320 | −1.01 [−1.22, −0.85] | perceptron conditional mean | −1.02 [−1.09, −0.93] | 0.99 [0.94, 1.05] |
| image | Size | 320 | −0.67 [−0.79, −0.54] | ridge conditional mean | −0.71 [−0.72, −0.70] | 0.94 [0.89, 0.98] |
| image | Size | 320 | −0.67 [−0.79, −0.54] | perceptron conditional mean | −0.68 [−0.72, −0.62] | 0.97 [0.92, 1.02] |
| image | Lesion-skin contrast | 320 | −1.41 [−1.63, −1.10] | ridge conditional mean | −1.46 [−1.48, −1.44] | 0.97 [0.91, 1.02] |
| image | Lesion-skin contrast | 320 | −1.41 [−1.63, −1.10] | perceptron conditional mean | −1.51 [−1.57, −1.45] | 0.93 [0.88, 0.98] |
| image | Color variegation | 1,000 | −0.71 [−1.10, −0.52] | ridge conditional mean | −1.03 [−1.04, −1.01] | 0.70 [0.64, 0.78] |
| image | Color variegation | 1,000 | −0.71 [−1.10, −0.52] | perceptron conditional mean | −1.01 [−1.05, −0.96] | 0.70 [0.64, 0.78] |
| image | Color variegation | 3,000 | −0.73 [−1.06, −0.35] | ridge conditional mean | −1.03 [−1.04, −1.01] | 0.72 [0.63, 0.80] |
| image | Color variegation | 3,000 | −0.73 [−1.06, −0.35] | perceptron conditional mean | −1.04 [−1.10, −1.00] | 0.70 [0.61, 0.79] |

In Table S25, the outcome is the calibrated contrast of the selected concept with 320 selected recorded negatives. Intervals are bootstrap intervals over the 20 replicates, conditional on the fixed cohort.

**TABLE S25. Effect of the joint training-and-validation dose.**

| Features | Concept | Mean Δ at η = −1 / η = 1 | τ̂(1, −1) [95% CI] | Predicted | Observed / predicted | Replicates changing sign |
| --- | --- | --- | --- | ---: | ---: | ---: |
| tabular | Color variegation | +3.79 / −1.28 | −5.08 [−5.53, −4.64] | −4.17 | 1.22 | 20 of 20 |
| tabular | Size | +2.90 / −0.63 | −3.53 [−3.89, −3.18] | −2.76 | 1.28 | 20 of 20 |
| tabular | Lesion-skin contrast | +3.36 / −1.93 | −5.29 [−5.73, −4.94] | −4.10 | 1.29 | 20 of 20 |
| image | Color variegation | +1.30 / −0.65 | −1.95 [−2.05, −1.84] | −2.05 | 0.95 | 20 of 20 |
| image | Size | +1.11 / −0.11 | −1.22 [−1.30, −1.15] | −1.41 | 0.86 | 18 of 20 |
| image | Lesion-skin contrast | +1.49 / −1.15 | −2.65 [−2.80, −2.49] | −2.91 | 0.91 | 20 of 20 |

In Table S26, entries are observed / predicted slopes with 320 selected recorded negatives.

**TABLE S26. Dose slopes of all five concepts under the joint dose.**

| Features | Selected concept | Prediction | Color variegation | Size | Lesion-skin contrast | Asymmetry | Border irregularity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tabular | Color variegation | Bayes-optimal shift | −2.58 / −2.09 | −1.67 / −1.45 | −1.70 / −1.40 | +0.88 / +0.82 | +0.68 / +0.65 |
| tabular | Size | Bayes-optimal shift | −1.50 / −1.06 | −1.80 / −1.38 | −0.81 / −0.61 | +0.46 / +0.31 | +0.02 / −0.05 |
| tabular | Lesion-skin contrast | Bayes-optimal shift | −1.96 / −1.40 | −0.96 / −0.84 | −2.65 / −2.05 | +1.56 / +1.19 | +1.55 / +1.18 |
| image | Color variegation | ridge | −1.01 / −1.03 | −0.59 / −0.67 | −1.14 / −1.08 | +0.81 / +0.64 | +0.79 / +0.59 |
| image | Color variegation | mlp | −1.01 / −1.02 | −0.59 / −0.67 | −1.14 / −1.08 | +0.81 / +0.65 | +0.79 / +0.60 |
| image | Size | ridge | −0.42 / −0.54 | −0.67 / −0.71 | −0.18 / −0.42 | +0.02 / +0.27 | −0.19 / +0.11 |
| image | Size | mlp | −0.42 / −0.54 | −0.67 / −0.68 | −0.18 / −0.45 | +0.02 / +0.29 | −0.19 / +0.12 |
| image | Lesion-skin contrast | ridge | −1.17 / −1.11 | −0.68 / −0.65 | −1.41 / −1.46 | +1.07 / +0.93 | +1.06 / +0.93 |
| image | Lesion-skin contrast | mlp | −1.17 / −1.15 | −0.68 / −0.68 | −1.41 / −1.51 | +1.07 / +0.97 | +1.06 / +0.97 |

**Sampling without replacement.** Script `vr21_dose_calibrated.py` ran the same experiment with exactly 320 recorded negatives drawn without replacement by the Gumbel top-*k* construction, for η up to 3. Its inclusion probabilities are not proportional to *w* (Section S1). Table S27 compares the nominal and actual expected number of upper-tertile lesions, checked by 300 Monte Carlo draws, and the predicted change in the contrast. For size, a few lesions with extreme area would have nominal probabilities above one, so the composition of the sample differs from the nominal one at η = 0.5. The predicted contrast changes by less than one percent in every case, because it averages log π over the test population. Table S28 gives the results of this design, which also drew validation lesions by the same rule and agrees with Table S24. Table S29 gives the effective sample size of its selection weights, which fell steeply for η > 1.

**TABLE S27. Inclusion probabilities under sampling without replacement, recorded-negative training pool of 241,263 lesions, 320 draws.**

| Concept | η | Expected upper-tertile draws, nominal / actual / Monte Carlo | Predicted contrast change, nominal / actual |
| --- | ---: | --- | --- |
| Color variegation | −1.0 | 31.8 / 31.9 / 31.9 | +2.086 / +2.085 |
| Color variegation | +0.5 | 180.3 / 180.2 / 180.4 | −1.043 / −1.042 |
| Color variegation | +1.0 | 253.2 / 253.0 / 252.7 | −2.086 / −2.085 |
| Size | −1.0 | 58.3 / 58.3 / 58.0 | +1.384 / +1.384 |
| Size | +0.5 | 318.1 / 226.8 / 225.8 | −0.692 / −0.690 |
| Size | +1.0 | 320.0 / 317.2 / 317.0 | −1.384 / −1.374 |
| Lesion-skin contrast | −1.0 | 33.0 / 33.1 / 33.6 | +2.049 / +2.048 |
| Lesion-skin contrast | +0.5 | 186.4 / 186.3 / 186.7 | −1.024 / −1.024 |
| Lesion-skin contrast | +1.0 | 268.9 / 267.7 / 268.2 | −2.049 / −2.047 |

In Table S28, ranges are 2.5 and 97.5 percentiles over replicates.

**TABLE S28. Selection dose under sampling without replacement.**

| Features | Selected concept | Calibrated slope, η in [−1, 1] [replicate range] | Raw slope | Predicted from input | Predicted if z_k observed | Crossing, observed / predicted | Concepts moving as predicted |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| tabular | Color variegation | −2.58 [−3.19, −2.28] | −2.83 | −2.09 | −2.09 | 0.47 / 0.63 | 5 of 5 |
| tabular | Size | −1.90 [−2.19, −1.63] | −1.95 | −1.38 | −1.38 | 0.59 / 0.73 | 4 of 5 |
| tabular | Lesion-skin contrast | −2.51 [−3.03, −1.95] | −2.78 | −2.05 | −2.05 | 0.07 / 0.06 | 5 of 5 |
| image | Color variegation | −1.03 [−1.22, −0.90] | −1.15 | −1.03 | −2.09 | 0.43 / 0.52 | 5 of 5 |
| image | Size | −0.67 [−0.75, −0.57] | −0.74 | −0.71 | −1.38 | 0.51 / 0.87 | 3 of 5 |
| image | Lesion-skin contrast | −1.46 [−1.75, −1.26] | −1.74 | −1.46 | −2.05 | 0.07 / 0.08 | 5 of 5 |

In Table S29, the effective sample size is Kish's (Kish, 1965), as a share of the 241,263 recorded-negative training lesions. Shares are the selection weight falling in the lower and upper tertiles.

**TABLE S29. Support of the selection dose for color variegation.**

| η | Effective sample size, share of pool | Share of selection weight in lower tertile | Share in upper tertile |
| ---: | ---: | ---: | ---: |
| −1.0 | 0.605 | 0.621 | 0.100 |
| −0.5 | 0.845 | 0.476 | 0.192 |
| 0.0 | 1.000 | 0.319 | 0.345 |
| 0.5 | 0.681 | 0.170 | 0.563 |
| 1.0 | 0.210 | 0.063 | 0.791 |
| 1.5 | 0.073 | 0.016 | 0.932 |
| 2.0 | 0.040 | 0.003 | 0.983 |
| 3.0 | 0.023 | 0.000 | 0.999 |

**Proposition 1 at the lesion level.** Script `vr22_pointwise_bridge.py` ran 200 joint bootstrap replicates for each frozen-feature family. Each replicate resampled training patients and trained M0, M2 and a propensity model ĝ with the same perceptron and input. Each model was Platt-calibrated on validation lesions from its own population: all validation lesions for M0, verified ones for M2, and recorded negatives for ĝ, with *S* as the label. After resampling test patients, the lesion-level calibrated gap logit *f*₀ − logit *f*₂ was regressed on log ĝ, for which Proposition 1 predicts slope one. The concept-level observed gap was compared with the stratum contrast of log ĝ. Table S30 reports medians and 95 percent percentile intervals over replicates.

**TABLE S30. Observed and predicted learner gap Δ_M0 − Δ_M2, calibrated learners, 200 joint resamples.**

| Features | Concept | Observed gap [95% CI] | Predicted gap [95% CI] | Same sign in resamples |
| --- | --- | --- | --- | ---: |
| tabular | Color variegation | +1.83 [1.01, 2.53] | +1.43 [0.92, 1.97] | 1.000 |
| tabular | Size | +0.90 [0.33, 1.46] | +0.73 [0.23, 1.16] | 0.995 |
| tabular | Lesion-skin contrast | +1.84 [0.92, 2.72] | +1.55 [1.03, 2.08] | 1.000 |
| tabular | Asymmetry | −1.56 [−2.54, −0.75] | −0.96 [−1.70, −0.42] | 0.995 |
| tabular | Border irregularity | −1.64 [−2.66, −0.77] | −1.03 [−1.79, −0.41] | 1.000 |
| tabular | lesion-level slope, median R² | 0.89 [0.67, 1.09] | R² 0.53 | |
| image | Color variegation | +0.92 [0.57, 1.31] | +1.05 [0.73, 1.29] | 1.000 |
| image | Size | +0.72 [0.48, 0.97] | +0.72 [0.53, 0.89] | 1.000 |
| image | Lesion-skin contrast | +0.89 [0.43, 1.40] | +1.14 [0.76, 1.40] | 1.000 |
| image | Asymmetry | −0.71 [−1.19, −0.34] | −0.92 [−1.16, −0.63] | 1.000 |
| image | Border irregularity | −0.60 [−1.12, −0.18] | −0.85 [−1.09, −0.57] | 0.995 |
| image | lesion-level slope, median R² | 0.86 [0.66, 1.08] | R² 0.71 | |

**Split-sample errors-in-variables diagnostic.** The regressor log ĝ is estimated, so a least-squares slope is attenuated toward zero. Script `vr27_bridge_eiv.py` ran 100 joint replicates per family in which two propensity models, ĝ_A and ĝ_B, were fitted on disjoint halves of the resampled training patients. The slope of the calibrated gap on log ĝ_A was instrumented by log ĝ_B, averaged over both orders, which is consistent when the estimation errors of the two halves are independent. The correlation of log ĝ_A and log ĝ_B estimates the reliability of a single estimate. Table S31 reports the results. The instrumented point estimates exceed one. The lesion-level regression is an association diagnostic rather than a test that the slope equals one.

In Table S31, the slope is that of the learner gap on log ĝ, with 100 joint replicates per family.

**TABLE S31. Split-sample errors-in-variables diagnostic for the lesion-level slope.**

| Features | Least-squares slope on log ĝ_A [95% CI] | Instrumented slope [95% CI] | Reliability of log ĝ [95% CI] |
| --- | --- | --- | --- |
| tabular | 0.98 [0.72, 1.20] | 1.33 [1.01, 1.69] | 0.74 [0.61, 0.83] |
| image | 0.98 [0.72, 1.28] | 1.32 [0.92, 1.75] | 0.76 [0.57, 0.85] |

Two propensity models trained on disjoint patients can still share systematic error from the same model class. The instrumented slope is therefore a sensitivity diagnostic, not a correction that recovers the true slope. Script `vr33_bridge_alt.py` repeated the lesion-level analysis in 100 joint replicates per family, with a second model class for ĝ and on the supported test population. The second class is gradient boosting of *S* among recorded negatives. Table S32 reports the results. Section S2 gives the slope that the same trained learners attain in a semi-synthetic design where the population identity holds exactly.

In Table S32, the gap is calibrated, slopes are shown on the full and supported test populations, and there are 100 joint replicates per family.

**TABLE S32. Lesion-level slope of the learner gap for two model classes of ĝ.**

| Features | Model for ĝ | Test population | Slope [95% CI] | Median R² | Concepts with sign agreement ≥ 97.5% | Concept calibration slope [95% CI] |
| --- | --- | --- | --- | ---: | ---: | --- |
| tabular | perceptron | full | 0.87 [0.66, 1.13] | 0.50 | 5 of 5 | 1.43 [0.89, 2.26] |
| tabular | perceptron | supported, 1st percentile | 0.85 [0.66, 1.12] | 0.46 | 4 of 5 | 1.46 [0.69, 2.68] |
| tabular | gradient boosting | full | 1.29 [0.91, 2.20] | 0.30 | 5 of 5 | 1.97 [1.14, 4.41] |
| tabular | gradient boosting | supported, 1st percentile | 1.17 [0.86, 2.09] | 0.23 | 5 of 5 | 1.90 [0.77, 4.24] |
| image | perceptron | full | 0.86 [0.66, 1.03] | 0.69 | 5 of 5 | 0.82 [0.50, 1.29] |
| image | perceptron | supported, 1st percentile | 0.83 [0.64, 1.01] | 0.65 | 5 of 5 | 0.70 [0.30, 1.34] |
| image | gradient boosting | full | 0.85 [0.62, 2.52] | 0.23 | 5 of 5 | 1.81 [0.86, 8.36] |
| image | gradient boosting | supported, 1st percentile | 0.76 [0.54, 1.98] | 0.20 | 5 of 5 | 1.50 [0.53, 6.45] |



**Verification pattern.** Among recorded negatives, verification was regressed on each concept split into its deviation from the patient's mean and the patient's mean, with sex, anatomical site, acquisition site and skin tone. Standard errors are clustered by patient. Table S33 reports the coefficients per standard deviation, to three decimals. The between-patient coefficients of asymmetry and border irregularity nearly coincide because the two measurements are highly correlated (Spearman 0.94).

**TABLE S33. Within-patient and between-patient dependence of verification on appearance among recorded negatives.**

| Concept | Within patient (SE) | Between patients (SE) |
| --- | --- | --- |
| Color variegation | +0.933 (0.037) | +0.101 (0.146) |
| Size | +0.312 (0.014) | −0.358 (0.229) |
| Lesion-skin contrast | +0.958 (0.033) | −0.102 (0.107) |
| Asymmetry | −0.518 (0.073) | −0.699 (0.152) |
| Border irregularity | −0.329 (0.072) | −0.700 (0.148) |

**Acquisition sites.** Table S34 reports the adjusted verified-benign contrast log *B*_V per acquisition site, for sites with at least ten verified benign lesions in the two outer tertiles. Site 3 had 15 verified benign lesions, all in the upper tertile, so its contrast is not estimable. Site 7 had 7 verified benign lesions and falls below the threshold. Sites are numbered alphabetically by name in Tables S34 and S36. Values are given to three decimals. Color variegation and lesion-skin contrast give similar values at Sites 4 and 5 because the two measurements are correlated (Spearman 0.65) and not because a value was reused. The fits use different lesions, for example 230 and 221 verified benign lesions at Site 5. Table S35 pools the estimable sites with a DerSimonian-Laird random-effects model (DerSimonian & Laird, 1986). Table S36 retrains M0 and M2, three seeds each, after removing one site's patients from training, validation and test.

**TABLE S34. log *B*_V by acquisition site, with standard errors.**

| Site | Verified benign lesions | Color variegation | Size | Lesion-skin contrast |
| --- | ---: | --- | --- | --- |
| Site 1 | 28 | +3.317 (1.026) | +1.445 (0.492) | +3.077 (1.052) |
| Site 2 | 82 | +2.862 (0.458) | +2.457 (0.458) | +2.971 (0.590) |
| Site 3 | 15 | separated | separated | separated |
| Site 4 | 114 | +1.496 (0.231) | +1.228 (0.208) | +1.501 (0.233) |
| Site 5 | 230 | +1.206 (0.167) | +1.187 (0.166) | +1.214 (0.171) |
| Site 6 | 120 | +1.308 (0.275) | +0.972 (0.217) | +1.791 (0.361) |
| Site 7 | 7 in all tertiles | fewer than ten verified benign | fewer than ten verified benign | fewer than ten verified benign |

**TABLE S35. Heterogeneity of log *B*_V across acquisition sites.**

| Concept | Estimable sites | Q (df) | I² | τ² | Pooled log *B*_V [95% CI] | Site range |
| --- | ---: | --- | ---: | ---: | --- | --- |
| Color variegation | 5 | 15.2 (4) | 0.74 | 0.232 | 1.70 [1.17, 2.23] | [1.21, 3.32] |
| Size | 5 | 8.8 (4) | 0.55 | 0.076 | 1.30 [0.96, 1.65] | [0.97, 2.46] |
| Lesion-skin contrast | 5 | 11.9 (4) | 0.66 | 0.199 | 1.75 [1.23, 2.28] | [1.21, 3.08] |

**TABLE S36. Learned contrasts with one acquisition site left out, mean over seeds 0 to 2.**

| Features | Site left out | Color Δ_M0 / Δ_M2 | Size Δ_M0 / Δ_M2 | Contrast Δ_M0 / Δ_M2 |
| --- | --- | --- | --- | --- |
| tabular | Site 1 | +0.85 / −1.05 | +0.52 / −0.49 | +0.18 / −1.70 |
| tabular | Site 2 | +0.59 / −1.06 | +0.39 / −0.46 | +0.09 / −1.69 |
| tabular | Site 3 | +1.02 / −1.01 | +0.58 / −0.48 | +0.50 / −1.55 |
| tabular | Site 4 | +1.19 / −0.96 | +0.75 / −0.32 | +0.72 / −1.59 |
| tabular | Site 5 | +0.79 / −0.83 | +0.41 / −0.37 | +0.74 / −1.22 |
| tabular | Site 6 | +1.03 / −1.08 | +0.58 / −0.58 | +0.58 / −1.56 |
| tabular | Site 7 | +0.81 / −0.95 | +0.48 / −0.42 | +0.35 / −1.51 |
| image | Site 1 | +1.15 / −0.68 | +1.18 / −0.36 | +0.38 / −1.08 |
| image | Site 2 | +0.70 / −0.80 | +0.98 / −0.41 | −0.08 / −1.26 |
| image | Site 3 | +0.92 / −0.69 | +1.00 / −0.34 | +0.21 / −1.09 |
| image | Site 4 | +1.50 / −0.62 | +1.42 / −0.37 | +0.75 / −0.92 |
| image | Site 5 | +2.19 / −0.59 | +1.74 / −0.28 | +1.84 / −0.90 |
| image | Site 6 | +0.62 / −0.56 | +0.81 / −0.29 | −0.08 / −0.93 |
| image | Site 7 | +1.31 / −0.75 | +1.32 / −0.39 | +0.59 / −1.17 |

**Overlap.** The common verification propensity σ̂ is a gradient-boosting model on the tabular features, used as one clinical support definition for both families and estimated without using the test fold. Within each joint bootstrap replicate, a threshold was set at the 1st, 5th or 10th percentile of σ̂ among verified training lesions. The test population was then restricted to lesions with σ̂ at or above it. Table S37 reports the contrasts. Table S38 reports the full, support-restricted and flagged test populations for the original fits.

In Table S37, intervals come from the one-seed joint bootstrap with 5,000 replicates per family.

**TABLE S37. Learned contrasts within the support of verified training lesions.**

| Features | Concept | Threshold percentile | Δ_M0 [95% CI] | Δ_M2 [95% CI] | Robust at 95% |
| --- | --- | ---: | --- | --- | --- |
| tabular | Color variegation | 1 | +0.35 [−0.64, 1.07] | −1.04 [−1.48, −0.68] | no |
| tabular | Color variegation | 5 | −0.05 [−1.13, 0.71] | −1.27 [−1.73, −0.82] | no |
| tabular | Color variegation | 10 | −0.13 [−1.22, 0.67] | −1.31 [−1.79, −0.81] | no |
| tabular | Size | 1 | +0.35 [−0.43, 0.88] | −0.47 [−0.73, −0.14] | no |
| tabular | Size | 5 | +0.34 [−0.52, 0.88] | −0.55 [−0.88, −0.15] | no |
| tabular | Size | 10 | +0.29 [−0.60, 0.91] | −0.58 [−0.96, −0.13] | no |
| tabular | Lesion-skin contrast | 1 | −0.11 [−1.27, 0.72] | −1.61 [−2.14, −1.15] | no |
| tabular | Lesion-skin contrast | 5 | −0.57 [−1.84, 0.33] | −1.84 [−2.40, −1.31] | no |
| tabular | Lesion-skin contrast | 10 | −0.61 [−1.90, 0.32] | −1.88 [−2.45, −1.33] | no |
| image | Color variegation | 1 | +0.78 [−0.43, 1.52] | −0.70 [−1.27, −0.45] | no |
| image | Color variegation | 5 | +0.13 [−1.08, 1.02] | −0.81 [−1.46, −0.52] | no |
| image | Color variegation | 10 | +0.34 [−0.91, 1.25] | −0.84 [−1.55, −0.53] | no |
| image | Size | 1 | +1.14 [0.28, 1.69] | −0.31 [−0.70, −0.13] | yes |
| image | Size | 5 | +1.09 [0.21, 1.69] | −0.29 [−0.73, −0.06] | yes |
| image | Size | 10 | +1.34 [0.35, 2.00] | −0.31 [−0.82, −0.04] | yes |
| image | Lesion-skin contrast | 1 | +0.08 [−1.33, 0.99] | −1.14 [−1.89, −0.82] | no |
| image | Lesion-skin contrast | 5 | −0.77 [−2.12, 0.22] | −1.27 [−2.10, −0.90] | no |
| image | Lesion-skin contrast | 10 | −0.66 [−2.02, 0.33] | −1.31 [−2.19, −0.92] | no |

In Table S38, intervals are patient bootstrap intervals conditional on the fitted models.

**TABLE S38. Learned contrasts on the full, support-restricted and flagged test populations.**

| Features | Population (share of test) | Concept | Δ_M0 [95% CI] | Δ_M2 [95% CI] | P(signs differ) |
| --- | --- | --- | --- | --- | ---: |
| tabular | all test lesions (100 percent) | Color variegation | +1.02 [0.77, 1.28] | −0.95 [−1.08, −0.83] | 1.000 |
| tabular | all test lesions (100 percent) | Size | +0.65 [0.47, 0.87] | −0.46 [−0.56, −0.37] | 1.000 |
| tabular | all test lesions (100 percent) | Lesion-skin contrast | +0.52 [0.19, 0.85] | −1.50 [−1.65, −1.34] | 1.000 |
| tabular | within support (68 percent) | Color variegation | +0.35 [0.06, 0.66] | −1.04 [−1.18, −0.90] | 0.991 |
| tabular | within support (68 percent) | Size | +0.35 [0.16, 0.56] | −0.47 [−0.59, −0.34] | 1.000 |
| tabular | within support (68 percent) | Lesion-skin contrast | −0.11 [−0.48, 0.28] | −1.61 [−1.78, −1.44] | 0.273 |
| tabular | flagged lesions (6 percent) | Color variegation | −0.22 [−0.55, 0.12] | −0.97 [−1.15, −0.74] | 0.105 |
| tabular | flagged lesions (6 percent) | Size | +0.59 [0.24, 0.93] | −0.36 [−0.61, −0.09] | 0.998 |
| tabular | flagged lesions (6 percent) | Lesion-skin contrast | −1.64 [−2.06, −1.26] | −1.75 [−1.92, −1.56] | 0.000 |
| image | all test lesions (100 percent) | Color variegation | +1.64 [1.38, 1.86] | −0.66 [−0.76, −0.56] | 1.000 |
| image | all test lesions (100 percent) | Size | +1.56 [1.31, 1.87] | −0.32 [−0.40, −0.26] | 1.000 |
| image | all test lesions (100 percent) | Lesion-skin contrast | +1.01 [0.63, 1.36] | −1.07 [−1.21, −0.96] | 1.000 |
| image | within support (68 percent) | Color variegation | +0.78 [0.45, 1.09] | −0.70 [−0.81, −0.59] | 1.000 |
| image | within support (68 percent) | Size | +1.14 [0.87, 1.43] | −0.31 [−0.39, −0.23] | 1.000 |
| image | within support (68 percent) | Lesion-skin contrast | +0.08 [−0.31, 0.46] | −1.14 [−1.28, −1.01] | 0.665 |
| image | flagged lesions (6 percent) | Color variegation | +1.32 [0.92, 1.69] | −0.77 [−0.94, −0.57] | 1.000 |
| image | flagged lesions (6 percent) | Size | +1.95 [1.58, 2.33] | −0.39 [−0.61, −0.15] | 1.000 |
| image | flagged lesions (6 percent) | Lesion-skin contrast | +0.03 [−0.55, 0.56] | −1.23 [−1.41, −1.04] | 0.503 |

**Other estimates of σ.** Support membership depends on the estimate of σ. Proposition 2 defines σ on each learner's input, so the perceptron trained on that input is the theory-matched definition. The tabular gradient-boosting estimate is a common clinical definition for both families and the only one with joint intervals. Script `vr37_overlap_sigma.py` repeated the overlap analysis with two further estimates: perceptrons trained on the tabular or on the image features, Platt-calibrated on validation data and averaged over seeds 0 to 2. It also reran the common estimate under the same procedure, with the original fits of M0 and M2 averaged over seeds 0 to 2 and 2,000 bootstrap resamples of test patients. These intervals condition on the fitted learners and are narrower than the joint intervals of Table S37. Table S39 reports the share of test lesions retained and the share that changed membership relative to the common estimate.

**TABLE S39. Overlap analysis under three estimates of σ, test-patient bootstrap conditional on the fitted learners.**

| Features | Estimate of σ | Threshold | Test retained | Changed membership | Color Δ_M0 [95% CI] | Color Δ_M2 [95% CI] | Robust at 95%: color / size / contrast |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| tabular | gradient boosting, tabular (common) | 1st | 0.68 | 0.00 | +0.35 [0.05, 0.67] | −1.04 [−1.18, −0.91] | yes / yes / no |
| tabular | gradient boosting, tabular (common) | 5th | 0.36 | 0.00 | −0.05 [−0.41, 0.35] | −1.27 [−1.44, −1.08] | no / yes / no |
| tabular | gradient boosting, tabular (common) | 10th | 0.23 | 0.00 | −0.13 [−0.52, 0.32] | −1.31 [−1.51, −1.08] | no / no / no |
| tabular | perceptron, tabular | 1st | 0.62 | 0.23 | +0.23 [0.06, 0.42] | −1.07 [−1.20, −0.94] | yes / yes / no |
| tabular | perceptron, tabular | 5th | 0.28 | 0.19 | −0.40 [−0.60, −0.18] | −1.30 [−1.46, −1.13] | no / yes / no |
| tabular | perceptron, tabular | 10th | 0.19 | 0.13 | −0.68 [−0.92, −0.44] | −1.38 [−1.57, −1.19] | no / no / no |
| tabular | perceptron, image | 1st | 0.73 | 0.28 | +0.85 [0.60, 1.12] | −1.02 [−1.15, −0.90] | yes / yes / no |
| tabular | perceptron, image | 5th | 0.42 | 0.31 | +0.88 [0.59, 1.18] | −1.13 [−1.27, −0.99] | yes / yes / no |
| tabular | perceptron, image | 10th | 0.22 | 0.23 | +1.10 [0.73, 1.48] | −1.21 [−1.37, −1.04] | yes / yes / yes |
| image | gradient boosting, tabular (common) | 1st | 0.68 | 0.00 | +0.78 [0.45, 1.07] | −0.70 [−0.80, −0.59] | yes / yes / no |
| image | gradient boosting, tabular (common) | 5th | 0.36 | 0.00 | +0.13 [−0.36, 0.52] | −0.81 [−0.94, −0.67] | no / yes / no |
| image | gradient boosting, tabular (common) | 10th | 0.23 | 0.00 | +0.34 [−0.20, 0.78] | −0.84 [−1.00, −0.68] | no / yes / no |
| image | perceptron, tabular | 1st | 0.62 | 0.23 | +1.23 [0.97, 1.49] | −0.74 [−0.85, −0.64] | yes / yes / yes |
| image | perceptron, tabular | 5th | 0.28 | 0.19 | +0.84 [0.52, 1.15] | −0.90 [−1.04, −0.76] | yes / yes / no |
| image | perceptron, tabular | 10th | 0.19 | 0.13 | +0.70 [0.32, 1.07] | −0.95 [−1.12, −0.78] | yes / yes / no |
| image | perceptron, image | 1st | 0.73 | 0.28 | +0.68 [0.46, 0.85] | −0.69 [−0.80, −0.59] | yes / yes / no |
| image | perceptron, image | 5th | 0.42 | 0.31 | −0.29 [−0.54, −0.06] | −0.76 [−0.88, −0.64] | no / yes / no |
| image | perceptron, image | 10th | 0.22 | 0.23 | −0.82 [−1.10, −0.56] | −0.83 [−0.98, −0.69] | no / yes / no |

**Composition of the supported population.** Restricting the test population to σ̂ ≥ *c* changes the joint distribution of the concepts within each tertile, so Δ can change even when the prediction function does not. Within each tertile, script `vr40_support_composition.py` reweights the supported lesions by the inverse of their estimated probability of being supported given the five standardized concepts. That probability comes from a logistic model fitted on the lesions of the tertile in the full test population, and the weights are truncated at their 99th percentile. The reweighted contrast compares supported lesions with the concept composition of the full population. Intervals come from 300 test-patient bootstrap resamples, refitting the weighting model, conditional on the fitted learners. Table S40 reports color variegation. The reweighting did not restore the full-population M0 contrast in any setting, and M2 stayed negative. Standardizing the supported lesions on the five measured concepts thus left the attenuation within support in place. This weighting model does not balance unmeasured appearance, patient or site variables. The analysis changes the evaluation population and does not isolate extrapolation by M2.

In Table S40, reweighting targets the full-population concept composition within each tertile, and original fits are averaged over seeds 0 to 2.

**TABLE S40. Supported contrasts of color variegation, raw and reweighted.**

| Features | Estimate of σ | Threshold | Share of test | M0: raw / reweighted [95% CI] | M2: raw / reweighted [95% CI] |
| --- | --- | ---: | ---: | --- | --- |
| tabular | full population | none | 1.00 | +1.02 [0.75, 1.27] | −0.95 [−1.07, −0.84] |
| tabular | perceptron, tabular | 1st | 0.62 | +0.23 / +0.23 [0.04, 0.40] | −1.07 / −1.02 [−1.15, −0.91] |
| tabular | perceptron, tabular | 5th | 0.28 | −0.40 / −0.40 [−0.60, −0.20] | −1.30 / −1.18 [−1.33, −1.05] |
| tabular | perceptron, tabular | 10th | 0.19 | −0.68 / −0.44 [−0.79, −0.10] | −1.38 / −1.25 [−1.45, −1.07] |
| tabular | perceptron, image | 1st | 0.73 | +0.85 / +0.78 [0.53, 1.03] | −1.02 / −0.98 [−1.10, −0.87] |
| tabular | perceptron, image | 5th | 0.42 | +0.88 / +0.63 [0.36, 0.87] | −1.13 / −1.04 [−1.16, −0.92] |
| tabular | perceptron, image | 10th | 0.22 | +1.10 / +0.58 [0.29, 0.85] | −1.21 / −1.05 [−1.19, −0.90] |
| tabular | gradient boosting, tabular | 1st | 0.68 | +0.35 / +0.26 [−0.01, 0.54] | −1.04 / −1.02 [−1.15, −0.90] |
| tabular | gradient boosting, tabular | 5th | 0.36 | −0.05 / −0.12 [−0.39, 0.16] | −1.27 / −1.10 [−1.23, −0.95] |
| tabular | gradient boosting, tabular | 10th | 0.23 | −0.13 / +0.22 [−0.17, 0.78] | −1.31 / −1.11 [−1.27, −0.90] |
| image | full population | none | 1.00 | +1.64 [1.41, 1.87] | −0.66 [−0.76, −0.55] |
| image | perceptron, tabular | 1st | 0.62 | +1.23 / +1.00 [0.78, 1.22] | −0.74 / −0.68 [−0.79, −0.57] |
| image | perceptron, tabular | 5th | 0.28 | +0.84 / +0.13 [−0.16, 0.38] | −0.90 / −0.76 [−0.89, −0.62] |
| image | perceptron, tabular | 10th | 0.19 | +0.70 / −0.44 [−0.96, −0.00] | −0.95 / −0.81 [−1.01, −0.62] |
| image | perceptron, image | 1st | 0.73 | +0.68 / +0.66 [0.45, 0.85] | −0.69 / −0.65 [−0.75, −0.55] |
| image | perceptron, image | 5th | 0.42 | −0.29 / −0.31 [−0.53, −0.10] | −0.76 / −0.68 [−0.79, −0.58] |
| image | perceptron, image | 10th | 0.22 | −0.82 / −0.82 [−1.07, −0.56] | −0.83 / −0.70 [−0.83, −0.60] |
| image | gradient boosting, tabular | 1st | 0.68 | +0.78 / +0.74 [0.42, 1.05] | −0.70 / −0.67 [−0.78, −0.56] |
| image | gradient boosting, tabular | 5th | 0.36 | +0.13 / −0.09 [−0.47, 0.34] | −0.81 / −0.66 [−0.78, −0.55] |
| image | gradient boosting, tabular | 10th | 0.23 | +0.34 / −0.30 [−0.68, 0.27] | −0.84 / −0.66 [−0.84, −0.51] |

**Marginal and conditional contrasts.** The tertile contrast is marginal over the other concepts. The same script also balanced the two tertiles of each primary concept on the other four concepts by inverse probability weighting on the full test population. This is a different estimand, the contrast with the other concepts held at a common distribution. Table S41 shows that it differs substantially from the marginal contrast because the concepts are correlated. The learned contrasts in this paper are therefore not effects of one concept with the others held fixed.

In Table S41, original fits are averaged over seeds 0 to 2.

**TABLE S41. Marginal tertile contrasts and contrasts balanced on the other concepts.**

| Features | Concept | Δ_M0: marginal / balanced on other concepts [95% CI] | Δ_M2: marginal / balanced on other concepts [95% CI] |
| --- | --- | --- | --- |
| tabular | Color variegation | +1.02 / +1.09 [0.75, 1.38] | −0.95 / −0.42 [−0.63, −0.19] |
| tabular | Size | +0.65 / −0.06 [−0.37, 0.21] | −0.46 / −0.07 [−0.33, 0.17] |
| tabular | Lesion-skin contrast | +0.52 / −1.12 [−1.61, −0.68] | −1.50 / −0.59 [−0.79, −0.40] |
| image | Color variegation | +1.64 / +0.73 [−0.29, 1.71] | −0.66 / −0.07 [−0.25, 0.11] |
| image | Size | +1.56 / +0.09 [−0.55, 0.77] | −0.32 / +0.30 [0.12, 0.51] |
| image | Lesion-skin contrast | +1.01 / +0.35 [−0.36, 0.89] | −1.07 / −0.06 [−0.25, 0.13] |

**PAD-UFES-20.** The release holds 2,298 images of 1,641 lesions from 1,373 patients, and each learner was trained and evaluated on images. The association analysis of Fig. 2(b) uses eleven features: six symptoms, an age tertile and four image measures. The learner analysis evaluates ten features, because age and body region enter the clinical learner as inputs and the age tertile is not evaluated separately. Patients were split 60/20/20 with a fixed seed. M0 was trained on all 1,402 training images with the recorded diagnosis, and M2 on the 814 biopsied training images. Both used L2-regularized logistic regression on frozen image features or on clinical features, namely age, body region and six symptoms, and were evaluated on the same 463 test images. Intervals and probabilities come from 200 joint bootstrap resamples of training and test patients, with refitting. Table S42 reports each feature.

**TABLE S42. Learned contrasts on PAD-UFES-20.**

| Feature | Δ_M0 [95% CI] | Δ_M2 [95% CI] | Category | P(signs differ) |
| --- | --- | --- | --- | ---: |
| Itch | +1.57 [1.04, 2.35] | +0.91 [0.27, 1.66] | attenuation | 0.00 |
| Grew | +2.64 [2.07, 3.39] | +2.32 [1.65, 3.28] | attenuation | 0.00 |
| Hurt | +3.35 [2.58, 4.37] | +1.76 [1.13, 2.74] | attenuation | 0.00 |
| Changed | +1.76 [0.51, 3.00] | +0.99 [−0.18, 2.19] | attenuation | 0.04 |
| Bleeding | +3.71 [3.18, 4.59] | +2.36 [1.86, 3.14] | attenuation | 0.00 |
| Elevation | +2.96 [2.36, 3.81] | +2.37 [1.80, 3.14] | attenuation | 0.00 |
| Image color variegation | +2.06 [1.47, 2.84] | +0.70 [0.22, 1.23] | attenuation | 0.00 |
| Image lesion-skin contrast | −0.06 [−0.64, 0.52] | +0.06 [−0.31, 0.59] | unresolved (Δ_M0 near 0) | 0.28 |
| Image asymmetry | +0.53 [−0.11, 1.13] | +0.21 [−0.26, 0.61] | attenuation | 0.19 |
| Image border irregularity | +0.73 [0.02, 1.33] | +0.09 [−0.50, 0.49] | attenuation | 0.39 |

**Slope version of the gap.** Before estimating *g*, we also compared the slope of the learned logit gap on each concept, adjusted for covariates, with the logistic slope of the verified-benign indicator *V*. That comparison relies on rare disease, which Proposition 1 does not need. Its correlation across the five concepts was 0.97 in every family. Correlations over five fixed concepts are descriptive and are not inference over a population of concepts.

## S7. Plug-in bounds for the disease target

**Proposition 2 on ISIC-2024.** The nuisance *q* was the Platt-calibrated M0 and σ an out-of-fold gradient-boosting model, both fitted without the test fold. All sets in this section use the stratum formula, which is sharp for the tabular-input target and an outer set for the image-feature target. With these plug-ins the bounds are plug-in bounds, an estimate of the identified set and not the population set of Proposition 2. Table S43 reports them at three floors, together with the plug-in tipping floor s∗. This is the smallest floor on a grid of step 0.01 at which ψ_L > 0, computed on the seed-averaged curve. The last entry is the 95th percentile of s∗ over the 5,000 one-seed joint replicates of training and test patients, with validation fixed. It is a bootstrap stability threshold, not a floor identified from the data. Table S16 gives the bootstrap stability threshold with validation patients also resampled, for three estimates of *q*.

In Table S43, bounds are under the stated malignant-verification floor for the frozen-feature families, averaged over seeds 0 to 2. They replace *q* and σ by estimates and are not the population identified set of Proposition 2.

**TABLE S43. Plug-in bounds for the disease target on the learner's scale.**

| Features | Concept | ψ bounds at *s*_min = 0.5 | ψ bounds at *s*_min = 0.7 | ψ bounds at *s*_min = 0.9 | Tipping floor s∗, original / 95th percentile |
| --- | --- | --- | --- | --- | --- |
| tabular | Color variegation | [0.28, 1.67] | [0.62, 1.33] | [0.87, 1.08] | 0.38 / 0.81 |
| tabular | Size | [−0.07, 1.32] | [0.27, 0.98] | [0.52, 0.73] | 0.54 / none |
| tabular | Lesion-skin contrast | [−0.19, 1.20] | [0.14, 0.86] | [0.40, 0.61] | 0.61 / none |
| image | Color variegation | [0.04, 1.43] | [0.38, 1.09] | [0.63, 0.84] | 0.49 / 0.80 |
| image | Size | [0.00, 1.39] | [0.34, 1.06] | [0.59, 0.80] | 0.50 / 0.74 |
| image | Lesion-skin contrast | [−0.24, 1.15] | [0.09, 0.81] | [0.35, 0.56] | 0.64 / none |

**Estimated-weight approximation for the image-feature target.** Script `vr42_sharp_weighted.py` refitted M0 with seeds 0 to 2 and Platt calibration. It estimated P(*t* | *x̃*) by gradient boosting on the tabular features or on 64 principal components of the image features, cross-fitted over five folds of test patients. It then computed the outer set and an estimated-weight approximation to the sharp set on the seed-averaged curve. Stability thresholds use 1,000 test-patient resamples conditional on the fitted learners and weights, so they are not comparable to the joint thresholds of Table S16. Table S44 reports the results. The agreement column is the share of outer-tertile test lesions whose estimated weight has the sign of their observed stratum. Because the stratum is not a function of the image input, the sign of *a*_k need not match a lesion's observed stratum. This share measures how well the input predicts the stratum; it is not the accuracy of the weight sign. The estimated-weight result is exploratory.

In Table S44, M0 is Platt-calibrated; sharpness holds for the population set with the true weight and is not guaranteed after replacing it by an estimate.

**TABLE S44. Plug-in tipping floor from the stratum formula and an estimated-weight approximation.**

| Features | Concept | Agreement of estimated weight sign with observed stratum | Stratum formula (outer set): point floor / stability threshold | Estimated-weight approximation: point floor / stability threshold |
| --- | --- | ---: | --- | --- |
| tabular | Color variegation | 1.00 | 0.38 / 0.47 | 0.38 / 0.47 |
| tabular | Size | 1.00 | 0.54 / 0.63 | 0.54 / 0.63 |
| tabular | Lesion-skin contrast | 1.00 | 0.61 / 0.80 | 0.61 / 0.80 |
| image | Color variegation | 0.79 | 0.49 / 0.54 | 0.29 / 0.34 |
| image | Size | 0.74 | 0.50 / 0.55 | 0.24 / 0.29 |
| image | Lesion-skin contrast | 0.87 | 0.64 / 0.74 | 0.55 / 0.66 |

**Calibration and nuisance choices.** Script `vr25_psi_sensitivity.py` recomputed s∗ on the original fits with six estimates built on the perceptron M0. Four vary the calibration: none, Platt (Platt, 1999), isotonic (Zadrozny & Elkan, 2002) or beta calibration (Kull et al., 2017). The other two use Platt calibration, either with σ replaced by a perceptron trained on the learner's input or with logit *q* and logit *U* winsorized at the 1st and 99th percentiles. On the logit scale a Platt slope multiplies the concept contrast, so the calibration step moves s∗. Table S47 reports the results, and Table S16 gives gradient boosting under the bootstrap. Beta calibration, the alternative σ and winsorizing left the Platt value almost unchanged. Isotonic calibration moved it by about 0.1. The uncalibrated image learner, whose Platt slope was about 0.45, lowered it to below 0.2.

**Fit of the estimates of *q*.** Table S45 reports the test log-loss and Brier score of each estimate of *q* for the original fits, averaged over seeds 0 to 2. It also shows calibration by bins of predicted risk: the number of lesions, their mean predicted risk and the number of malignant lesions in each bin. Most test lesions fall below a predicted risk of 0.002, where the bins hold few malignant lesions, so these diagnostics cannot tell the estimates apart where ψ_L is decided.

In Table S45, original fits are averaged over seeds 0 to 2.

**TABLE S45. Fit of the estimates of *q* on the test population.**

| Features | Estimate of *q* | Log-loss | Brier | Predicted risk below 0.0005: lesions; expected / observed malignant | 0.0005 to 0.002 | 0.002 to 0.01 | Above 0.01 |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| tabular | perceptron, Platt | 0.00582 | 0.000979 | 62,003; 6.5 / 5.3 | 10,080; 10.0 / 15.0 | 4,812; 20.9 / 17.3 | 1,731; 53.3 / 41.3 |
| tabular | perceptron, isotonic | 0.00598 | 0.000987 | 65,733; 4.4 / 9.0 | 6,641; 6.6 / 12.7 | 4,452; 24.9 / 16.7 | 1,800; 64.4 / 40.7 |
| tabular | gradient boosting, Platt | 0.00740 | 0.001051 | 9,642; 2.2 / 2.7 | 66,182; 62.9 / 42.3 | 2,468; 8.3 / 22.0 | 332; 25.2 / 12.0 |
| tabular | perceptron, uncalibrated | 0.00583 | 0.000980 | 64,099; 5.9 / 6.7 | 8,680; 8.6 / 14.7 | 4,255; 18.5 / 17.7 | 1,591; 50.6 / 40.0 |
| image | perceptron, Platt | 0.00713 | 0.000999 | 40,705; 7.3 / 9.7 | 24,370; 25.4 / 20.3 | 12,398; 50.5 / 31.7 | 1,152; 16.5 / 17.3 |
| image | perceptron, isotonic | 0.00714 | 0.001002 | 35,376; 6.6 / 8.0 | 33,261; 31.0 / 26.7 | 9,266; 42.4 / 30.7 | 723; 25.0 / 13.7 |
| image | gradient boosting, Platt | 0.00750 | 0.001041 | 9,171; 1.9 / 3.0 | 64,828; 60.1 / 41.7 | 4,280; 14.0 / 27.0 | 346; 17.6 / 7.3 |
| image | perceptron, uncalibrated | 0.00802 | 0.001024 | 64,875; 3.6 / 29.7 | 6,966; 7.2 / 9.3 | 4,638; 20.9 / 16.0 | 2,146; 66.6 / 24.0 |

**Why the estimates differ.** Script `vr38_logit_tail.py` works within the outer tertiles of color variegation on the test population. It compares the observed number of malignant lesions, the expected number under each estimate of *q*, the mean of *q* and the mean of logit *q*. Empirical calibration approximately matches aggregate event risk, a sum of *q* over lesions, although it is not an exact constraint for every calibration method or bin. The disease target averages logit *q*, which is steep and concave near zero. Estimates with nearly the same expected events can differ widely in mean logit *q* when most lesions have very small *q*. Table S46 reports the values.

In Table S46, values are on the test population for the original fits averaged over seeds 0 to 2.

**TABLE S46. Events and logit-scale mean of *q* in the outer tertiles of color variegation.**

| Features | Estimate of *q* | Tertile | Lesions | Observed malignant | Expected (sum of *q*) | Mean *q* | Mean logit *q* | Share with *q* < 0.0005 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tabular | perceptron, Platt | lower | 27,316 | 25 | 26.9 | 0.00098 | −9.41 | 0.82 |
| tabular | perceptron, Platt | upper | 25,451 | 47 | 53.6 | 0.00211 | −8.44 | 0.66 |
| tabular | perceptron, isotonic | lower | 27,316 | 25 | 25.3 | 0.00092 | −10.89 | 0.87 |
| tabular | perceptron, isotonic | upper | 25,451 | 47 | 58.7 | 0.00231 | −9.60 | 0.73 |
| tabular | gradient boosting, Platt | lower | 27,316 | 25 | 34.4 | 0.00126 | −7.12 | 0.16 |
| tabular | gradient boosting, Platt | upper | 25,451 | 47 | 36.7 | 0.00144 | −6.82 | 0.06 |
| image | perceptron, Platt | lower | 27,316 | 25 | 26.6 | 0.00098 | −8.06 | 0.61 |
| image | perceptron, Platt | upper | 25,451 | 47 | 42.5 | 0.00167 | −7.32 | 0.40 |
| image | perceptron, isotonic | lower | 27,316 | 25 | 27.2 | 0.00099 | −7.83 | 0.55 |
| image | perceptron, isotonic | upper | 25,451 | 47 | 42.8 | 0.00168 | −7.27 | 0.34 |
| image | gradient boosting, Platt | lower | 27,316 | 25 | 27.9 | 0.00102 | −7.11 | 0.14 |
| image | gradient boosting, Platt | upper | 25,451 | 47 | 38.4 | 0.00151 | −6.93 | 0.09 |

In Table S47, all rows use the perceptron M0 and the seed-averaged curve, and "none" means that no floor up to one made ψ_L positive.

**TABLE S47. Plug-in tipping floor under calibration and nuisance choices.**

| Estimate of *q* | tabular, color | tabular, size | tabular, contrast | image, color | image, size | image, contrast |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| uncalibrated M0 | 0.37 | 0.53 | 0.60 | 0.17 | 0.18 | 0.33 |
| Platt | 0.38 | 0.54 | 0.61 | 0.49 | 0.50 | 0.64 |
| isotonic | 0.28 | 0.43 | 0.51 | 0.58 | 0.57 | 0.72 |
| beta calibration | 0.38 | 0.54 | 0.61 | 0.48 | 0.50 | 0.64 |
| Platt, σ from the learner's input | 0.38 | 0.54 | 0.61 | 0.49 | 0.50 | 0.64 |
| Platt, winsorized at 1 and 99 percent | 0.39 | 0.54 | 0.61 | 0.49 | 0.51 | 0.64 |

**Two floors.** A single floor on the lower stratum must hold even where verification is rarest. Let the floor be s_low for test lesions whose σ̂ lies below the 1st or 5th percentile threshold of Section S6 and s_high elsewhere. Table S48 gives, on a grid of step 0.05, the smallest s_low at which ψ_L > 0 for each s_high. A strong floor outside the low-support region lets a weaker one suffice inside it. This analysis uses the Platt-calibrated M0 only and is exploratory; the dependence on the estimator of *q* shown in Table S16 applies to it as well.

In Table S48, the floor elsewhere is given, and M0 is Platt-calibrated.

**TABLE S48. Smallest low-support floor that keeps ψ_L positive.**

| Features | Concept | Low-support region | s_low needed at s_high = 0.5 / 0.6 / 0.7 / 0.8 / 0.9 / 1.0 |
| --- | --- | --- | --- |
| tabular | Color variegation | below the 1st percentile, 32 percent of test lesions | 0.30 / 0.25 / 0.25 / 0.20 / 0.20 / 0.20 |
| tabular | Color variegation | below the 5th percentile, 64 percent of test lesions | 0.40 / 0.35 / 0.35 / 0.35 / 0.35 / 0.35 |
| tabular | Size | below the 1st percentile, 32 percent of test lesions | 0.60 / 0.50 / 0.40 / 0.35 / 0.30 / 0.30 |
| tabular | Size | below the 5th percentile, 64 percent of test lesions | 0.55 / 0.55 / 0.50 / 0.50 / 0.50 / 0.50 |
| tabular | Lesion-skin contrast | below the 1st percentile, 32 percent of test lesions | 0.75 / 0.65 / 0.55 / 0.45 / 0.40 / 0.35 |
| tabular | Lesion-skin contrast | below the 5th percentile, 64 percent of test lesions | 0.65 / 0.65 / 0.60 / 0.60 / 0.60 / 0.55 |
| image | Color variegation | below the 1st percentile, 32 percent of test lesions | 0.50 / 0.40 / 0.35 / 0.35 / 0.30 / 0.25 |
| image | Color variegation | below the 5th percentile, 64 percent of test lesions | 0.50 / 0.50 / 0.45 / 0.45 / 0.45 / 0.45 |
| image | Size | below the 1st percentile, 32 percent of test lesions | 0.50 / 0.45 / 0.35 / 0.30 / 0.30 / 0.25 |
| image | Size | below the 5th percentile, 64 percent of test lesions | 0.50 / 0.50 / 0.50 / 0.45 / 0.45 / 0.45 |
| image | Lesion-skin contrast | below the 1st percentile, 32 percent of test lesions | 0.85 / 0.70 / 0.60 / 0.50 / 0.45 / 0.40 |
| image | Lesion-skin contrast | below the 5th percentile, 64 percent of test lesions | 0.70 / 0.65 / 0.65 / 0.65 / 0.60 / 0.60 |

**Marginal risk ratio.** The marginal version of Section S1 identifies the disease risk ratio between the outer tertiles as above one exactly when *A* < RR_Y. A floor *s*_min on the average malignant verification in the lower tertile alone implies *A* ≤ 1/*s*_min. This analysis was specified after color variegation had emerged as the central case and is a post-inspection sensitivity analysis. Script `vr35_marginal_rr.py` computes RR_Y directly from counts, with no learner, on the test population and on the whole cohort, with 2,000 patient-cluster bootstrap replicates. The sign of RR_D is identified as positive at lower-tertile floors above 1/RR_Y, and the floor 1/q₀.₀₅ uses the 5th percentile of RR_Y. Table S50 also gives the log odds ratio of the recorded label between the tertiles among verified lesions, the verified-only association of Lemma 1. The 95 percent resampling stability threshold is the smallest floor at which two events held together in at least 95 percent of the same patient-cluster resamples: the disease risk ratio was identified as positive, and the verified-only log odds ratio was negative. This joint event is the marginal association reversal, conditional on that floor. The next column gives the same threshold on the scale of *A*. Both are thresholds of a resampling procedure, not lower confidence bounds for malignant verification, and nothing controls error for the post-inspection choice of concept and analysis. The last column is the observed benign ratio *B*_V = *g*₁/*g*₀, a plausibility benchmark for *A* rather than an estimate of it. Malignant and benign lesions may reach biopsy through different routes. The whole cohort is used because the estimand needs no fitted learner; the test population, on which the learner analyses are evaluated, is reported alongside. It is a different estimand from ψ on a different scale and population, so it is neither an estimator nor a check of ψ and does not validate Proposition 2. The site-standardized row describes the observed mixture of sites under a common within-site floor; it is a mixture estimand and does not show a reversal in each site.

**Sites and the scale of the benchmark.** Script `vr41_marginal_site.py` repeats the marginal analysis by acquisition site, standardized across sites and with one site left out at a time. The standardized risk ratio is Σ_g *w*_g P(*Y* = 1 | *t* = 1, *g*) / Σ_g *w*_g P(*Y* = 1 | *t* = 0, *g*), with one reference distribution *w*_g, the share of outer-tertile lesions in site *g*, used for both tertiles. Suppose the lower-tertile floor π₀¹(*g*) ≥ *s* holds in every site *g*. Then the standardized disease risk ratio is at least *s* times the standardized RR_Y. Each site's upper-tertile disease risk is at least its recorded risk, its lower-tertile disease risk is at most its recorded risk divided by *s*, and the common weights carry both inequalities to the sums. The same one-sided argument applies with site-specific verification. The result describes the observed mixture of sites and does not by itself identify the corresponding result at a new site. The verified-only association is pooled with the Mantel-Haenszel odds ratio. Table S49 reports the results for color variegation and size. The pooled conclusion survived standardization, but the association differed across sites. For color variegation, the two sites with the most malignant lesions would need floors near 0.8 on their own.

The script also compares two ways of transporting the appearance dependence of benign verification to malignant verification. On the ratio scale, *A* = *B*_V is impossible once π₀¹ > 1/*B*_V. On the logit scale, a common gradient γ = log *B*_V gives *A*(π₀¹, γ) = expit(logit π₀¹ + γ)/π₀¹, which falls below the largest *A* of Table S50 once π₀¹ exceeds the value marked in Fig. S3. Neither scale is identified, and clinicians may use information that differs between malignant and benign lesions, so both are benchmarks.

In Fig. S3, the contour lies in the plane of lower-tertile malignant verification π₀¹ and its logit gradient γ, for color variegation and size. To the right of each curve *A* is below the threshold and the positive disease sign is identified. As γ grows the curve approaches π₀¹ = 1/*A*, the lower-tertile floor of Table S50. Points mark γ = log *B*_V. The figure file is `figures/figS3_logit_benchmark.png`.

**Fig. S3.** Largest *A* compatible with the marginal association reversal.

In Table S49, floors are point values; the 95 percent resampling stability threshold without a site uses 500 patient-cluster resamples and the standardized row 2,000.

**TABLE S49. Marginal analysis by acquisition site.**

| Concept | Site | Malignant, lower / upper tertile | Verified benign, lower / upper | RR_Y | Floor 1/RR_Y | Verified-only log OR | 95% resampling stability threshold without this site |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Color variegation | Site 1 | 4 / 25 | 1 / 27 | 9.35 | 0.11 | −1.46 | 0.69 |
| Color variegation | Site 2 | 10 / 56 | 5 / 77 | 6.35 | 0.16 | −1.01 | 0.76 |
| Color variegation | Site 3 | 0 / 5 | 0 / 15 | none | none | not estimable | 0.62 |
| Color variegation | Site 4 | 35 / 33 | 28 / 86 | 1.31 | 0.76 | −1.18 | 0.58 |
| Color variegation | Site 5 | 65 / 93 | 47 / 183 | 1.26 | 0.79 | −1.00 | 0.40 |
| Color variegation | Site 6 | 3 / 9 | 18 / 102 | 2.22 | 0.45 | −0.64 | 0.62 |
| Color variegation | Site 7 | 1 / 11 | 1 / 6 | 10.08 | 0.10 | +0.61 | 0.63 |
| Color variegation | standardized across sites | | | 2.05 [1.56, 2.75] | | −1.02 [−1.40, −0.66] | 0.62 (all sites) |
| Size | Site 1 | 6 / 25 | 5 / 23 | 4.84 | 0.21 | −0.10 | 0.61 |
| Size | Site 2 | 9 / 56 | 5 / 77 | 5.33 | 0.19 | −0.91 | 0.67 |
| Size | Site 3 | 0 / 5 | 0 / 15 | none | none | not estimable | 0.58 |
| Size | Site 4 | 36 / 34 | 36 / 77 | 1.51 | 0.66 | −0.82 | 0.54 |
| Size | Site 5 | 67 / 99 | 54 / 175 | 1.52 | 0.66 | −0.79 | 0.43 |
| Size | Site 6 | 3 / 9 | 27 / 93 | 2.63 | 0.38 | −0.14 | 0.58 |
| Size | Site 7 | 1 / 13 | 2 / 4 | 10.49 | 0.10 | +1.87 | 0.59 |
| Size | standardized across sites | | | 2.15 [1.67, 2.82] | | −0.68 [−1.04, −0.36] | 0.58 (all sites) |

In Table S50, values come from counts with patient-cluster bootstrap intervals, and floors refer to the average malignant verification in the lower tertile. The stability threshold and the largest *A* are thresholds of the resampling procedure, not confidence bounds. The threshold lies on a grid of step 0.05 and the largest *A* on a grid of step 0.01, so they are not exact reciprocals. The observed benign ratio *B*_V is a plausibility benchmark for *A*, not an estimate of it.

**TABLE S50. Marginal risk ratio of the recorded label between the outer tertiles.**

| Population | Concept | Malignant, lower / upper tertile | RR_Y [95% CI] | Floor 1/RR_Y | Floor 1/q₀.₀₅ | Verified-only log OR [95% CI] | 95% resampling stability threshold | Largest *A* at 95% | *B*_V [95% CI] |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| whole cohort | Color variegation | 118 / 232 | 1.97 [1.48, 2.59] | 0.51 | 0.64 | −0.93 [−1.26, −0.57] | 0.65 | 1.55 | 4.96 [3.94, 6.25] |
| whole cohort | Size | 122 / 241 | 1.98 [1.54, 2.59] | 0.51 | 0.62 | −0.60 [−0.92, −0.27] | 0.65 | 1.60 | 3.60 [2.94, 4.45] |
| whole cohort | Lesion-skin contrast | 128 / 172 | 1.34 [1.01, 1.82] | 0.74 | 0.95 | −1.30 [−1.63, −0.95] | 0.95 | 1.05 | 4.93 [3.89, 6.28] |
| whole cohort | Asymmetry | 121 / 151 | 1.25 [0.93, 1.66] | 0.80 | none | +1.37 [1.03, 1.70] | none | none | 0.32 [0.25, 0.40] |
| whole cohort | Border irregularity | 97 / 158 | 1.63 [1.23, 2.18] | 0.61 | 0.78 | +1.62 [1.28, 1.97] | none | none | 0.32 [0.26, 0.40] |
| test population | Color variegation | 25 / 47 | 2.02 [1.11, 4.13] | 0.50 | 0.82 | −0.79 [−1.61, 0.00] | 0.95 | 1.10 | 4.44 [2.89, 7.91] |
| test population | Size | 25 / 45 | 1.85 [1.04, 3.68] | 0.54 | 0.89 | −0.52 [−1.30, 0.24] | none | none | 3.12 [2.08, 5.08] |
| test population | Lesion-skin contrast | 21 / 34 | 1.75 [0.89, 4.21] | 0.57 | none | −0.91 [−1.80, 0.03] | none | none | 4.38 [2.82, 7.75] |
| test population | Asymmetry | 23 / 35 | 1.41 [0.78, 2.47] | 0.71 | none | +1.31 [0.61, 2.07] | none | none | 0.38 [0.22, 0.61] |
| test population | Border irregularity | 15 / 35 | 2.13 [1.20, 4.15] | 0.47 | 0.76 | +2.02 [1.28, 2.87] | none | none | 0.28 [0.16, 0.47] |

**Lesion-level sets and a constrained learner (exploratory).** Five true floors *s*_min ∈ {0.5, 0.6, 0.7, 0.8, 0.9} generate malignant verification π¹(*x*) = *s*_min + (1 − *s*_min)·expit(*w*₁ᵀ*c*), with eight features, five concepts with positive disease effects and 300,000 lesions. Benign verification is appearance-driven, with mean 0.3 percent. For each true floor we estimate *q* and σ by gradient boosting without using the test fold. We then train an identification-constrained learner at seven assumed floors from 0.3 to 0.9. For each lesion it minimizes the larger of the two Kullback-Leibler regrets at the endpoints of the lesion-level interval of Section S1. Table S51 shows selected cells and Fig. S4 the full grid.

In Table S51, only selected cells are shown. Coverage here is the share of lesions whose true disease probability lies in the lesion-level set, a containment rate rather than the coverage of a confidence procedure. Oracle coverage uses the true nuisance functions, and plug-in coverage uses estimated nuisances.

**TABLE S51. Constrained learner on the true-by-assumed floor grid.**

| True floor | Assumed floor | Oracle coverage of true *p* | Plug-in coverage of true *p* | Mean \|log p̂ − log *p*\| | AUROC for *D* | Concept sign errors |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.3 | 1.000 | 0.663 | 0.658 | 0.740 | 0 |
| 0.5 | 0.5 | 1.000 | 0.580 | 0.337 | 0.740 | 0 |
| 0.5 | 0.9 | 0.166 | 0.109 | 0.227 | 0.739 | 0 |
| 0.7 | 0.5 | 1.000 | 0.528 | 0.413 | 0.741 | 0 |
| 0.7 | 0.7 | 1.000 | 0.377 | 0.239 | 0.741 | 0 |
| 0.7 | 0.9 | 0.313 | 0.128 | 0.170 | 0.741 | 0 |
| 0.9 | 0.5 | 1.000 | 0.451 | 0.493 | 0.725 | 0 |
| 0.9 | 0.9 | 1.000 | 0.120 | 0.210 | 0.724 | 0 |

The constrained learner never produced a concept sign error, whereas the verified-only learner produced three of five in every setting. The lesion-level set contained the true disease probability in every setting where the assumed floor did not exceed the true one, and lost coverage when the floor was overstated. The estimated set covered the truth for 11 to 69 percent of lesions, so estimation error in *q* dominates at the lesion level. At a decision threshold of 0.01 on ISIC-2024, 98.0 percent of constrained-learner decisions with tabular features did not change across floors from 0.3 to 0.9.

In Fig. S4, the panels show oracle coverage and log error. The figure file is `figures/figS4_icdl.png`.

**Fig. S4.** Constrained learner on the true-by-assumed floor grid.


## S8. Reproducibility

Table S52 maps every reported result to the script that produces it. The analysis lock, which fixed primary and secondary outcomes before the tests of alternative explanations, is in `analysis_lock.md`.

In Table S52, scripts are in `Code/` and outputs in `Result/` of the repository.

**TABLE S52. Where each result comes from.**

| Result | Script | Output |
| --- | --- | --- |
| Lemma 1 and 2 checks; slope identity | `vr1_reversal_theory.py` | `vr1_reversal_theory.json` |
| Table S4 | `vr10_symbolic.py` | `vr10_symbolic.json` |
| Unit tests | `test_reversal_theory.py` | 30 tests |
| Table S5 | `vr23_psi_sim.py` | `vr23_psi_sim.json` |
| Table S6 | `vr28_psi_stress.py` | `vr28_psi_stress.json` |
| Fig. S1 | `vr2_phase_diagram.py` | `vr2_phase_diagram.json` |
| Fig. 2; Table S10 | `vr9_closing.py` | `vr9_closing.json`; `closing_heads.npz`, not distributed |
| Table 1 point estimates (original fits, seeds 0 to 2); one-seed check in Tables S15, S37 and S43 | `vr19_primary_bootstrap.py` | `vr19_primary_bootstrap.json`, `vr19/` |
| Plug-in point tipping floor in Table S43 | `vr24_full_bootstrap.py` | `vr24_full_bootstrap.json` |
| Tables S21, S28 and S29 | `vr21_dose_calibrated.py` | `vr21_dose_calibrated.json` |
| Table S27 | `vr26_dose_poisson.py` | `vr26_dose_poisson.json` |
| Table S25 | `vr44_dose_estimand.py` | `vr44_dose_estimand.json` |
| Share of test lesions retained at each support threshold, Section IV-D | `vr45_support_retention.py` | `vr45_support_retention.json` |
| Table S31 | `vr27_bridge_eiv.py` | `vr27_bridge_eiv.json` |
| Table 3; Tables S16 and S45 | `vr30_q_bootstrap.py` | `vr30_q_bootstrap.json`, `vr30/` |
| Table S17 | `vr36_subsample.py` | `vr36_subsample.json`, `vr36/` |
| Fig. 3(a); dose results of Section IV-B; Tables S22 and S23 | `vr46_dose_fixed_val.py` | `vr46_dose_fixed_val.json`, `vr46/` |
| Tables S24 and S26 (joint training-and-validation dose) | `vr32_dose_nuisance.py` | `vr32_dose_nuisance.json` |
| Table S32 | `vr33_bridge_alt.py` | `vr33_bridge_alt.json` |
| Tables S7 and S8 | `vr34_semisynth.py` (argument `mlp` or `gbm`) | `vr34_semisynth.json`, `vr34_semisynth_gbm.json` |
| Table S39 | `vr37_overlap_sigma.py` | `vr37_overlap_sigma.json` |
| Table S46 | `vr38_logit_tail.py` | `vr38_logit_tail.json` |
| Two further seeds per replicate for the three-seed bootstrap; Table S14 | `vr39_seed_variance.py` | `vr39/`, `vr39_seed_variance.json` |
| Table S9 | `vr34_semisynth.py` (argument `local`) | `vr34_semisynth_local.json` |
| Tables S40 and S41 | `vr40_support_composition.py` | `vr40_support_composition.json`; `vr40_sigma.npz`, not distributed |
| Table S49, Fig. S3 | `vr41_marginal_site.py` | `vr41_marginal_site.json` |
| Tables S2 and S44 | `vr42_sharp_weighted.py` | `vr42_sharp_weighted.json` |
| Table 1 intervals and labels (primary three-seed bootstrap, from `vr19/` and `vr39/`); Table S18 | `vr43_three_seed_primary.py` | `vr43_three_seed_primary.json` |
| Table S50 | `vr35_marginal_rr.py` | `vr35_marginal_rr.json` |
| Table S19 (Monte Carlo audit of the one-seed check) | `vr29_bootstrap_audit.py` | `vr29_bootstrap_audit.json` |
| Tables S47 and S48 | `vr25_psi_sensitivity.py` | `vr25_psi_sensitivity.json` |
| Fig. 3(b); Table S30 | `vr22_pointwise_bridge.py` | `vr22_pointwise_bridge.json` |
| Tables S33, S34 and S38 | `vr16_mechanism.py` | `vr16_mechanism.json` |
| Tables S35 and S36; PAD-UFES-20 counts | `vr20_audit.py` | `vr20_audit.json` |
| Tables S20 and S42 | `vr12_claim_validation.py` | `vr12_claim_validation.json` |
| Slope version of the gap | `vr12_claim_validation.py --parts b` | `vr12_quantitative.json` |
| Fine-tuned and linear-probe models | `vr4_finetune.py` | `vr4_finetune.json`; `finetune/*.npz` and checkpoints, not distributed |
| Table S12; Fig. S2 | `vr5_representation.py` | `vr5_representation.json` |
| Table S13 | `vr13_headswap.py` | `vr13_headswap.json` |
| Fig. 4; PAD-UFES-20 associations | `vr6_pad_boundary.py`; ISIC-2024 reference positions from `sv2_bracket.py` | `vr6_pad_boundary.json`, `sv2_bracket.json` |
| Patient split; nuisance estimates of *q* and σ | `vl_common.py`, `vl1_identification.py`, `vl3_learning.py` | `vl_split.json`; `vl_nuisance.npz` and `vl_nuisance_image.npz`, not distributed |
| Frozen ResNet-50 embeddings | `vl0_embed.py` | `embeddings/`, not distributed |
| Table S51; Fig. S4 | `vr3_icdl.py` | `vr3_icdl.json` |
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
