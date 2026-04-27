# Technical Report: Does 401(k) Eligibility Causally Increase Household Savings?

**ECON 5200: Causal Machine Learning & Applied Analytics**  
**Julian Lechner | Spring 2026**

---

## 1. Research Question

Does eligibility for an employer-sponsored 401(k) retirement savings plan causally increase net household financial assets? This question is policy-relevant for financial services firms, policymakers evaluating SECURE 2.0 provisions, and employers deciding whether to extend benefit coverage to part-time and seasonal workers. The answer cannot be obtained from a predictive model alone — it requires a credible causal identification strategy, for reasons developed in Section 4.

---

## 2. Data

We use the Survey of Income and Program Participation (SIPP), a nationally representative household survey, accessed via the `doubleml` Python package (Chernozhukov et al., 2018; Abadie, 2003). The dataset contains **9,915 household observations**.

**Key variables:**

| Variable | Description | Type |
|----------|-------------|------|
| `net_tfa` | Net total financial assets ($) | Outcome |
| `e401` | 401(k) plan eligibility | Treatment (binary) |
| `inc` | Annual family income ($) | Confounder |
| `age` | Age in years | Confounder |
| `fsize` | Family size | Confounder |
| `educ` | Years of education | Confounder |
| `marr` | Married (0/1) | Confounder |
| `twoearn` | Two-earner household (0/1) | Confounder |
| `pira` | IRA participation (0/1) | Confounder |
| `hown` | Homeowner (0/1) | Confounder |

**Descriptive statistics:** Net total financial assets (`net_tfa`) have a mean of $18,052 and a median of $1,499, reflecting a heavily right-skewed distribution. Approximately 39% of households in the sample are eligible for a 401(k) plan.

**Balance check:** A pre-treatment comparison reveals substantial differences between eligible and ineligible households. Eligible households earn $46,862 annually on average versus $31,494 for ineligible households — a gap of $15,368. They are also more likely to hold IRAs and own homes. This imbalance is the core motivation for a causal approach: naively comparing eligible to ineligible households will conflate the effect of plan access with the effect of being a higher-income worker at a better employer.

---

## 3. Identification Strategy

**Strategy:** Double Machine Learning (DML) — Partially Linear Regression (PLR) specification.

**Identifying assumption:** Conditional independence. After controlling for income, age, education, family size, marital status, IRA participation, and homeownership, the assignment of 401(k) eligibility — which is determined at the employer level — is uncorrelated with unobserved savings preferences.

**Intuition:** Two workers with identical income, age, education, and family structure may work at different employers, one of which happens to offer a 401(k) plan. Conditional on all observable characteristics, the variation in eligibility comes from employer-level plan decisions that are plausibly unrelated to any individual worker's savings preferences. This is the identifying variation we exploit.

**Why not an experiment?** 401(k) eligibility cannot be randomly assigned in observational data — it is a firm-level policy decision correlated with employer quality. This creates the confounding described above, which DML is designed to address.

---

## 4. Prediction vs. Causation: Why a Predictive Model Is Insufficient

A natural first instinct is to train a machine learning model — say, a Random Forest — to predict `net_tfa` from all available features including `e401`. This produces a prediction R² of 0.137, meaning the model explains about 14% of variance in net financial assets. The model would assign non-trivial feature importance to `e401`.

**But this does not answer the causal question.** Here is why:

1. **The predictive model answers:** "Given that I observe a worker who is eligible for a 401(k), what would I predict their savings to be?" This is a description of the world as it is.

2. **The causal question asks:** "If I make this worker eligible for a 401(k) — changing their status from 0 to 1 — what happens to their savings?" This is a counterfactual about intervening in the world.

These are fundamentally different questions. The Random Forest's feature importance for `e401` reflects the *correlation* between eligibility and savings, which includes both the causal effect *and* the income selection effect. A worker's employer type (which determines eligibility) is correlated with their income, financial literacy, and savings preferences — all of which independently increase `net_tfa`. A predictive model cannot untangle these channels.

**Concretely:** the predictive association between `e401` and `net_tfa` is approximately $19,559 (the naive OLS estimate). The causal effect, after removing income confounding, is $9,581. A policy decision based on the predictive figure would overstate the expected impact of extending eligibility by approximately $10,000 per household — a 52% overstatement. At scale (50,000 workers, 30% participation), this translates to a $74M error in projected aggregate savings impact.

The fundamental problem is that the predictive model's job is to minimize prediction error on observed data, not to simulate the outcome of an intervention. These objectives diverge whenever the treatment variable is correlated with unobserved determinants of the outcome — which is almost always true in observational settings.

---

## 5. Methodology: Double Machine Learning

We implement the Partially Linear Regression (PLR) variant of Double Machine Learning (Chernozhukov et al., 2018). The model is:

$$Y_i = \theta \cdot T_i + g(W_i) + \varepsilon_i$$

where $Y_i$ is net financial assets, $T_i$ is 401(k) eligibility, $W_i$ is the vector of eight confounders, $g(\cdot)$ is an unknown function approximated by machine learning, and $\theta$ is the causal average treatment effect we wish to estimate.

**Estimation procedure (cross-fitting):**

1. Split data into 5 folds. For each fold, train nuisance models on the remaining 4 folds.
2. **Outcome nuisance:** Predict $Y$ from $W$ using a Gradient Boosted Machine (GBM) with 100 trees, depth 3, learning rate 0.05. Compute residuals $\tilde{Y}_i = Y_i - \hat{E}[Y_i | W_i]$.
3. **Treatment nuisance:** Predict $T$ from $W$ using the same GBM specification. Compute residuals $\tilde{T}_i = T_i - \hat{E}[T_i | W_i]$.
4. **Final stage:** Regress $\tilde{Y}$ on $\tilde{T}$ via OLS. The coefficient is $\hat{\theta}$, the causal ATE.

**Why cross-fitting?** Using the same data to fit the nuisance models and estimate $\theta$ introduces regularization bias — machine learning models overfit to their training data, and this overfitting corrupts inference on $\theta$. Cross-fitting (Neyman orthogonalization) eliminates this bias, ensuring valid confidence intervals even when the nuisance models are high-dimensional approximations.

**Why GBM for nuisance models?** The relationship between income and savings is highly nonlinear (savings accelerate sharply at higher income levels), and GBMs capture this nonlinearity better than linear regression. This produces cleaner residuals $\tilde{Y}$ and $\tilde{T}$, which in turn produce a less biased estimate of $\theta$.

---

## 6. Results

### 6.1 Naive OLS Estimate

A simple OLS regression of `net_tfa` on `e401` (no controls) yields:

- **Naive ATE: $19,559** (95% CI: [$16,790, $22,329])

This estimate is substantially upward biased due to income confounding.

### 6.2 Causal DML Estimate

After applying DML with GBM nuisance models:

- **Causal ATE: $9,581** (SE: $1,479, 95% CI: [$6,682, $12,481])

Interpretation: 401(k) eligibility causally increases net household financial assets by $9,581, on average, after controlling for income, age, education, family size, and other observable confounders.

### 6.3 Naive vs. Causal Comparison

| Estimator | ATE | 95% CI |
|-----------|-----|--------|
| Naive OLS | $19,559 | [$16,790, $22,329] |
| DML (GBM nuisance) | $9,581 | [$6,682, $12,481] |
| DML (RF nuisance) | $8,984 | [$6,408, $11,561] |

The difference between naive and causal estimates ($9,978) represents the income selection bias — savings differences that would exist between these households even in the absence of 401(k) access. The DML estimate removes this confound.

---

## 7. Robustness Checks

To verify the causal estimate is not an artifact of the GBM specification, we re-estimate the DML model replacing GBM nuisance models with Random Forest regressors (100 trees, `max_features='sqrt'`).

**Result:** The RF nuisance model yields an ATE of **$8,984** (95% CI: [$6,408, $11,561]) — a difference of $597 from the GBM estimate, well within one standard error of either model. This stability across structurally different learners provides strong evidence that the causal estimate reflects a genuine signal rather than a modeling artifact.

Additionally, a bias sensitivity analysis shows that the causal effect remains positive unless unobserved confounding accounts for more than approximately 75% of the total DML estimate — an implausibly large residual given that our observables explain substantial variation in both treatment and outcome.

---

## 8. Policy Implications

The causal ATE of $9,581 per household is economically and statistically significant. At 30% take-up among 50,000 newly eligible workers, this implies approximately $143.7 million in new aggregate retirement savings. Even at the lower bound of the 95% confidence interval ($6,682), the implied benefit-cost ratio exceeds 13-to-1 relative to typical 401(k) plan administration costs of ~$500 per eligible employee.

**Recommendation:** Extend 401(k) eligibility to currently ineligible workers. The causal evidence supports a meaningful savings impact. We recommend a staggered rollout with pre-registered measurement — ideally exploiting variation in rollout timing across business units as a natural experiment — to validate the contemporary effect with more recent data.

---

## References

- Abadie, A. (2003). Semiparametric instrumental variable estimation of treatment response models. *Journal of Econometrics*, 113(2), 231–263.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Bach, P., Chernozhukov, V., Kurz, M. S., & Spindler, M. (2022). DoubleML — An object-oriented implementation of double machine learning in Python. *Journal of Machine Learning Research*, 23(53), 1–6.
