# ECON 5200 Final Project: Does 401(k) Eligibility Causally Increase Household Savings?

**Author:** Wanchen Lang | lang.w@northeastern.edu  
**Course:** ECON 5200 — Causal Machine Learning & Applied Analytics | Spring 2026

---

## Research Question

Does eligibility for an employer-sponsored 401(k) plan causally increase net household financial assets? Using Double Machine Learning (DML) on SIPP data (N=9,915), we estimate a causal ATE of **$9,581** (95% CI: [$6,682, $12,481]), compared to a naive OLS estimate of $19,559 — a 52% overstatement from income selection bias.

## Live Dashboard

[https://econ5200-final.streamlit.app](https://econ5200-final.streamlit.app) — interactive policy simulator with confounding sensitivity controls and heterogeneous effects by income quintile.

## Repository Structure

```
├── 5200-final-project-starter.ipynb   # Full analysis notebook (Parts 1–7)
├── app.py                             # Streamlit dashboard
├── causal_analysis.py                 # Reusable DML estimation module
├── requirements.txt                   # Streamlit Cloud dependencies
├── technical_report.md                # 5-page technical report (source)
├── technical_report.html              # Rendered report (print to PDF)
└── naive_vs_causal.png                # Comparison plot (naive OLS vs DML)
```

## Reproducing the Analysis

```bash
# 1. Install dependencies (Python 3.11+)
pip install doubleml scikit-learn numpy pandas matplotlib seaborn statsmodels

# 2. Run notebook
jupyter notebook 5200-final-project-starter.ipynb

# 3. Run Streamlit dashboard locally
pip install streamlit plotly
streamlit run app.py
```

## Key Results

| Estimator | ATE | 95% CI |
|-----------|-----|--------|
| Naive OLS (biased) | $19,559 | [$16,790, $22,329] |
| DML — GBM nuisance | $9,581 | [$6,682, $12,481] |
| DML — RF nuisance (robustness check) | $8,984 | [$6,408, $11,561] |

## Method

Double Machine Learning — Partially Linear Regression (`doubleml.DoubleMLPLR`), 5-fold cross-fitting with Gradient Boosted Machine nuisance models. Identification assumption: conditional independence of 401(k) eligibility given income, age, education, family size, marital status, IRA participation, and homeownership.

## References

- Chernozhukov et al. (2018). Double/debiased machine learning. *Econometrics Journal*, 21(1), C1–C68.
- Bach et al. (2022). DoubleML — An object-oriented implementation. *JMLR*, 23(53), 1–6.
- Abadie (2003). Semiparametric IV estimation. *Journal of Econometrics*, 113(2), 231–263.
