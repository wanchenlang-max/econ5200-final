"""
causal_analysis.py — Reusable DML Estimation Module

Implements Double Machine Learning (Partially Linear Regression) for
estimating causal Average Treatment Effects from observational data.

Author: Wanchen Lang
Course: ECON 5200 — Causal Machine Learning & Applied Analytics
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from doubleml import DoubleMLData, DoubleMLPLR


def build_dml_data(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    controls: list[str]
) -> DoubleMLData:
    """Construct a DoubleMLData object from a DataFrame."""
    return DoubleMLData(df, y_col=outcome, d_cols=treatment, x_cols=controls)


def fit_dml_plr(
    dml_data: DoubleMLData,
    learner: str = "gbm",
    n_folds: int = 5,
    random_state: int = 42
) -> dict:
    """
    Fit a Partially Linear Regression DML model.

    Parameters
    ----------
    dml_data : DoubleMLData
    learner  : "gbm" (GradientBoosting) or "rf" (RandomForest)
    n_folds  : number of cross-fitting folds
    random_state : for reproducibility

    Returns
    -------
    dict with keys: ate, se, ci_lower, ci_upper, model
    """
    if learner == "gbm":
        ml_l = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            random_state=random_state
        )
        ml_m = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            random_state=random_state
        )
    elif learner == "rf":
        ml_l = RandomForestRegressor(
            n_estimators=100, max_features="sqrt", random_state=random_state
        )
        ml_m = RandomForestRegressor(
            n_estimators=100, max_features="sqrt", random_state=random_state
        )
    else:
        raise ValueError(f"Unknown learner '{learner}'. Use 'gbm' or 'rf'.")

    plr = DoubleMLPLR(dml_data, ml_l=ml_l, ml_m=ml_m,
                      n_folds=n_folds, score="partialling out")
    plr.fit()

    ci = plr.confint()
    return {
        "ate":      float(plr.coef[0]),
        "se":       float(plr.se[0]),
        "ci_lower": float(ci.iloc[0, 0]),
        "ci_upper": float(ci.iloc[0, 1]),
        "model":    plr,
    }


def naive_ols_ate(df: pd.DataFrame, outcome: str, treatment: str) -> dict:
    """Biased naive OLS estimate (no confounding adjustment) for comparison."""
    from statsmodels.formula.api import ols
    formula = f"{outcome} ~ {treatment}"
    res = ols(formula, data=df).fit()
    coef = res.params[treatment]
    ci = res.conf_int().loc[treatment]
    return {
        "ate":      coef,
        "ci_lower": ci[0],
        "ci_upper": ci[1],
        "pvalue":   res.pvalues[treatment],
    }


def summarize_results(naive: dict, causal: dict, label: str = "DML (GBM)") -> pd.DataFrame:
    """Return a comparison DataFrame: naive OLS vs causal DML."""
    rows = [
        {
            "Estimator": "Naive OLS (biased)",
            "ATE": naive["ate"],
            "CI Lower": naive["ci_lower"],
            "CI Upper": naive["ci_upper"],
        },
        {
            "Estimator": label,
            "ATE": causal["ate"],
            "CI Lower": causal["ci_lower"],
            "CI Upper": causal["ci_upper"],
        },
    ]
    df = pd.DataFrame(rows).set_index("Estimator")
    for col in ["ATE", "CI Lower", "CI Upper"]:
        df[col] = df[col].map("${:,.0f}".format)
    return df


# --- Quick self-test ---
if __name__ == "__main__":
    from doubleml.datasets import fetch_401K

    print("Loading SIPP 401(k) data...")
    df = fetch_401K(return_type="DataFrame")
    W = ["age", "inc", "fsize", "educ", "marr", "twoearn", "pira", "hown"]

    print("Fitting naive OLS...")
    naive = naive_ols_ate(df, outcome="net_tfa", treatment="e401")
    print(f"  Naive ATE: {naive['ate']:,.0f}  CI: [{naive['ci_lower']:,.0f}, {naive['ci_upper']:,.0f}]")

    print("Fitting DML (GBM, 5-fold)...")
    dml_data = build_dml_data(df, outcome="net_tfa", treatment="e401", controls=W)
    causal = fit_dml_plr(dml_data, learner="gbm")
    print(f"  Causal ATE: {causal['ate']:,.0f}  SE: {causal['se']:,.0f}  CI: [{causal['ci_lower']:,.0f}, {causal['ci_upper']:,.0f}]")

    print("\nSummary:")
    print(summarize_results(naive, causal))
    print("Self-test passed.")
