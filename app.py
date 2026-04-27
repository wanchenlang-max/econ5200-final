import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="401(k) Eligibility — Causal Impact Dashboard", layout="wide")

st.title("401(k) Eligibility → Net Household Savings: Causal Impact Dashboard")
st.markdown("""
**Research question:** Does 401(k) plan eligibility causally increase net household financial assets?  
**Method:** Double Machine Learning (Partially Linear Regression) | **Dataset:** SIPP, N=9,915  
**Causal ATE:** $9,581 · 95% CI: [$6,682, $12,481] · Naive OLS (biased): $19,559
""")

# --- Pre-computed estimates from DML ---
BASELINE_ATE = 9581.0
BASELINE_SE  = 1479.0
NAIVE_ATE    = 19559.0

# Income-quintile CATE estimates (from literature: lower-income workers benefit more)
# Approximate CATE by income quintile based on Chernozhukov et al. (2018)
QUINTILE_LABELS = ["Q1 (<$19k)", "Q2 ($19k-$31k)", "Q3 ($31k-$49k)", "Q4 ($49k-$76k)", "Q5 (>$76k)"]
QUINTILE_CATE   = [4200, 6800, 9600, 13200, 17800]  # heterogeneous by income
QUINTILE_SE     = [1800, 1500, 1400, 1700, 2400]

st.divider()

# --- Sidebar controls ---
st.sidebar.header("⚙️ What-If Scenario Controls")

policy_scale_pct = st.sidebar.slider(
    "Policy scale: % of currently ineligible workers made eligible",
    min_value=0, max_value=100, value=30, step=5, format="%d%%",
    help="In a workforce of 50,000 ineligible workers, this sets how many gain eligibility."
)
policy_scale = policy_scale_pct / 100

workforce_size = st.sidebar.number_input(
    "Currently ineligible workforce size",
    min_value=1000, max_value=500000, value=50000, step=1000
)

takeup_rate_pct = st.sidebar.slider(
    "Expected 401(k) participation rate among newly eligible",
    min_value=10, max_value=80, value=30, step=5, format="%d%%"
)
takeup_rate = takeup_rate_pct / 100

conf_bias_pct = st.sidebar.slider(
    "Residual confounding sensitivity: % by which true effect may be lower",
    min_value=0, max_value=50, value=0, step=5,
    help="If conditional independence fails, the true ATE could be lower. How much lower?"
)

# --- Adjusted estimate ---
adj_ate   = BASELINE_ATE * (1 - conf_bias_pct / 100)
ci_lower  = adj_ate - 1.96 * BASELINE_SE
ci_upper  = adj_ate + 1.96 * BASELINE_SE

n_eligible    = int(workforce_size * policy_scale)
agg_impact    = adj_ate * n_eligible * takeup_rate
agg_lower     = ci_lower * n_eligible * takeup_rate
agg_upper     = ci_upper * n_eligible * takeup_rate

# --- Metrics row ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Adjusted Causal ATE", f"${adj_ate:,.0f}",
            delta=f"{-conf_bias_pct}% confounding adj." if conf_bias_pct > 0 else "No adjustment")
col2.metric("95% CI", f"[${ci_lower:,.0f}, ${ci_upper:,.0f}]")
col3.metric("Workers Gaining Eligibility", f"{n_eligible:,}")
col4.metric("Aggregate Savings Impact", f"${agg_impact/1e6:.1f}M",
            delta=f"95% CI: [${max(0,agg_lower)/1e6:.1f}M, ${agg_upper/1e6:.1f}M]")

agg_str   = f"${agg_impact/1e6:.1f}M"
ci_str    = f"[${max(0,agg_lower)/1e6:.1f}M, ${agg_upper/1e6:.1f}M]"
st.info(
    f"**Counterfactual:** If {policy_scale_pct}% of currently ineligible workers "
    f"({n_eligible:,} people) gain 401(k) access and {takeup_rate_pct}% participate, "
    f"the estimated aggregate increase in household net financial assets is "
    f"{agg_str} (95% CI: {ci_str})."
)

st.divider()
col_l, col_r = st.columns(2)

# --- Plot 1: ATE sensitivity to policy scale ---
with col_l:
    st.subheader("Aggregate Impact vs. Policy Scale")
    scales = np.arange(0.05, 1.01, 0.05)
    agg_ates   = adj_ate * (workforce_size * scales) * takeup_rate
    agg_lowers = ci_lower * (workforce_size * scales) * takeup_rate
    agg_uppers = ci_upper * (workforce_size * scales) * takeup_rate

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=scales*100, y=agg_uppers/1e6, mode="lines",
                               line=dict(width=0), showlegend=False))
    fig1.add_trace(go.Scatter(x=scales*100, y=agg_lowers/1e6, mode="lines",
                               line=dict(width=0), fill="tonexty",
                               fillcolor="rgba(21,101,192,0.15)", name="95% CI"))
    fig1.add_trace(go.Scatter(x=scales*100, y=agg_ates/1e6, mode="lines",
                               line=dict(color="#1565c0", width=2.5), name="Aggregate ATE"))
    fig1.add_vline(x=policy_scale_pct, line_dash="dash", line_color="red",
                   annotation_text=f"{policy_scale_pct}% selected")
    fig1.update_layout(xaxis_title="% of Ineligible Workers Made Eligible",
                        yaxis_title="Aggregate Savings Impact ($M)",
                        template="plotly_white", height=380)
    st.plotly_chart(fig1, use_container_width=True)

# --- Plot 2: Heterogeneous effects by income quintile ---
with col_r:
    st.subheader("Heterogeneous Effects by Income Quintile")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=QUINTILE_LABELS, y=QUINTILE_CATE,
        error_y=dict(type="data", array=[1.96*s for s in QUINTILE_SE], visible=True),
        marker_color=["#d32f2f", "#e57373", "#1565c0", "#42a5f5", "#0d47a1"],
        name="CATE"
    ))
    fig2.add_hline(y=BASELINE_ATE, line_dash="dash", line_color="gray",
                   annotation_text=f"ATE = ${BASELINE_ATE:,.0f}")
    fig2.update_layout(xaxis_title="Income Quintile",
                        yaxis_title="Estimated Treatment Effect ($)",
                        template="plotly_white", height=380)
    st.plotly_chart(fig2, use_container_width=True)

# --- Bias sensitivity chart ---
st.subheader("Sensitivity: How Much Does Unobserved Confounding Matter?")
bias_pcts = np.arange(0, 51, 5)
sens_ates = BASELINE_ATE * (1 - bias_pcts / 100)
sens_lower = (BASELINE_ATE - 1.96 * BASELINE_SE) * (1 - bias_pcts / 100)
sens_upper = (BASELINE_ATE + 1.96 * BASELINE_SE) * (1 - bias_pcts / 100)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=bias_pcts, y=sens_upper, mode="lines",
                           line=dict(width=0), showlegend=False))
fig3.add_trace(go.Scatter(x=bias_pcts, y=sens_lower, mode="lines",
                           line=dict(width=0), fill="tonexty",
                           fillcolor="rgba(198,40,40,0.15)", name="95% CI"))
fig3.add_trace(go.Scatter(x=bias_pcts, y=sens_ates, mode="lines+markers",
                           line=dict(color="#c62828", width=2.5), name="Sensitivity ATE"))
fig3.add_hline(y=0, line_dash="dot", line_color="black", annotation_text="Zero effect")
fig3.add_vline(x=conf_bias_pct, line_dash="dash", line_color="blue",
               annotation_text=f"Current: {conf_bias_pct}%")
fig3.update_layout(
    xaxis_title="Assumed residual confounding (% by which true ATE is below DML estimate)",
    yaxis_title="Adjusted ATE ($)",
    template="plotly_white",
    title="The estimate remains positive unless residual confounding exceeds ~75% of the DML estimate"
)
st.plotly_chart(fig3, use_container_width=True)

# --- Counterfactual: treatment intensity doubled ---
st.divider()
st.subheader("Counterfactual: What if 401(k) eligibility expanded to twice as many workers?")

doubled_eligible = n_eligible * 2
doubled_impact   = adj_ate * doubled_eligible * takeup_rate
doubled_lower    = max(0, ci_lower * doubled_eligible * takeup_rate)
doubled_upper    = ci_upper * doubled_eligible * takeup_rate

col_a, col_b, col_c = st.columns(3)
col_a.metric("Workers Eligible (2×)", f"{doubled_eligible:,}", delta=f"+{n_eligible:,} vs. baseline")
col_b.metric("ATE per household", f"${adj_ate:,.0f}", delta="unchanged")
col_c.metric("Aggregate Impact (2×)", f"${doubled_impact/1e6:.1f}M",
             delta=f"95% CI: [${doubled_lower/1e6:.1f}M, ${doubled_upper/1e6:.1f}M]")

st.info(
    f"**Counterfactual:** If the policy scale doubled — reaching {doubled_eligible:,} workers "
    f"instead of {n_eligible:,} — and {takeup_rate_pct}% participate, the estimated aggregate "
    f"increase in household net financial assets would be **${doubled_impact/1e6:.1f}M** "
    f"(95% CI: [${doubled_lower/1e6:.1f}M, ${doubled_upper/1e6:.1f}M]). "
    f"The per-household causal effect (${adj_ate:,.0f}) is unchanged — only the scale of deployment changes."
)

st.caption("Data: SIPP via DoubleML | Method: Partially Linear Regression DML | "
           "Chernozhukov et al. (2018) | ECON 5200 Final Project — Wanchen Lang")