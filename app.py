import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "credit-risk-matplotlib")
)

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

st.set_page_config(
    page_title="Credit Portfolio Analytics",
    page_icon=":bar_chart:",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    model = joblib.load("lgbm_model.pkl")
    columns = joblib.load("feature_columns.pkl")
    medians = joblib.load("feature_medians.pkl")
    encoders = joblib.load("label_encoders.pkl")
    return model, columns, medians, encoders


model, feature_columns, feature_medians, encoders = load_artifacts()

OPT_THRESHOLD = 0.538
HOLDOUT_LOANS = 225_611
HOLDOUT_DEFAULT_RATE = 0.2128
OPT_APPROVAL_RATE = 0.695
APPROVE_ALL_PNL = -16.1
DEFAULT_THRESHOLD_PNL = 101.1
OPT_THRESHOLD_PNL = 104.1


def dollars_millions(value):
    prefix = "-$" if value < 0 else "$"
    return f"{prefix}{abs(value):.1f}M"


def amortized_payment(principal, annual_rate, months):
    monthly_rate = (annual_rate / 100) / 12
    if monthly_rate == 0:
        return principal / months
    return principal * monthly_rate / (1 - (1 + monthly_rate) ** (-months))


def encode_category(row, col, value):
    if col not in encoders:
        return
    try:
        row[col] = int(encoders[col].transform([value])[0])
    except ValueError:
        pass


def build_feature_row(inputs):
    row = {col: feature_medians.get(col, 0) for col in feature_columns}
    row.update(inputs)
    encode_category(row, "purpose", inputs["purpose"])
    encode_category(row, "home_ownership", inputs["home_ownership"])
    encode_category(row, "grade", inputs["grade"])
    return pd.DataFrame([row])[feature_columns]


def risk_tier(score):
    if score < 0.15:
        return "Low", "#2e7d32"
    if score < 0.25:
        return "Moderate", "#b7791f"
    if score < 0.40:
        return "High", "#c05621"
    return "Very high", "#b91c1c"


def draw_score_bar(score, threshold):
    fig, ax = plt.subplots(figsize=(7.5, 1.5))
    bar_color = "#2e7d32" if score < threshold else "#b91c1c"
    ax.barh([""], [score], color=bar_color, height=0.48)
    ax.barh([""], [1 - score], left=[score], color="#e5e7eb", height=0.48)
    ax.axvline(threshold, color="#111827", linewidth=2, linestyle="--")
    ax.text(threshold + 0.01, 0.18, f"policy {threshold:.3f}", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Default risk score")
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    plt.tight_layout()
    return fig


def feature_label(name):
    labels = {
        "int_rate": "Interest rate",
        "dti": "Debt-to-income ratio",
        "fico_range_low": "FICO score",
        "fico_range_high": "FICO score",
        "loan_amnt": "Loan amount",
        "annual_inc": "Annual income",
        "installment": "Monthly installment",
        "loan_to_income": "Loan-to-income ratio",
        "installment_to_income": "Installment-to-income ratio",
        "term_months": "Loan term",
        "emp_length_yrs": "Employment length",
        "open_acc": "Open credit lines",
        "delinq_2yrs": "Recent delinquencies",
        "purpose": "Loan purpose",
        "grade": "Credit grade",
        "home_ownership": "Home ownership",
    }
    return labels.get(name, name.replace("_", " ").title())


def top_decision_drivers(input_df):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    values = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
    driver_df = (
        pd.DataFrame({"Feature": feature_columns, "Contribution": values})
        .assign(
            AbsContribution=lambda df: df["Contribution"].abs(),
            Direction=lambda df: np.where(
                df["Contribution"] > 0, "Raises risk", "Lowers risk"
            ),
        )
        .sort_values("AbsContribution", ascending=False)
        .head(8)
    )
    driver_df["Feature"] = driver_df["Feature"].map(feature_label)
    return driver_df[["Feature", "Direction", "Contribution"]]


st.title("Credit Portfolio Analytics Dashboard")
st.markdown(
    "Underwriting decision support for Lending Club loans, combining portfolio KPIs, "
    "policy thresholds, segment monitoring, and borrower-level risk review."
)

with st.sidebar:
    st.header("Policy Settings")
    policy_threshold = st.slider(
        "Approval threshold",
        min_value=0.30,
        max_value=0.75,
        value=OPT_THRESHOLD,
        step=0.005,
        help="Approve when the default risk score is below this threshold.",
    )
    st.metric("Validated optimum", f"{OPT_THRESHOLD:.3f}")
    st.caption("Optimized on the 2017-2018 out-of-time holdout.")

st.divider()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Holdout Loans", f"{HOLDOUT_LOANS:,}")
kpi2.metric("Default Rate", f"{HOLDOUT_DEFAULT_RATE:.1%}")
kpi3.metric("Approval at t=0.538", f"{OPT_APPROVAL_RATE:.1%}")
kpi4.metric("P&L vs Approve-All", "+$120.2M")

tab_review, tab_policy, tab_monitoring = st.tabs(
    ["Application Review", "Policy Analysis", "Monitoring"]
)

with tab_review:
    left, middle, right = st.columns(3)

    with left:
        st.subheader("Loan Terms")
        loan_amnt = st.slider("Loan Amount ($)", 500, 40_000, 15_000, step=500)
        int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 13.5, step=0.25)
        term_label = st.selectbox("Term", ["36 months", "60 months"])
        purpose = st.selectbox(
            "Loan Purpose",
            sorted(
                [
                    "debt_consolidation",
                    "credit_card",
                    "home_improvement",
                    "car",
                    "medical",
                    "small_business",
                    "major_purchase",
                    "moving",
                    "vacation",
                    "other",
                ]
            ),
        )

    with middle:
        st.subheader("Borrower Profile")
        annual_inc = st.number_input(
            "Annual Income ($)", 10_000, 500_000, 65_000, step=1_000
        )
        dti = st.slider("Debt-to-Income (%)", 0.0, 50.0, 18.0, step=0.5)
        fico = st.slider("FICO Score", 580, 850, 700, step=5)
        emp_length = st.selectbox(
            "Employment Length",
            [
                "< 1 year",
                "1 year",
                "2 years",
                "3 years",
                "4 years",
                "5 years",
                "6 years",
                "7 years",
                "8 years",
                "9 years",
                "10+ years",
            ],
        )

    with right:
        st.subheader("Credit Profile")
        grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
        home_ownership = st.selectbox(
            "Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"]
        )
        delinq_2yrs = st.slider("Delinquencies, past 2 years", 0, 10, 0)
        open_acc = st.slider("Open Credit Lines", 1, 40, 10)

    emp_map = {
        "< 1 year": 0,
        "1 year": 1,
        "2 years": 2,
        "3 years": 3,
        "4 years": 4,
        "5 years": 5,
        "6 years": 6,
        "7 years": 7,
        "8 years": 8,
        "9 years": 9,
        "10+ years": 10,
    }
    term_months = 36 if "36" in term_label else 60
    emp_length_yrs = emp_map[emp_length]
    installment = amortized_payment(loan_amnt, int_rate, term_months)
    loan_to_income = loan_amnt / (annual_inc + 1)
    installment_to_income = installment / (annual_inc / 12 + 1)

    input_df = build_feature_row(
        {
            "loan_amnt": loan_amnt,
            "int_rate": int_rate,
            "annual_inc": annual_inc,
            "dti": dti,
            "fico_range_low": fico,
            "fico_range_high": fico + 4,
            "term_months": term_months,
            "emp_length_yrs": emp_length_yrs,
            "installment": installment,
            "loan_to_income": loan_to_income,
            "installment_to_income": installment_to_income,
            "delinq_2yrs": delinq_2yrs,
            "open_acc": open_acc,
            "purpose": purpose,
            "home_ownership": home_ownership,
            "grade": grade,
        }
    )

    score = float(model.predict_proba(input_df)[0][1])
    approved = score < policy_threshold
    tier, tier_color = risk_tier(score)

    st.divider()
    decision_col, chart_col = st.columns([1, 2])
    with decision_col:
        st.subheader("Decision")
        decision = "Approve" if approved else "Decline / Manual Review"
        st.metric("Policy Result", decision)
        st.metric("Default Risk Score", f"{score:.1%}")
        st.markdown(
            f"<span style='color:{tier_color}; font-weight:700'>Risk tier: {tier}</span>",
            unsafe_allow_html=True,
        )
    with chart_col:
        fig = draw_score_bar(score, policy_threshold)
        st.pyplot(fig)
        plt.close(fig)

    ratio_col1, ratio_col2, ratio_col3 = st.columns(3)
    ratio_col1.metric("Monthly Payment", f"${installment:,.0f}")
    ratio_col2.metric("Loan / Income", f"{loan_to_income:.1%}")
    ratio_col3.metric("Payment / Monthly Income", f"{installment_to_income:.1%}")

    st.subheader("Top Decision Drivers")
    with st.spinner("Calculating contribution drivers..."):
        st.dataframe(
            top_decision_drivers(input_df),
            hide_index=True,
            width="stretch",
        )

with tab_policy:
    st.subheader("Policy Threshold Comparison")
    policy_df = pd.DataFrame(
        {
            "Policy": ["Approve all", "Threshold 0.500", "Threshold 0.538"],
            "Portfolio P&L": [
                dollars_millions(APPROVE_ALL_PNL),
                dollars_millions(DEFAULT_THRESHOLD_PNL),
                dollars_millions(OPT_THRESHOLD_PNL),
            ],
            "Approval Rate": ["100.0%", "~75.0%", "69.5%"],
            "Analyst Takeaway": [
                "Growth without risk screening destroys value",
                "Strong baseline policy",
                "Best tested operating point",
            ],
        }
    )
    st.dataframe(policy_df, hide_index=True, width="stretch")

    if os.path.exists("fig_18_profit_curve.png"):
        st.image("fig_18_profit_curve.png", caption="Expected portfolio P&L by threshold")

    st.info(
        "The score is useful for ranking and approval-policy decisions. Raw default "
        "probabilities need calibration before risk-based pricing."
    )

with tab_monitoring:
    st.subheader("Segment Risk and Fairness Screen")
    fairness_df = pd.DataFrame(
        {
            "Purpose": ["car", "credit_card", "debt_consolidation", "small_business"],
            "Approval Rate": ["82.4%", "75.3%", "65.7%", "48.9%"],
            "Default Rate": ["15.5%", "18.4%", "22.3%", "36.1%"],
            "DI Ratio": [1.00, 0.91, 0.80, 0.59],
            "Monitoring View": [
                "Low risk segment",
                "Within screen",
                "Borderline",
                "Flag for review",
            ],
        }
    )
    st.dataframe(fairness_df, hide_index=True, width="stretch")

    monitor_left, monitor_right = st.columns(2)
    with monitor_left:
        st.subheader("Feature Drift")
        if os.path.exists("fig_22_feature_drift.png"):
            st.image("fig_22_feature_drift.png", caption="Top-driver PSI values")
        st.markdown("Top score drivers were below PSI 0.10 in the holdout window.")
    with monitor_right:
        st.subheader("Calibration")
        if os.path.exists("fig_20_calibration.png"):
            st.image("fig_20_calibration.png", caption="Calibration curve")
        st.markdown(
            "Brier Skill Score was -0.15, so calibration is required before pricing."
        )

st.divider()
st.caption(
    "Data: Lending Club 2007-2018 closed loans. Validation: train 2007-2016, "
    "test 2017-2018. Dashboard metrics are based on the out-of-time holdout."
)
