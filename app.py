import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


from model import (
    load_and_prepare,
    run_anomaly_detection,
    run_kmeans_segmentation,
    train_failure_model,
    score_transactions
)




st.set_page_config(
    page_title="Finance Transaction Risk Analytics",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)




st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    [data-testid="stMetricValue"] {
        font-size: 28px;
    }

    </style>
    """,
    unsafe_allow_html=True
)





TXN_PATH = "finance_transactions.csv"
CUSTOMER_PATH = "customers.csv"


# =========================================================
# CACHED DATA LOADING
# =========================================================

@st.cache_data(
    show_spinner="Loading finance transaction data..."
)
def get_data():

    return load_and_prepare(
        TXN_PATH,
        CUSTOMER_PATH
    )


# =========================================================
# CACHED ANOMALY DETECTION
# =========================================================

@st.cache_data(
    show_spinner="Running anomaly detection..."
)
def get_anomaly_data(df):

    return run_anomaly_detection(df)


# =========================================================
# CACHED CUSTOMER SEGMENTATION
# =========================================================

@st.cache_data(
    show_spinner="Creating customer segments..."
)
def get_customer_segments(df):

    return run_kmeans_segmentation(
        df,
        n_clusters=4
    )


# =========================================================
# CACHED FAILURE MODEL
# =========================================================

@st.cache_resource(
    show_spinner="Training failure prediction model..."
)
def get_failure_model(df):

    return train_failure_model(df)


# =========================================================
# LOAD DATA
# =========================================================

try:

    df = get_data()

except Exception as e:

    st.error(
        "Unable to load the finance datasets."
    )

    st.exception(e)

    st.stop()


# =========================================================
# ML PROCESSING
# =========================================================

try:

    # Anomaly detection
    df = get_anomaly_data(df)

    # Customer segmentation
    customer_segments = get_customer_segments(df)

    # Failure model
    (
        clf,
        le_dict,
        feat_cols,
        df_model,
        X_test,
        y_test,
        y_pred,
        y_prob,
        auc
    ) = get_failure_model(df)

    # Risk scoring
    df = score_transactions(
        df,
        clf,
        df_model,
        feat_cols
    )

except Exception as e:

    st.error(
        "Machine learning pipeline failed."
    )

    st.exception(e)

    st.stop()


# =========================================================
# HEADER
# =========================================================

st.title(
    "💰 Finance Transaction Risk Analytics"
)

st.caption(
    "Transaction performance, customer behaviour, "
    "failure prediction, anomaly detection and risk analytics"
)


# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.header("Dashboard Filters")


# Year
years = sorted(
    df["txn_year"]
    .dropna()
    .unique()
)

selected_year = st.sidebar.multiselect(
    "Year",
    options=years,
    default=years
)


# Customer segment
if "customer_segment" in df.columns:

    segments = sorted(
        df["customer_segment"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_segments = st.sidebar.multiselect(
        "Customer Segment",
        options=segments,
        default=segments
    )

else:

    selected_segments = []


# Occupation
if "occupation" in df.columns:

    occupations = sorted(
        df["occupation"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_occupations = st.sidebar.multiselect(
        "Occupation",
        options=occupations,
        default=occupations
    )

else:

    selected_occupations = []


# Transaction type
if "transaction_type" in df.columns:

    transaction_types = sorted(
        df["transaction_type"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_types = st.sidebar.multiselect(
        "Transaction Type",
        options=transaction_types,
        default=transaction_types
    )

else:

    selected_types = []


# =========================================================
# APPLY FILTERS
# =========================================================

filtered_df = df.copy()


if selected_year:

    filtered_df = filtered_df[
        filtered_df["txn_year"].isin(
            selected_year
        )
    ]


if selected_segments:

    filtered_df = filtered_df[
        filtered_df["customer_segment"]
        .astype(str)
        .isin(selected_segments)
    ]


if selected_occupations:

    filtered_df = filtered_df[
        filtered_df["occupation"]
        .astype(str)
        .isin(selected_occupations)
    ]


if selected_types:

    filtered_df = filtered_df[
        filtered_df["transaction_type"]
        .astype(str)
        .isin(selected_types)
    ]


# =========================================================
# EMPTY DATA CHECK
# =========================================================

if filtered_df.empty:

    st.warning(
        "No transactions match the selected filters."
    )

    st.stop()


# =========================================================
# KPI CALCULATIONS
# =========================================================

total_transactions = len(filtered_df)

total_amount = filtered_df["amount"].sum()

failed_transactions = filtered_df[
    "failed_flag"
].sum()

fraud_transactions = filtered_df[
    "fraud_flag"
].sum()

failure_rate = (
    failed_transactions
    / total_transactions
    * 100
)

fraud_rate = (
    fraud_transactions
    / total_transactions
    * 100
)

avg_transaction = (
    filtered_df["amount"].mean()
)

high_risk_count = (
    filtered_df["risk_band"]
    .isin(["High", "Very High"])
    .sum()
)

anomaly_count = (
    filtered_df["anomaly"]
    == -1
).sum()


# =========================================================
# KPI ROW
# =========================================================

c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )


with c2:

    st.metric(
        "Total Amount",
        f"₹{total_amount:,.0f}"
    )


with c3:

    st.metric(
        "Failure Rate",
        f"{failure_rate:.2f}%"
    )


with c4:

    st.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%"
    )


st.divider()


# =========================================================
# SECOND KPI ROW
# =========================================================

c5, c6, c7, c8 = st.columns(4)


with c5:

    st.metric(
        "Average Transaction",
        f"₹{avg_transaction:,.0f}"
    )


with c6:

    st.metric(
        "High Risk Transactions",
        f"{high_risk_count:,}"
    )


with c7:

    st.metric(
        "Anomalies",
        f"{anomaly_count:,}"
    )


with c8:

    st.metric(
        "Model AUC",
        f"{auc:.4f}"
    )


# =========================================================
# MONTHLY TREND
# =========================================================

st.subheader(
    "📈 Monthly Transaction Trend"
)


monthly = (
    filtered_df
    .groupby(
        ["txn_year", "txn_month"],
        as_index=False
    )
    .agg(
        total_amount=("amount", "sum"),
        transaction_count=("transaction_id", "count")
    )
)


monthly["period"] = (
    monthly["txn_year"].astype(int).astype(str)
    + "-"
    + monthly["txn_month"]
    .astype(int)
    .astype(str)
    .str.zfill(2)
)


fig_monthly = px.line(
    monthly,
    x="period",
    y="total_amount",
    markers=True,
    title="Monthly Transaction Amount"
)

fig_monthly.update_layout(
    xaxis_title="Period",
    yaxis_title="Amount (INR)"
)

st.plotly_chart(
    fig_monthly,
    use_container_width=True
)


# =========================================================
# TWO COLUMN SECTION
# =========================================================

col1, col2 = st.columns(2)


# ---------------------------------------------------------
# Transaction Status
# ---------------------------------------------------------

with col1:

    st.subheader(
        "Transaction Status"
    )

    status_data = (
        filtered_df["transaction_status"]
        .value_counts()
        .reset_index()
    )

    status_data.columns = [
        "status",
        "count"
    ]

    fig_status = px.pie(
        status_data,
        names="status",
        values="count",
        hole=0.45
    )

    st.plotly_chart(
        fig_status,
        use_container_width=True
    )


# ---------------------------------------------------------
# Transaction Type
# ---------------------------------------------------------

with col2:

    st.subheader(
        "Transaction Type"
    )

    type_data = (
        filtered_df
        .groupby("transaction_type")
        .agg(
            transaction_count=(
                "transaction_id",
                "count"
            ),
            total_amount=(
                "amount",
                "sum"
            )
        )
        .reset_index()
    )

    fig_type = px.bar(
        type_data,
        x="transaction_type",
        y="total_amount",
        title="Amount by Transaction Type"
    )

    st.plotly_chart(
        fig_type,
        use_container_width=True
    )


# =========================================================
# CUSTOMER SEGMENT ANALYSIS
# =========================================================

st.subheader(
    "👥 Customer Segment Analysis"
)


segment_data = (
    filtered_df
    .groupby("customer_segment")
    .agg(
        transactions=(
            "transaction_id",
            "count"
        ),
        total_amount=(
            "amount",
            "sum"
        ),
        avg_amount=(
            "amount",
            "mean"
        ),
        failure_rate=(
            "failed_flag",
            "mean"
        )
    )
    .reset_index()
)


segment_data["failure_rate"] *= 100


fig_segment = px.bar(
    segment_data,
    x="customer_segment",
    y="total_amount",
    title="Transaction Amount by Customer Segment",
    text_auto=".2s"
)

st.plotly_chart(
    fig_segment,
    use_container_width=True
)


# =========================================================
# CHANNEL ANALYSIS
# =========================================================

st.subheader(
    "📊 Channel Performance"
)


channel_data = (
    filtered_df
    .groupby("channel")
    .agg(
        transactions=(
            "transaction_id",
            "count"
        ),
        total_amount=(
            "amount",
            "sum"
        ),
        failure_rate=(
            "failed_flag",
            "mean"
        )
    )
    .reset_index()
)


channel_data["failure_rate"] *= 100


fig_channel = px.bar(
    channel_data,
    x="channel",
    y="failure_rate",
    title="Failure Rate by Channel",
    text_auto=".2f"
)

fig_channel.update_layout(
    yaxis_title="Failure Rate (%)"
)

st.plotly_chart(
    fig_channel,
    use_container_width=True
)


# =========================================================
# RISK DISTRIBUTION
# =========================================================

st.subheader(
    "⚠️ Transaction Risk Distribution"
)


risk_data = (
    filtered_df["risk_band"]
    .value_counts()
    .reindex(
        [
            "Very Low",
            "Low",
            "Medium",
            "High",
            "Very High"
        ],
        fill_value=0
    )
    .reset_index()
)


risk_data.columns = [
    "risk_band",
    "count"
]


fig_risk = px.bar(
    risk_data,
    x="risk_band",
    y="count",
    title="Failure Risk Bands"
)

st.plotly_chart(
    fig_risk,
    use_container_width=True
)

st.subheader(
    "🚨 Anomaly Detection"
)


anomaly_data = (
    filtered_df
    .assign(
        anomaly_label=np.where(
            filtered_df["anomaly"] == -1,
            "Anomaly",
            "Normal"
        )
    )
    ["anomaly_label"]
    .value_counts()
    .reset_index()
)


anomaly_data.columns = [
    "status",
    "count"
]


fig_anomaly = px.pie(
    anomaly_data,
    names="status",
    values="count",
    hole=0.45,
    title="Normal vs Anomalous Transactions"
)

st.plotly_chart(
    fig_anomaly,
    use_container_width=True
)




st.subheader(
    "🎯 Customer Segmentation — K-Means"
)


fig_clusters = px.scatter(
    customer_segments,
    x="pc1",
    y="pc2",
    color="cluster",
    hover_data=[
        "customer_id",
        "total_spent",
        "txn_count",
        "avg_amount",
        "fail_rate",
        "fraud_rate",
        "avg_risk"
    ],
    title="Customer Segments"
)

st.plotly_chart(
    fig_clusters,
    use_container_width=True
)




st.subheader(
    "💳 Transaction Amount vs Risk"
)


sample_df = filtered_df

if len(sample_df) > 5000:

    sample_df = sample_df.sample(
        5000,
        random_state=42
    )


fig_risk_scatter = px.scatter(
    sample_df,
    x="amount",
    y="fail_prob",
    color="risk_band",
    hover_data=[
        "transaction_id",
        "transaction_type",
        "channel",
        "transaction_status"
    ],
    title="Transaction Amount vs Failure Probability"
)

fig_risk_scatter.update_layout(
    xaxis_title="Transaction Amount (INR)",
    yaxis_title="Failure Probability"
)

st.plotly_chart(
    fig_risk_scatter,
    use_container_width=True
)




st.subheader(
    " Fraud Rate by Merchant Category"
)


merchant_data = (
    filtered_df
    .groupby("merchant_category")
    .agg(
        transactions=(
            "transaction_id",
            "count"
        ),
        fraud_rate=(
            "fraud_flag",
            "mean"
        )
    )
    .reset_index()
)


merchant_data["fraud_rate"] *= 100


fig_merchant = px.bar(
    merchant_data,
    x="merchant_category",
    y="fraud_rate",
    title="Fraud Rate by Merchant Category",
    text_auto=".2f"
)

fig_merchant.update_layout(
    yaxis_title="Fraud Rate (%)"
)

st.plotly_chart(
    fig_merchant,
    use_container_width=True
)



# DATA TABLE


st.subheader(
    "📋 Transaction Details"
)


display_columns = [
    "transaction_id",
    "transaction_date",
    "customer_id",
    "amount",
    "transaction_type",
    "transaction_status",
    "channel",
    "merchant_category",
    "customer_segment",
    "risk_score",
    "fail_prob",
    "risk_band",
    "anomaly"
]


display_columns = [
    c for c in display_columns
    if c in filtered_df.columns
]


display_df = filtered_df[
    display_columns
].copy()


st.dataframe(
    display_df,
    use_container_width=True,
    height=450
)




st.divider()

st.caption(
    f"Finance Transaction Risk Analytics • "
    f"{len(df):,} transactions • "
    f"Random Forest AUC: {auc:.4f}"
)
