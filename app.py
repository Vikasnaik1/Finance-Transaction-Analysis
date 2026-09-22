import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
import streamlit as st

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay,
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Finance Risk Analytics", layout="wide")
st.title("💳 Finance Transaction Risk Analytics")

# Folder where this script lives — always correct on Streamlit Cloud & locally
_HERE = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════
# CACHED FUNCTIONS  (run once, reused forever)
# ══════════════════════════════════════════════

@st.cache_data(show_spinner="Loading & preparing data…")
def load_and_prepare():
    txn  = pd.read_csv(os.path.join(_HERE, "finance_transactions.csv"))
    cust = pd.read_csv(os.path.join(_HERE, "customers.csv"))

    txn.columns  = txn.columns.str.strip()
    cust.columns = cust.columns.str.strip()
    cust.rename(columns={"fisrt_name": "first_name"}, inplace=True)

    for col in txn.select_dtypes("object"):  txn[col]  = txn[col].str.strip()
    for col in cust.select_dtypes("object"): cust[col] = cust[col].str.strip()

    txn["transaction_date"] = pd.to_datetime(txn["transaction_date"], dayfirst=True, errors="coerce")
    cust["date_of_birth"]   = pd.to_datetime(cust["date_of_birth"],   dayfirst=True, errors="coerce")
    cust["join_date"]       = pd.to_datetime(cust["join_date"],       dayfirst=True, errors="coerce")

    txn["fee_amount"] = pd.to_numeric(txn["fee_amount"], errors="coerce").fillna(0)
    txn["tax_amount"] = pd.to_numeric(txn["tax_amount"], errors="coerce").fillna(0)
    txn["risk_score"] = pd.to_numeric(txn["risk_score"], errors="coerce")
    txn["risk_score"].fillna(txn["risk_score"].median(), inplace=True)

    txn["fraud_flag"]  = txn["is_fraud"].str.lower().map({"yes": 1, "no": 0}).fillna(0).astype(int)
    txn["failed_flag"] = (txn["transaction_status"].str.lower() == "failed").astype(int)

    df = txn.merge(cust, on="customer_id", how="left")

    ref = df["transaction_date"].max()
    df["txn_year"]      = df["transaction_date"].dt.year
    df["txn_month"]     = df["transaction_date"].dt.month
    df["txn_dow"]       = df["transaction_date"].dt.dayofweek
    df["txn_quarter"]   = df["transaction_date"].dt.quarter
    df["age"]           = ((ref - df["date_of_birth"]).dt.days / 365.25).round(1)
    df["tenure_months"] = ((ref - df["join_date"]).dt.days / 30.44).round(1)
    df["total_cost"]    = df["amount"] + df["fee_amount"] + df["tax_amount"]
    df["fee_rate"]      = (df["fee_amount"] / df["amount"].replace(0, np.nan)).fillna(0).round(4)
    df["high_amount"]   = (df["amount"] > df["amount"].quantile(0.90)).astype(int)

    cust_agg = df.groupby("customer_id").agg(
        cust_txn_count  = ("transaction_id", "count"),
        cust_avg_amount = ("amount", "mean"),
        cust_fail_rate  = ("failed_flag", "mean"),
        cust_fraud_rate = ("fraud_flag", "mean"),
    ).reset_index()
    df = df.merge(cust_agg, on="customer_id", how="left")
    return df


@st.cache_data(show_spinner="Running anomaly detection…")
def run_anomaly(_df):
    iso_features = ["amount", "fee_amount", "tax_amount", "risk_score", "fee_rate", "total_cost"]
    X_iso = _df[iso_features].fillna(0)
    iso = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    out = _df.copy()
    out["anomaly"]       = iso.fit_predict(X_iso)
    out["anomaly_score"] = -iso.score_samples(X_iso)
    return out


@st.cache_data(show_spinner="Running customer segmentation…")
def run_segmentation(_df):
    cust_feat = _df.groupby("customer_id").agg(
        total_spent = ("amount",         "sum"),
        txn_count   = ("transaction_id", "count"),
        avg_amount  = ("amount",         "mean"),
        fail_rate   = ("failed_flag",    "mean"),
        fraud_rate  = ("fraud_flag",     "mean"),
        avg_risk    = ("risk_score",     "mean"),
    ).reset_index()
    X_km = StandardScaler().fit_transform(cust_feat.drop("customer_id", axis=1).fillna(0))
    cust_feat["cluster"] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X_km)
    coords = PCA(n_components=2, random_state=42).fit_transform(X_km)
    cust_feat["pc1"] = coords[:, 0]
    cust_feat["pc2"] = coords[:, 1]
    return cust_feat


@st.cache_resource(show_spinner="Training failure model…")
def train_model(_df):
    CAT_COLS = ["transaction_type", "channel", "merchant_category", "currency",
                "customer_segment", "occupation", "gender"]
    NUM_COLS = ["amount", "fee_amount", "tax_amount", "risk_score", "fee_rate",
                "total_cost", "high_amount", "txn_month", "txn_dow",
                "cust_txn_count", "cust_avg_amount", "cust_fail_rate", "age", "tenure_months"]

    le_dict  = {}
    df_model = _df.copy()
    for c in CAT_COLS:
        if c in df_model.columns:
            le = LabelEncoder()
            df_model[c] = le.fit_transform(df_model[c].astype(str))
            le_dict[c]  = le

    feat_cols = NUM_COLS + [c for c in CAT_COLS if c in df_model.columns]
    X = df_model[feat_cols].fillna(0)
    y = df_model["failed_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=150, max_depth=12,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    return clf, le_dict, feat_cols, df_model, X_test, y_test, y_pred, y_prob, auc


# ══════════════════════════════════════════════
# RUN ALL HEAVY WORK  (cached after first run)
# ══════════════════════════════════════════════
df        = load_and_prepare()
df        = run_anomaly(df)
cust_feat = run_segmentation(df)
clf, le_dict, feat_cols, df_model, X_test, y_test, y_pred, y_prob, auc = train_model(df)

df = df.copy()
df["fail_prob"] = clf.predict_proba(df_model[feat_cols].fillna(0))[:, 1]
df["risk_band"] = pd.cut(
    df["fail_prob"],
    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    labels=["Very Low", "Low", "Medium", "High", "Very High"],
)

# ══════════════════════════════════════════════
# 1. BUSINESS UNDERSTANDING
# ══════════════════════════════════════════════
st.header("1. 🏦 Business Understanding")
st.markdown("""
**Goal:** Analyse 50,000+ banking transactions to:
- Detect fraudulent / anomalous transactions
- Predict transaction failures
- Score customer risk
- Understand customer behaviour by segment, channel and merchant category
""")

# ══════════════════════════════════════════════
# 2. DATA CLEANING & FEATURE ENGINEERING
# ══════════════════════════════════════════════
st.header("2. 🧹 Data Cleaning & Feature Engineering")

col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", f"{len(df):,}")
col2.metric("Unique Customers",   f"{df['customer_id'].nunique():,}")
col3.metric("Null Values",        int(df.isnull().sum().sum()))

with st.expander("Cleaned Data Sample"):
    st.dataframe(df.head(20), use_container_width=True)

st.success("Features added: age, tenure_months, total_cost, fee_rate, high_amount, txn_month, txn_dow, cust_fail_rate, cust_fraud_rate")

with st.expander("Feature Sample"):
    st.dataframe(
        df[["transaction_id", "amount", "total_cost", "fee_rate",
            "age", "tenure_months", "failed_flag", "fraud_flag"]].head(10),
        use_container_width=True,
    )

# ══════════════════════════════════════════════
# 3. EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════
st.header("3. 🔍 Exploratory Data Analysis")

tab1, tab2, tab3, tab4 = st.tabs(["Amount Distribution", "Transaction Types", "Status Split", "Monthly Trend"])

with tab1:
    fig = px.histogram(df, x="amount", nbins=60, title="Transaction Amount Distribution",
                       color_discrete_sequence=["#3b82d4"])
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    vc = df["transaction_type"].value_counts().reset_index()
    vc.columns = ["type", "count"]
    fig = px.bar(vc, x="type", y="count", title="Transaction Type Counts",
                 color="count", color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    sc = df["transaction_status"].value_counts().reset_index()
    sc.columns = ["status", "count"]
    fig = px.pie(sc, names="status", values="count", title="Transaction Status Split")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    monthly = df.groupby(["txn_year", "txn_month"])["amount"].sum().reset_index()
    monthly["period"] = (
        monthly["txn_year"].astype(str) + "-" + monthly["txn_month"].astype(str).str.zfill(2)
    )
    fig = px.line(monthly, x="period", y="amount",
                  title="Monthly Transaction Volume (INR)", markers=True)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════
# 4. CUSTOMER BEHAVIOUR ANALYSIS
# ══════════════════════════════════════════════
st.header("4. 👤 Customer Behaviour Analysis")

tab_a, tab_b, tab_c = st.tabs(["By Segment", "By Channel", "By Merchant Category"])

with tab_a:
    seg = df.groupby("customer_segment").agg(
        txn_count  = ("transaction_id", "count"),
        avg_amount = ("amount",         "mean"),
        fraud_rate = ("fraud_flag",     "mean"),
        fail_rate  = ("failed_flag",    "mean"),
    ).reset_index()
    st.dataframe(
        seg.style.format({"avg_amount": "{:.0f}", "fraud_rate": "{:.3f}", "fail_rate": "{:.3f}"}),
        use_container_width=True,
    )
    fig = px.bar(seg, x="customer_segment", y="txn_count", color="fraud_rate",
                 title="Transactions per Segment (colour = fraud rate)", color_continuous_scale="Reds")
    st.plotly_chart(fig, use_container_width=True)

with tab_b:
    ch = (df.groupby("channel")
          .agg(txn_count=("transaction_id", "count"), avg_amount=("amount", "mean"), fail_rate=("failed_flag", "mean"))
          .reset_index().sort_values("txn_count", ascending=False))
    fig = px.bar(ch, x="channel", y="txn_count", color="fail_rate",
                 title="Transactions per Channel (colour = failure rate)", color_continuous_scale="Oranges")
    st.plotly_chart(fig, use_container_width=True)

with tab_c:
    mc = (df.groupby("merchant_category")
          .agg(txn_count=("transaction_id", "count"), fraud_rate=("fraud_flag", "mean"))
          .reset_index().sort_values("fraud_rate", ascending=False))
    fig = px.bar(mc, x="merchant_category", y="fraud_rate",
                 title="Fraud Rate by Merchant Category", color="fraud_rate", color_continuous_scale="Reds")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════
# 5. TRANSACTION FAILURE ANALYSIS
# ══════════════════════════════════════════════
st.header("5. ❌ Transaction Failure Analysis")

col_a, col_b = st.columns(2)
with col_a:
    fc = df.groupby("channel")["failed_flag"].mean().sort_values(ascending=False).reset_index()
    fc.columns = ["channel", "failure_rate"]
    fig = px.bar(fc, x="channel", y="failure_rate", title="Failure Rate by Channel",
                 color="failure_rate", color_continuous_scale="Reds")
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    ft = df.groupby("transaction_type")["failed_flag"].mean().sort_values(ascending=False).reset_index()
    ft.columns = ["type", "failure_rate"]
    fig = px.bar(ft, x="type", y="failure_rate", title="Failure Rate by Transaction Type",
                 color="failure_rate", color_continuous_scale="Oranges")
    st.plotly_chart(fig, use_container_width=True)

fig = px.box(df, x="transaction_status", y="amount", color="transaction_status",
             title="Amount Distribution: Failed vs Success")
st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════
# 6. ANOMALY DETECTION
# ══════════════════════════════════════════════
st.header("6. 🚨 Anomaly Detection (Isolation Forest)")

st.metric("Anomalies Detected (2% contamination)", f"{(df['anomaly'] == -1).sum():,}")

sample = df.sample(min(5000, len(df)), random_state=1)
fig = px.scatter(sample, x="amount", y="risk_score",
                 color=sample["anomaly"].map({1: "Normal", -1: "Anomaly"}),
                 title="Anomaly Detection — Amount vs Risk Score",
                 color_discrete_map={"Normal": "#3b82d4", "Anomaly": "red"}, opacity=0.5)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Top 20 Anomalous Transactions"):
    st.dataframe(
        df[df["anomaly"] == -1][
            ["transaction_id", "transaction_date", "customer_id",
             "amount", "channel", "transaction_type", "transaction_status", "anomaly_score"]
        ].sort_values("anomaly_score", ascending=False).head(20),
        use_container_width=True,
    )

# ══════════════════════════════════════════════
# 7. CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════
st.header("7. 🗂️ Customer Segmentation (K-Means)")

fig = px.scatter(cust_feat, x="pc1", y="pc2", color=cust_feat["cluster"].astype(str),
                 title="Customer Clusters (PCA 2-D)", labels={"color": "Cluster"}, opacity=0.6)
st.plotly_chart(fig, use_container_width=True)

st.dataframe(
    cust_feat.groupby("cluster")[
        ["total_spent", "txn_count", "avg_amount", "fail_rate", "fraud_rate", "avg_risk"]
    ].mean().round(2),
    use_container_width=True,
)

# ══════════════════════════════════════════════
# 8. FAILURE PREDICTION MODEL
# ══════════════════════════════════════════════
st.header("8. 🤖 Failure Prediction Model")
st.success("✅ Random Forest trained on 80% of data.")

# ══════════════════════════════════════════════
# 9. FEATURE IMPORTANCE
# ══════════════════════════════════════════════
st.header("9. 📊 Feature Importance")
fi = pd.DataFrame({"feature": feat_cols, "importance": clf.feature_importances_})
fi = fi.sort_values("importance", ascending=False).head(15)
fig = px.bar(fi, x="importance", y="feature", orientation="h",
             title="Top 15 Feature Importances", color="importance", color_continuous_scale="Blues")
fig.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════
# 10. MODEL EVALUATION
# ══════════════════════════════════════════════
st.header("10. 📈 Model Evaluation")

col1, col2 = st.columns(2)
col1.metric("ROC-AUC", f"{auc:.4f}")
col2.metric("Test Samples", f"{len(y_test):,}")

report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
st.dataframe(report.style.format(precision=3), use_container_width=True)

col_r1, col_r2 = st.columns(2)
with col_r1:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}", color="#3b82d4")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve"); ax.legend()
    st.pyplot(fig)
    plt.close(fig)

with col_r2:
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_pred), display_labels=["Success", "Failed"]
    ).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    plt.close(fig)

# ══════════════════════════════════════════════
# 11. RISK SCORING
# ══════════════════════════════════════════════
st.header("11. 🎯 Risk Scoring")

band_counts = df["risk_band"].value_counts().sort_index().reset_index()
band_counts.columns = ["band", "count"]
fig = px.bar(band_counts, x="band", y="count", color="band",
             title="Transactions by Risk Band",
             color_discrete_sequence=["#22c55e", "#86efac", "#fbbf24", "#f97316", "#ef4444"])
st.plotly_chart(fig, use_container_width=True)

with st.expander("High / Very High Risk Transactions"):
    st.dataframe(
        df[df["risk_band"].isin(["High", "Very High"])][
            ["transaction_id", "transaction_date", "customer_id", "amount",
             "channel", "transaction_type", "transaction_status", "fail_prob", "risk_band"]
        ].sort_values("fail_prob", ascending=False).head(200),
        use_container_width=True,
    )

# ══════════════════════════════════════════════
# 12. MODEL SAVING
# ══════════════════════════════════════════════
st.header("12. 💾 Model Saving")

MODEL_PATH = os.path.join(_HERE, "failure_model.joblib")

if st.button("💾 Save Model"):
    joblib.dump({"model": clf, "features": feat_cols, "encoders": le_dict}, MODEL_PATH)
    st.success(f"Model saved → {MODEL_PATH}")

if os.path.exists(MODEL_PATH):
    st.info(f"✅ Model file exists: `{MODEL_PATH}` ({os.path.getsize(MODEL_PATH)/1024:.1f} KB)")

# ══════════════════════════════════════════════
# 13. FINAL SUMMARY
# ══════════════════════════════════════════════
st.header("13. ✅ Final Summary")

st.markdown(f"""
| Metric | Value |
|---|---|
| Total Transactions | {len(df):,} |
| Unique Customers | {df['customer_id'].nunique():,} |
| Failure Rate | {df['failed_flag'].mean()*100:.2f}% |
| Fraud Rate | {df['fraud_flag'].mean()*100:.2f}% |
| Anomalies Detected | {(df['anomaly']==-1).sum():,} |
| Model ROC-AUC | {auc:.4f} |
| High / Very High Risk Transactions | {df['risk_band'].isin(['High','Very High']).sum():,} |
""")

# ══════════════════════════════════════════════
# 14. LIVE TRANSACTION RISK PREDICTOR
# ══════════════════════════════════════════════
st.header("14. 🔮 Live Transaction Risk Predictor")

with st.form("predict_form"):
    c1, c2, c3 = st.columns(3)
    amount_in = c1.number_input("Amount (INR)", min_value=1.0,  value=10000.0)
    fee_in    = c2.number_input("Fee Amount",   min_value=0.0,  value=50.0)
    tax_in    = c3.number_input("Tax Amount",   min_value=0.0,  value=9.0)

    c4, c5, c6 = st.columns(3)
    risk_in   = c4.slider("Risk Score", 0, 100, 50)
    month_in  = c5.selectbox("Month", list(range(1, 13)))
    dow_in    = c6.selectbox("Day of Week (0=Mon)", list(range(7)))

    c7, c8 = st.columns(2)
    type_in    = c7.selectbox("Transaction Type", sorted(df["transaction_type"].dropna().unique()))
    channel_in = c8.selectbox("Channel",          sorted(df["channel"].dropna().unique()))

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    row = {c: 0 for c in feat_cols}
    row["amount"]          = amount_in
    row["fee_amount"]      = fee_in
    row["tax_amount"]      = tax_in
    row["risk_score"]      = risk_in
    row["total_cost"]      = amount_in + fee_in + tax_in
    row["fee_rate"]        = fee_in / amount_in if amount_in else 0
    row["high_amount"]     = int(amount_in > df["amount"].quantile(0.90))
    row["txn_month"]       = month_in
    row["txn_dow"]         = dow_in
    row["cust_txn_count"]  = df["cust_txn_count"].mean()
    row["cust_avg_amount"] = df["cust_avg_amount"].mean()
    row["cust_fail_rate"]  = df["cust_fail_rate"].mean()
    row["age"]             = df["age"].mean()
    row["tenure_months"]   = df["tenure_months"].mean()

    for c, le in le_dict.items():
        val = type_in if c == "transaction_type" else channel_in if c == "channel" else le.classes_[0]
        val = val if val in set(le.classes_) else le.classes_[0]
        row[c] = int(le.transform([val])[0])

    X_live = pd.DataFrame([row])[feat_cols].fillna(0)
    prob   = clf.predict_proba(X_live)[0][1]
    band   = pd.cut([prob], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    labels=["Very Low", "Low", "Medium", "High", "Very High"])[0]

    colour = {"Very Low": "green", "Low": "lightgreen", "Medium": "orange",
              "High": "orangered", "Very High": "red"}.get(str(band), "grey")

    st.markdown(f"### Failure Probability: `{prob*100:.1f}%`")
    st.markdown(
        f"### Risk Band: <span style='color:{colour};font-weight:bold'>{band}</span>",
        unsafe_allow_html=True,
    )
    st.progress(float(prob))
