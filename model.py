"""
model.py — Finance Risk Analytics: data pipeline + ML models.

Exports
-------
load_and_prepare(txn_path, cust_path) -> pd.DataFrame
    Full pipeline: load → clean → feature-engineer → merge.

train_failure_model(df) -> (clf, le_dict, feat_cols, X_test, y_test, y_pred, y_prob, auc)
    Train Random Forest failure predictor and return artefacts.

run_anomaly_detection(df) -> pd.DataFrame
    Append 'anomaly' and 'anomaly_score' columns to df.

run_kmeans_segmentation(df) -> pd.DataFrame
    Customer-level K-Means clustering; returns cust_feat with 'cluster', 'pc1', 'pc2'.

score_transactions(df, clf, df_model, feat_cols) -> pd.DataFrame
    Append 'fail_prob' and 'risk_band' columns to df.
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ─────────────────────────────────────────────
# 1. Load & Prepare
# ─────────────────────────────────────────────

def load_and_prepare(txn_path: str, cust_path: str) -> pd.DataFrame:
    """Load CSVs, clean, engineer features, and return merged DataFrame."""
    txn  = pd.read_csv(txn_path)
    cust = pd.read_csv(cust_path)

    # Fix column names
    txn.columns  = txn.columns.str.strip()
    cust.columns = cust.columns.str.strip()
    cust.rename(columns={"fisrt_name": "first_name"}, inplace=True)

    # Strip string whitespace
    for col in txn.select_dtypes("object"):  txn[col]  = txn[col].str.strip()
    for col in cust.select_dtypes("object"): cust[col] = cust[col].str.strip()

    # Parse dates
    txn["transaction_date"] = pd.to_datetime(txn["transaction_date"], dayfirst=True, errors="coerce")
    cust["date_of_birth"]   = pd.to_datetime(cust["date_of_birth"],   dayfirst=True, errors="coerce")
    cust["join_date"]       = pd.to_datetime(cust["join_date"],       dayfirst=True, errors="coerce")

    # Fix numeric nulls
    txn["fee_amount"] = pd.to_numeric(txn["fee_amount"], errors="coerce").fillna(0)
    txn["tax_amount"] = pd.to_numeric(txn["tax_amount"], errors="coerce").fillna(0)
    txn["risk_score"] = pd.to_numeric(txn["risk_score"], errors="coerce").fillna(txn["risk_score"].median())

    # Binary flags
    txn["fraud_flag"]  = txn["is_fraud"].str.lower().map({"yes": 1, "no": 0}).fillna(0).astype(int)
    txn["failed_flag"] = (txn["transaction_status"].str.lower() == "failed").astype(int)

    # Merge
    df = txn.merge(cust, on="customer_id", how="left")

    # ── Feature Engineering ──────────────────
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


# ─────────────────────────────────────────────
# 2. Anomaly Detection
# ─────────────────────────────────────────────

def run_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Fit Isolation Forest and append anomaly columns. Returns modified df."""
    iso_features = ["amount", "fee_amount", "tax_amount", "risk_score", "fee_rate", "total_cost"]
    X_iso = df[iso_features].fillna(0)

    iso = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    df = df.copy()
    df["anomaly"]       = iso.fit_predict(X_iso)
    df["anomaly_score"] = -iso.score_samples(X_iso)
    return df


# ─────────────────────────────────────────────
# 3. Customer Segmentation
# ─────────────────────────────────────────────

def run_kmeans_segmentation(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    Compute per-customer K-Means clusters.
    Returns a customer-level DataFrame with columns:
    customer_id, total_spent, txn_count, avg_amount, fail_rate,
    fraud_rate, avg_risk, cluster, pc1, pc2.
    """
    cust_feat = df.groupby("customer_id").agg(
        total_spent = ("amount", "sum"),
        txn_count   = ("transaction_id", "count"),
        avg_amount  = ("amount", "mean"),
        fail_rate   = ("failed_flag", "mean"),
        fraud_rate  = ("fraud_flag", "mean"),
        avg_risk    = ("risk_score", "mean"),
    ).reset_index()

    X_km = StandardScaler().fit_transform(cust_feat.drop("customer_id", axis=1).fillna(0))
    cust_feat["cluster"] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X_km)

    coords = PCA(n_components=2, random_state=42).fit_transform(X_km)
    cust_feat["pc1"] = coords[:, 0]
    cust_feat["pc2"] = coords[:, 1]

    return cust_feat


# ─────────────────────────────────────────────
# 4. Failure Prediction Model
# ─────────────────────────────────────────────

CAT_COLS = ["transaction_type", "channel", "merchant_category", "currency",
            "customer_segment", "occupation", "gender"]
NUM_COLS = ["amount", "fee_amount", "tax_amount", "risk_score", "fee_rate",
            "total_cost", "high_amount", "txn_month", "txn_dow",
            "cust_txn_count", "cust_avg_amount", "cust_fail_rate", "age", "tenure_months"]


def train_failure_model(df: pd.DataFrame):
    """
    Encode features, train Random Forest, and return:
    (clf, le_dict, feat_cols, X_test, y_test, y_pred, y_prob, auc)
    """
    le_dict  = {}
    df_model = df.copy()
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


# ─────────────────────────────────────────────
# 5. Risk Scoring
# ─────────────────────────────────────────────

def score_transactions(df: pd.DataFrame, clf, df_model: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    """Append 'fail_prob' and 'risk_band' columns to df. Returns modified df."""
    df = df.copy()
    df["fail_prob"] = clf.predict_proba(df_model[feat_cols].fillna(0))[:, 1]
    df["risk_band"] = pd.cut(
        df["fail_prob"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["Very Low", "Low", "Medium", "High", "Very High"],
    )
    return df


# ─────────────────────────────────────────────
# 6. Model Persistence
# ─────────────────────────────────────────────

def save_model(clf, feat_cols: list, le_dict: dict, path: str = "failure_model.joblib"):
    joblib.dump({"model": clf, "features": feat_cols, "encoders": le_dict}, path)


def load_model(path: str = "failure_model.joblib"):
    artefact = joblib.load(path)
    return artefact["model"], artefact["features"], artefact["encoders"]
