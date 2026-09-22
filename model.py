"""
Finance Transaction Risk Analytics
-----------------------------------
Data loading, feature engineering, ML models,
anomaly detection, customer segmentation and risk scoring.
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



# 1. LOAD + CLEAN + FEATURE ENGINEERING


def load_and_prepare(txn_path: str, cust_path: str) -> pd.DataFrame:

    txn = pd.read_csv(txn_path)
    cust = pd.read_csv(cust_path)

    
    # Clean column names
    

    txn.columns = txn.columns.str.strip()
    cust.columns = cust.columns.str.strip()

    # Fix typo if present
    cust.rename(
        columns={"fisrt_name": "first_name"},
        inplace=True
    )

    
    # Strip whitespace from text columns
    

    for col in txn.select_dtypes(include="object").columns:
        txn[col] = txn[col].astype(str).str.strip()

    for col in cust.select_dtypes(include="object").columns:
        cust[col] = cust[col].astype(str).str.strip()

    
    # Dates
    

    txn["transaction_date"] = pd.to_datetime(
        txn["transaction_date"],
        dayfirst=True,
        errors="coerce"
    )

    cust["date_of_birth"] = pd.to_datetime(
        cust["date_of_birth"],
        dayfirst=True,
        errors="coerce"
    )

    cust["join_date"] = pd.to_datetime(
        cust["join_date"],
        dayfirst=True,
        errors="coerce"
    )

    
    # Numeric columns
    

    for col in ["amount", "fee_amount", "tax_amount", "risk_score"]:

        if col in txn.columns:

            txn[col] = pd.to_numeric(
                txn[col],
                errors="coerce"
            )

    # Missing numeric values
    txn["amount"] = txn["amount"].fillna(0)
    txn["fee_amount"] = txn["fee_amount"].fillna(0)
    txn["tax_amount"] = txn["tax_amount"].fillna(0)

    risk_median = txn["risk_score"].median()

    if pd.isna(risk_median):
        risk_median = 0

    txn["risk_score"] = txn["risk_score"].fillna(risk_median)

    
    # Binary flags
    

    txn["fraud_flag"] = (
        txn["is_fraud"]
        .astype(str)
        .str.lower()
        .map({
            "yes": 1,
            "no": 0
        })
        .fillna(0)
        .astype(int)
    )

    txn["failed_flag"] = (
        txn["transaction_status"]
        .astype(str)
        .str.lower()
        .eq("failed")
        .astype(int)
    )

    
    # Merge transaction + customer
    

    df = txn.merge(
        cust,
        on="customer_id",
        how="left"
    )

    
    # Feature Engineering
    

    ref_date = df["transaction_date"].max()

    if pd.isna(ref_date):
        ref_date = pd.Timestamp.today()

    df["txn_year"] = (
        df["transaction_date"]
        .dt.year
    )

    df["txn_month"] = (
        df["transaction_date"]
        .dt.month
    )

    df["txn_dow"] = (
        df["transaction_date"]
        .dt.dayofweek
    )

    df["txn_quarter"] = (
        df["transaction_date"]
        .dt.quarter
    )

    df["age"] = (
        (ref_date - df["date_of_birth"]).dt.days
        / 365.25
    ).round(1)

    df["tenure_months"] = (
        (ref_date - df["join_date"]).dt.days
        / 30.44
    ).round(1)

    df["total_cost"] = (
        df["amount"]
        + df["fee_amount"]
        + df["tax_amount"]
    )

    df["fee_rate"] = (
        df["fee_amount"]
        /
        df["amount"].replace(0, np.nan)
    ).fillna(0).round(4)

    amount_90 = df["amount"].quantile(0.90)

    df["high_amount"] = (
        df["amount"] > amount_90
    ).astype(int)

    
    # Customer historical features
    #
    # IMPORTANT:
    # Do not use customer failure/fraud rate here.
    # Those can leak target information.
    

    cust_agg = (
        df.groupby("customer_id")
        .agg(
            cust_txn_count=(
                "transaction_id",
                "count"
            ),
            cust_avg_amount=(
                "amount",
                "mean"
            )
        )
        .reset_index()
    )

    df = df.merge(
        cust_agg,
        on="customer_id",
        how="left"
    )

    
    # Clean infinite values
    

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    return df



# 2. ANOMALY DETECTION


def run_anomaly_detection(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    iso_features = [
        "amount",
        "fee_amount",
        "tax_amount",
        "risk_score",
        "fee_rate",
        "total_cost"
    ]

    X_iso = (
        df[iso_features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    iso = IsolationForest(
        n_estimators=100,
        contamination=0.02,
        random_state=42,
        n_jobs=-1
    )

    df["anomaly"] = iso.fit_predict(X_iso)

    df["anomaly_score"] = (
        -iso.score_samples(X_iso)
    )

    return df



# 3. CUSTOMER SEGMENTATION


def run_kmeans_segmentation(
    df: pd.DataFrame,
    n_clusters: int = 4
) -> pd.DataFrame:

    cust_feat = (
        df.groupby("customer_id")
        .agg(
            total_spent=("amount", "sum"),
            txn_count=("transaction_id", "count"),
            avg_amount=("amount", "mean"),
            fail_rate=("failed_flag", "mean"),
            fraud_rate=("fraud_flag", "mean"),
            avg_risk=("risk_score", "mean")
        )
        .reset_index()
    )

    feature_columns = [
        "total_spent",
        "txn_count",
        "avg_amount",
        "fail_rate",
        "fraud_rate",
        "avg_risk"
    ]

    X = (
        cust_feat[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    cust_feat["cluster"] = model.fit_predict(
        X_scaled
    )

    # PCA
    if len(cust_feat) >= 2:

        pca = PCA(
            n_components=2,
            random_state=42
        )

        coords = pca.fit_transform(X_scaled)

        cust_feat["pc1"] = coords[:, 0]
        cust_feat["pc2"] = coords[:, 1]

    else:

        cust_feat["pc1"] = 0
        cust_feat["pc2"] = 0

    return cust_feat



# 4. FAILURE PREDICTION


CAT_COLS = [
    "transaction_type",
    "channel",
    "merchant_category",
    "currency",
    "customer_segment",
    "occupation",
    "gender"
]

NUM_COLS = [
    "amount",
    "fee_amount",
    "tax_amount",
    "risk_score",
    "fee_rate",
    "total_cost",
    "high_amount",
    "txn_month",
    "txn_dow",
    "cust_txn_count",
    "cust_avg_amount",
    "age",
    "tenure_months"
]


def train_failure_model(df: pd.DataFrame):

    df_model = df.copy()

    
    # Encode categorical variables
    

    le_dict = {}

    for col in CAT_COLS:

        if col in df_model.columns:

            le = LabelEncoder()

            df_model[col] = le.fit_transform(
                df_model[col]
                .fillna("Unknown")
                .astype(str)
            )

            le_dict[col] = le

    
    # Select available features
    

    available_num = [
        c for c in NUM_COLS
        if c in df_model.columns
    ]

    available_cat = [
        c for c in CAT_COLS
        if c in df_model.columns
    ]

    feat_cols = (
        available_num
        + available_cat
    )

    X = (
        df_model[feat_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    y = df_model["failed_flag"].astype(int)

    
    # Safety check
    

    if y.nunique() < 2:

        raise ValueError(
            "Failure target contains only one class. "
            "Random Forest requires both successful and failed transactions."
        )

    
    # Train/Test split
    

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    
    # Random Forest
    

    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    clf.fit(
        X_train,
        y_train
    )

    
    # Predictions
    

    y_pred = clf.predict(X_test)

    y_prob = clf.predict_proba(
        X_test
    )[:, 1]

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    return (
        clf,
        le_dict,
        feat_cols,
        df_model,
        X_test,
        y_test,
        y_pred,
        y_prob,
        auc
    )



# 5. RISK SCORING


def score_transactions(
    df: pd.DataFrame,
    clf,
    df_model: pd.DataFrame,
    feat_cols: list
) -> pd.DataFrame:

    df = df.copy()

    X_score = (
        df_model[feat_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    df["fail_prob"] = (
        clf.predict_proba(X_score)[:, 1]
    )

    df["risk_band"] = pd.cut(
        df["fail_prob"],
        bins=[
            0,
            0.20,
            0.40,
            0.60,
            0.80,
            1.00
        ],
        labels=[
            "Very Low",
            "Low",
            "Medium",
            "High",
            "Very High"
        ],
        include_lowest=True
    )

    return df



# 6. MODEL PERSISTENCE


def save_model(
    clf,
    feat_cols: list,
    le_dict: dict,
    path: str = "failure_model.joblib"
):

    joblib.dump(
        {
            "model": clf,
            "features": feat_cols,
            "encoders": le_dict
        },
        path
    )


def load_model(
    path: str = "failure_model.joblib"
):

    artefact = joblib.load(path)

    return (
        artefact["model"],
        artefact["features"],
        artefact["encoders"]
    )
