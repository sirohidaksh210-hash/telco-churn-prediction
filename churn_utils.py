"""Shared helpers used by both the notebook (training) and the Streamlit app (inference).
Keeping them in one module guarantees the saved pipeline can be unpickled identically."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SERVICE_COLS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "NumServices"]
CATEGORICAL_FEATURES = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
                        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                        "Contract", "PaperlessBilling", "PaymentMethod"]
RAW_FEATURES = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
                "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
                "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless feature engineering (safe to apply before the train/test split - no leakage,
    because it uses only each row's own values)."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
    # Count of subscribed services (proxy for how embedded the customer is)
    df["NumServices"] = sum((df[c] == "Yes").astype(int) for c in SERVICE_COLS) + \
        (df["InternetService"] != "No").astype(int)
    return df


class Winsorizer(BaseEstimator, TransformerMixin):
    """Caps outliers at IQR fences learned ONLY from the training data (no leakage)."""
    def __init__(self, k=1.5):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(X, 25, axis=0), np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_, self.upper_ = q1 - self.k * iqr, q3 + self.k * iqr
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, names=None):
        return np.asarray(names)
