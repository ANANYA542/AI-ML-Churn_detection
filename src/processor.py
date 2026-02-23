import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def preprocess_data(df, _feature_names):
    """
    Cleans and aligns uploaded dataframe with training features.
    """
    df_processed = df.copy()

    # Remove customerID if exists
    if "customerID" in df_processed.columns:
        df_processed = df_processed.drop("customerID", axis=1)

    # Convert TotalCharges if needed
    if "TotalCharges" in df_processed.columns:
        df_processed["TotalCharges"] = pd.to_numeric(
            df_processed["TotalCharges"], errors="coerce"
        )
        # Handle NAs which might result from coerce
        df_processed["TotalCharges"] = df_processed["TotalCharges"].fillna(
            df_processed["TotalCharges"].median() if not df_processed["TotalCharges"].isna().all() else 0
        )

    # Encode categorical columns
    df_processed = pd.get_dummies(df_processed)

    # Align columns with training features
    for col in _feature_names:
        if col not in df_processed.columns:
            df_processed[col] = 0

    return df_processed[_feature_names]

def get_risk_label(prob):
    if prob > 0.7:
        return "High Risk"
    elif prob > 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"
