import streamlit as st
import pandas as pd
import numpy as np
import pickle
import altair as alt

# PAGE CONFIG
st.set_page_config(
    page_title="Customer Churn Prediction System",
    layout="wide"
)

st.title("Customer Churn Prediction System")
st.write("Upload customer data to predict churn risk.")

# LOAD SAVED FILES
@st.cache_resource
def load_model():
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_scaler():
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return scaler

@st.cache_resource
def load_feature_names():
    with open("models/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    return feature_names

model = load_model()
scaler = load_scaler()
feature_names = load_feature_names()

# FILE UPLOAD
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

  
    # BASIC PREPROCESSING
    df_processed = df.copy()

    # Remove customerID if exists
    if "customerID" in df_processed.columns:
        df_processed = df_processed.drop("customerID", axis=1)

    # Convert TotalCharges if needed
    if "TotalCharges" in df_processed.columns:
        df_processed["TotalCharges"] = pd.to_numeric(
            df_processed["TotalCharges"], errors="coerce"
        )
        df_processed["TotalCharges"].fillna(
            df_processed["TotalCharges"].median(), inplace=True
        )

    # Encode categorical columns
    df_processed = pd.get_dummies(df_processed)

    # Align columns with training features
    for col in feature_names:
        if col not in df_processed.columns:
            df_processed[col] = 0

    df_processed = df_processed[feature_names]


    # SCALING
    df_scaled = scaler.transform(df_processed)

    # PREDICTIONS
    predictions = model.predict(df_scaled)
    probabilities = model.predict_proba(df_scaled)[:, 1]

    df["Churn_Prediction"] = predictions
    df["Churn_Probability"] = probabilities

    # Risk Level
    def risk_label(prob):
        if prob > 0.7:
            return "High Risk"
        elif prob > 0.4:
            return "Medium Risk"
        else:
            return "Low Risk"

    df["Risk_Level"] = df["Churn_Probability"].apply(risk_label)

    st.subheader("Prediction Results")
    st.dataframe(df)

    # METRICS
    st.subheader("Summary Metrics")

    col1, col2, col3 = st.columns(3)

    total_customers = len(df)
    high_risk = (df["Risk_Level"] == "High Risk").sum()
    avg_prob = df["Churn_Probability"].mean()

    col1.metric("Total Customers", total_customers)
    col2.metric("High Risk Customers", high_risk)
    col3.metric("Average Churn Probability", f"{avg_prob:.2f}")

    # FEATURE IMPORTANCE
    st.subheader("Top Churn Driving Features")

    if hasattr(model, "coef_"):
        importance = model.coef_[0]
    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        importance = None

    if importance is not None:
        feature_importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        })

        feature_importance_df["Importance"] = feature_importance_df["Importance"].abs()

        # Sort properly 
        feature_importance_df = (
            feature_importance_df
            .sort_values(by="Importance", ascending=False)
            .head(5)
        )

        base = alt.Chart(feature_importance_df).encode(
            x=alt.X(
                "Feature:N",
                sort="-y",
                axis=alt.Axis(labelAngle=0, title="Feature")
            ),
            y=alt.Y(
                "Importance:Q",
                axis=alt.Axis(title="Importance")
            )
        )

        # Bars
        bars = base.mark_bar()

        # Value labels on top of bars
        text = base.mark_text(
            align="center",
            dy=-5,
            color="white"
        ).encode(
            text=alt.Text("Importance:Q", format=".2f")
        )

        chart = (bars + text).properties(
            height=400
        )

        st.altair_chart(chart, use_container_width=True)

    # DOWNLOAD OPTION
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Predictions CSV",
        data=csv,
        file_name="churn_predictions.csv",
        mime="text/csv",
    )

else:
    st.info("Please upload a CSV file to start prediction.")