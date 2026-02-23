import streamlit as st
import pandas as pd
import numpy as np
import pickle
import altair as alt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Customer Churn Prediction System",
    layout="wide"
)

st.title("Customer Churn Prediction System")
st.write("Upload customer data to predict churn risk.")

# =========================
# LOAD SAVED FILES
# =========================
@st.cache_resource
def load_model():
    with open("models/model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    with open("models/scaler.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_feature_names():
    with open("models/feature_names.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
scaler = load_scaler()
feature_names = load_feature_names()

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # =========================
    # PREPROCESSING
    # =========================
    df_processed = df.copy()

    if "customerID" in df_processed.columns:
        df_processed = df_processed.drop("customerID", axis=1)

    if "TotalCharges" in df_processed.columns:
        df_processed["TotalCharges"] = pd.to_numeric(
            df_processed["TotalCharges"], errors="coerce"
        )
        df_processed["TotalCharges"] = df_processed["TotalCharges"].fillna(
            df_processed["TotalCharges"].median()
        )

    df_processed = pd.get_dummies(df_processed)

    for col in feature_names:
        if col not in df_processed.columns:
            df_processed[col] = 0

    df_processed = df_processed[feature_names]

    # =========================
    # PREDICTIONS
    # =========================
    df_scaled = scaler.transform(df_processed)

    predictions = model.predict(df_scaled)
    probabilities = model.predict_proba(df_scaled)[:, 1]

    df["Churn_Prediction"] = predictions
    df["Churn_Probability"] = probabilities

    def risk_label(prob):
        if prob > 0.7:
            return "High Risk"
        elif prob > 0.4:
            return "Medium Risk"
        else:
            return "Low Risk"

    df["Risk_Level"] = df["Churn_Probability"].apply(risk_label)

    # =====================================================
    # 1️⃣ SUMMARY METRICS (MINIMAL CARDS)
    # =====================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Summary Metrics")

    total_customers = len(df)
    high_risk = (df["Risk_Level"] == "High Risk").sum()
    avg_prob = df["Churn_Probability"].mean()

    st.markdown(
        """
        <style>
        .metric-card {
            background-color: rgba(255, 255, 255, 0.03);
            padding: 30px;
            border-radius: 14px;
            border: 1px solid rgba(76, 120, 168, 0.4);
            text-align: center;
            transition: 0.3s ease;
        }
        .metric-card:hover {
            border: 1px solid rgba(76, 120, 168, 0.7);
        }
        .metric-title {
            font-size: 14px;
            color: #9ca3af;
            margin-bottom: 10px;
            letter-spacing: 0.5px;
        }
        .metric-value {
            font-size: 38px;
            font-weight: 600;
            color: white;
        }
        div.stDownloadButton > button {
            border: 1px solid rgba(76, 120, 168, 0.4);
            background-color: transparent;
            color: white;
            border-radius: 8px;
        }

        div.stDownloadButton > button:hover {
            border: 1px solid rgba(76, 120, 168, 0.7);
            background-color: rgba(76, 120, 168, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Customers</div>
                <div class="metric-value">{total_customers}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">High Risk Customers</div>
                <div class="metric-value">{high_risk}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Average Churn Probability</div>
                <div class="metric-value">{avg_prob:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # =====================================================
    # 2️⃣ UPLOADED DATA PREVIEW
    # =====================================================
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # =====================================================
    # 3️⃣ PREDICTION RESULTS
    # =====================================================
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.subheader("Prediction Results")
    st.dataframe(df)

    # =====================================================
    # 4️⃣ TOP CHURN DRIVING FEATURES
    # =====================================================
    st.markdown("<br><br>", unsafe_allow_html=True)
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

        bars = base.mark_bar()

        text = base.mark_text(dy=-5).encode(
            text=alt.Text("Importance:Q", format=".2f")
        )

        chart = (bars + text).properties(height=400)

        st.altair_chart(chart, use_container_width=True)

    # =====================================================
    # DOWNLOAD BUTTON
    # =====================================================
    st.markdown("<br><br>", unsafe_allow_html=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Predictions CSV",
        data=csv,
        file_name="churn_predictions.csv",
        mime="text/csv",
    )

else:
    st.info("Please upload a CSV file to start prediction.")