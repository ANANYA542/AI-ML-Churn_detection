import streamlit as st
import pandas as pd
import numpy as np
from src.model_loader import load_model, load_scaler, load_feature_names
from src.processor import preprocess_data, get_risk_label
from src.ui_components import (
    render_kpi_cards, 
    render_churn_distribution, 
    render_feature_importance, 
    render_tenure_analysis,
    render_business_insights
)

# PAGE CONFIG
st.set_page_config(
    page_title="ChurnGuard AI | Enterprise Retention Dashboard",
    page_icon="📊",
    layout="wide"
)

# STYLING
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# SIDEBAR / HEADER
st.title("📊 ChurnGuard AI")
st.subheader("Enterprise Customer Retention & Churn Analytics")

# LOAD MODELS
model = load_model()
scaler = load_scaler()
feature_names = load_feature_names()

# FILE UPLOAD
with st.sidebar:
    st.header("Data Source")
    uploaded_file = st.file_uploader("Upload Customer CSV", type=["csv"])
    
    if uploaded_file:
        st.success("File uploaded successfully!")
    else:
        st.info("Upload a CSV file (e.g., Telco Churn format) to begin analysis.")

if uploaded_file is not None:
    # DATA PROCESSING
    df_raw = pd.read_csv(uploaded_file)
    
    with st.spinner("Processing data & generating predictions..."):
        # Preprocess
        df_processed = preprocess_data(df_raw, feature_names)
        
        # Scale
        df_scaled = scaler.transform(df_processed)
        
        # Predict
        predictions = model.predict(df_scaled)
        probabilities = model.predict_proba(df_scaled)[:, 1]
        
        # Enriched Main Dataframe
        df_results = df_raw.copy()
        df_results["Churn_Prediction"] = predictions
        df_results["Churn_Probability"] = probabilities
        df_results["Risk_Level"] = df_results["Churn_Probability"].apply(get_risk_label)

    # TABS
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Executive Dashboard", 
        "🔍 Risk Analysis", 
        "⚙️ Model Performance", 
        "📄 Raw Data Explorer"
    ])

    with tab1:
        st.markdown("### Executive Summary")
        render_kpi_cards(df_results)
        
        col1, col2 = st.columns(2)
        with col1:
            render_churn_distribution(df_results)
        with col2:
            render_tenure_analysis(df_results)
            
        render_business_insights(df_results, model, feature_names)

    with tab2:
        st.header("Risk Segmentation & Drivers")
        col_a, col_b = st.columns([1, 1])
        
        with col_a:
            render_feature_importance(model, feature_names)
        
        with col_b:
            st.subheader("High Risk Customer Profiles")
            high_risk_df = df_results[df_results["Risk_Level"] == "High Risk"].sort_values(by="Churn_Probability", ascending=False)
            st.dataframe(high_risk_df.head(10), use_container_width=True)
            st.caption("Showing top 10 most vulnerable customers.")

    with tab3:
        st.header("Model Evaluation Metrics")
        st.info("Validation metrics based on the current uploaded dataset.")
        
        # If 'Churn' column exists in upload, we can show performance
        target_col = "Churn" if "Churn" in df_results.columns else None
        if target_col:
            from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
            
            y_true = df_results[target_col].map({"Yes": 1, "No": 0, 1: 1, 0: 0})
            y_pred = df_results["Churn_Prediction"]
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Model Accuracy", f"{accuracy_score(y_true, y_pred):.2%}")
            
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            cm_df = pd.DataFrame(cm, index=["Actual No", "Actual Yes"], columns=["Predicted No", "Predicted Yes"])
            st.table(cm_df)
            
            st.subheader("Detailed Classification Report")
            st.text(classification_report(y_true, y_pred))
        else:
            st.warning("Upload data with a 'Churn' column to see model performance metrics.")

    with tab4:
        st.header("Dataset Explorer")
        st.dataframe(df_results, use_container_width=True)
        
        csv = df_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Full Prediction Results",
            data=csv,
            file_name="churn_predictions_export.csv",
            mime="text/csv",
        )

else:
    # LANDING STATE
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Welcome to ChurnGuard AI
        Upload your customer dataset to unlock:
        - **Real-time Churn Prediction** using advanced ML.
        - **Risk Segmentation** to identify vulnerable customers.
        - **Actionable Insights** to boost your retention rates.
        - **Feature Analysis** to understand what drives customer behavior.
        """)
    with col2:
        st.image("feature_importance.png", caption="Sample Feature Importance Analysis")