import streamlit as st
import pandas as pd
import numpy as np
from src.model_loader import load_model, load_scaler, load_feature_names, load_metrics
from src.processor import preprocess_data, get_risk_label, validate_input, analyze_data_quality
from src.agent import run_agent
from src.explainability import explain_prediction, generate_shap_plot
from src.ui_components import (
    render_kpi_cards, 
    render_churn_distribution, 
    render_feature_importance, 
    render_tenure_analysis,
    render_business_insights,
    render_model_baseline
)

# PAGE CONFIG
st.set_page_config(
    page_title="ChurnGuard AI | Enterprise Retention Dashboard",
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
st.title("ChurnGuard AI")
st.subheader("Enterprise Customer Retention and Churn Analytics")

# LOAD MODELS
model = load_model()
scaler = load_scaler()
feature_names = load_feature_names()
metrics = load_metrics()

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

    # VALIDATE INPUT
    validation_errors = validate_input(df_raw)
    if validation_errors:
        for error in validation_errors:
            st.error(error)
        st.stop()

    # DATA QUALITY REPORT
    quality = analyze_data_quality(df_raw)
    if quality["missing"] or quality["outliers"]:
        cols_missing = len(quality["missing"])
        if cols_missing:
            st.warning(
                f"{cols_missing} column(s) had missing values — imputed with median/mode. "
                f"Results may be less reliable."
            )
        if quality["outliers"]:
            out_summary = ", ".join(f"{k} ({v})" for k, v in quality["outliers"].items())
            st.info(f"Outliers detected (IQR): {out_summary}")
        with st.expander("Data quality details"):
            st.metric("Data Quality Score", f"{quality['quality_score']} / 100")
            if quality["missing"]:
                st.write("**Missing values per column:**")
                st.json(quality["missing"])
            if quality["outliers"]:
                st.write("**Outliers per column (IQR method):**")
                st.json(quality["outliers"])
    else:
        st.success(f"Data quality score: {quality['quality_score']} / 100")

    try:
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
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.stop()

    # TAB PERSISTENCE (session state + JS workaround) 
    if "active_tab_index" not in st.session_state:
        st.session_state.active_tab_index = 0
    if "agent_result" not in st.session_state:
        st.session_state.agent_result = None
    if "agent_customer_id" not in st.session_state:
        st.session_state.agent_customer_id = None

    def _restore_tab():
        """Inject minimal JS to click the correct tab button after rerun."""
        idx = st.session_state.active_tab_index
        if idx > 0:
            st.components.v1.html(
                f"""
                <script>
                const tryClick = () => {{
                    const tabs = window.parent.document.querySelectorAll(
                        '[data-baseweb="tab-list"] button[role="tab"]'
                    );
                    if (tabs.length > {idx}) {{
                        tabs[{idx}].click();
                    }} else {{
                        setTimeout(tryClick, 100);
                    }}
                }};
                tryClick();
                </script>
                """,
                height=0,
            )

    _restore_tab()

    # TABS
    tab1, tab2, tab3, tab4 = st.tabs([
        "Executive Dashboard", 
        "Risk Analysis", 
        "Model Performance", 
        "Raw Data Explorer"
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

        # AI RETENTION INSIGHTS 
        st.markdown("---")
        st.subheader("🤖 AI Retention Insights")

        # Build customer dropdown from high-risk customers (fall back to all if none)
        if "customerID" in df_results.columns:
            high_risk_ids = df_results[df_results["Risk_Level"] == "High Risk"].sort_values(
                by="Churn_Probability", ascending=False
            )["customerID"].tolist()
            all_ids = df_results["customerID"].tolist()
            dropdown_ids = high_risk_ids if high_risk_ids else all_ids

            selected_id = st.selectbox(
                "Select Customer ID",
                options=dropdown_ids,
                index=0,
                help="High-risk customers are shown first. Select one to generate AI retention insights.",
            )

            if selected_id:
                # Pin tab to Risk Analysis on any interaction
                st.session_state.active_tab_index = 1

                row = df_results[df_results["customerID"] == selected_id].iloc[0]
                churn_prob = float(row["Churn_Probability"])
                customer_data = row.drop(
                    labels=["Churn_Prediction", "Churn_Probability", "Risk_Level"],
                    errors="ignore",
                ).to_dict()

                if st.button("Generate Retention Strategy", key="run_agent_btn"):
                    with st.spinner("Agent is analyzing customer profile…"):
                        row_idx = df_results.index[df_results["customerID"] == selected_id][0]
                        customer_scaled = df_scaled[list(df_results.index).index(row_idx)]
                        try:
                            shap_top = explain_prediction(
                                model, customer_scaled, feature_names,
                                background=df_scaled[:100], top_k=5,
                            )
                        except Exception as shap_err:
                            shap_top = []
                            st.info(f"SHAP explanation unavailable: {shap_err}")
                        result = run_agent(customer_data, churn_prob, shap_explanation=shap_top)
                        st.session_state.agent_result = result
                        st.session_state.agent_customer_id = selected_id
                        st.session_state.shap_top = shap_top
                        st.session_state.shap_row = customer_scaled

                # Display cached result (persists across reruns)
                result = st.session_state.agent_result
                if result and st.session_state.agent_customer_id == selected_id:
                    # Fallback warning
                    if result.get("is_fallback"):
                        st.warning("⚠️ AI advisor unavailable, showing rule-based suggestions")

                    # 1. Risk Summary & Level
                    risk = result.get("risk_level", "Unknown")
                    if risk == "High":
                        st.error(f"Risk Level: **{risk}**")
                    elif risk == "Medium":
                        st.warning(f"Risk Level: **{risk}**")
                    else:
                        st.success(f"Risk Level: **{risk}**")
                    
                    if result.get("risk_summary"):
                        st.info(result.get("risk_summary"))

                    # 2. Contributing Factors
                    st.markdown("#### Contributing Factors")
                    for factor in result.get("contributing_factors", []):
                        st.markdown(f"- {factor}")

                    # SHAP per-customer explanation
                    shap_top = st.session_state.get("shap_top") or []
                    shap_row = st.session_state.get("shap_row")
                    if shap_top:
                        st.markdown("#### Per-customer Feature Attribution (SHAP)")
                        for f in shap_top:
                            arrow = "↑" if f["shap_value"] > 0 else "↓"
                            st.markdown(
                                f"- **{f['feature']}** {arrow} "
                                f"(shap={f['shap_value']:+.3f}, value={f['value']:.3g}) — {f['direction']}"
                            )
                        try:
                            fig = generate_shap_plot(
                                model, shap_row, feature_names,
                                background=df_scaled[:100], top_k=10,
                            )
                            st.pyplot(fig)
                        except Exception as plot_err:
                            st.caption(f"SHAP plot unavailable: {plot_err}")

                    # 3. Recommended Actions
                    st.markdown("#### Recommended Retention Strategy")
                    for idx, act in enumerate(result.get("recommended_actions", []), 1):
                        priority = act.get("priority", "")
                        color = "red" if priority == "High" else "orange" if priority == "Medium" else "green"
                        st.markdown(f"**{idx}. {act.get('action')}** (Priority: :{color}[{priority}])")
                        st.caption(f"*Rationale:* {act.get('rationale')}")

                    # 4. Supporting Insights / Disclaimer
                    with st.expander("Supporting Insights & Disclaimer"):
                        for source in result.get("sources", []):
                            st.markdown(f"- {source}")
                        
                        if result.get("disclaimers"):
                            st.markdown("---")
                            for disc in result.get("disclaimers", []):
                                st.caption(disc)
        else:
            st.info("No `customerID` column found — cannot select individual customers.")

    with tab3:
        st.header("Model Evaluation Metrics")
        render_model_baseline(metrics)

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
        - Real-time Churn Prediction using advanced ML.
        - Risk Segmentation to identify vulnerable customers.
        - Actionable Insights to boost your retention rates.
        - Feature Analysis to understand what drives customer behavior.
        """)
    with col2:
        st.image("feature_importance.png", caption="Sample Feature Importance Analysis")