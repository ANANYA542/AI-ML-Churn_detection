import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

def render_kpi_cards(df):
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = len(df)
    churn_count = (df["Churn_Prediction"] == 1).sum()
    churn_rate = (churn_count / total_customers) * 100 if total_customers > 0 else 0
    high_risk_count = (df["Risk_Level"] == "High Risk").sum()
    avg_prob = df["Churn_Probability"].mean()

    with col1:
        st.metric("Total Customers", total_customers)
    with col2:
        st.metric("Predicted Churn Rate", f"{churn_rate:.1f}%", delta=f"{churn_count} users")
    with col3:
        st.metric("High Risk Users", high_risk_count)
    with col4:
        st.metric("Avg. Churn Prob.", f"{avg_prob:.2f}")

def render_churn_distribution(df):
    st.subheader("Churn Distribution Overview")
    
    # Churn Prediction Count
    churn_counts = df["Churn_Prediction"].map({0: "Retained", 1: "Churned"}).value_counts().reset_index()
    churn_counts.columns = ["Status", "Count"]

    chart = alt.Chart(churn_counts).mark_bar().encode(
        x=alt.X("Status", sort="-y"),
        y="Count",
        color=alt.Color("Status", scale=alt.Scale(domain=["Retained", "Churned"], range=["#2ca02c", "#d62728"])),
        tooltip=["Status", "Count"]
    ).properties(height=300)
    
    st.altair_chart(chart, use_container_width=True)

def render_feature_importance(model, feature_names):
    st.subheader("Top Churn Drivers (Model Coefficients)")
    
    if hasattr(model, "coef_"):
        importance = model.coef_[0]
    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        st.warning("Model does not expose feature importance.")
        return

    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": importance
    }).sort_values(by="Impact", ascending=False)
    
    # Top 10 driving churn (positive impact) and preventing churn (negative impact)
    top_10 = pd.concat([feat_df.head(5), feat_df.tail(5)])
    
    chart = alt.Chart(top_10).mark_bar().encode(
        x="Impact",
        y=alt.Y("Feature", sort="-x"),
        color=alt.condition(
            alt.datum.Impact > 0,
            alt.value("#d62728"), # Red for churn drivers
            alt.value("#2ca02c")  # Green for retention factors
        ),
        tooltip=["Feature", "Impact"]
    ).properties(height=400)
    
    st.altair_chart(chart, use_container_width=True)

def render_tenure_analysis(df):
    st.subheader("Tenure vs Churn Risk")
    
    chart = alt.Chart(df).mark_area(opacity=0.5).encode(
        x=alt.X("tenure:Q", bin=alt.Bin(maxbins=20), title="Tenure (Months)"),
        y=alt.Y("count()", stack=None, title="Customer Count"),
        color=alt.Color("Risk_Level:N", scale=alt.Scale(domain=["High Risk", "Medium Risk", "Low Risk"], range=["#d62728", "#ff7f0e", "#2ca02c"])),
        tooltip=["Risk_Level", "count()"]
    ).properties(height=300)
    
    st.altair_chart(chart, use_container_width=True)

def render_business_insights(df, model, feature_names):
    st.header("Business Strategy and Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Key Findings")
        if hasattr(model, "coef_"):
            top_feature = _feature_names[np.argmax(model.coef_[0])]
            st.write(f"1. **Primary Churn Driver:** `{top_feature}` has the strongest positive correlation with churn.")
            
            # Month-to-month check
            if "Contract_Month-to-month" in _feature_names:
                st.write("2. **Contract Vulnerability:** Month-to-month contracts significantly increase churn risk.")
        
        high_risk_pct = (df["Risk_Level"] == "High Risk").mean() * 100
        st.write(f"3. **Urgency:** {high_risk_pct:.1f}% of uploaded customers are in the 'High Risk' category.")

    with col2:
        st.subheader("Actionable Recommendations")
        st.info("Targeted retention for High Monthly Charges: Offer loyalty discounts to customers with high spend but low tenure.")
        st.success("Contract Upselling: Transition month-to-month customers to annual contracts via bundle incentives.")
        st.warning("Fiber Optic Check: Customers with Fiber Optic service show higher churn; investigate service quality or pricing.")

def render_performance_metrics(y_true, y_pred, y_probs):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
    
    # metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc:.2%}")
    m2.metric("Precision", f"{prec:.2%}")
    m3.metric("Recall", f"{rec:.2%}")
    m4.metric("F1 Score", f"{f1:.2%}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(
            cm, 
            index=["Actual: No", "Actual: Yes"], 
            columns=["Predicted: No", "Predicted: Yes"]
        ).stack().reset_index()
        cm_df.columns = ["Actual", "Predicted", "Count"]
        
        cm_chart = alt.Chart(cm_df).mark_rect().encode(
            x="Predicted:O",
            y="Actual:O",
            color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["Actual", "Predicted", "Count"]
        ).properties(height=350)
        
        text = cm_chart.mark_text(baseline='middle').encode(
            text='Count:Q',
            color=alt.condition(
                alt.datum.Count > cm.max() / 2,
                alt.value('white'),
                alt.value('black')
            )
        )
        st.altair_chart(cm_chart + text, use_container_width=True)

    with col2:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        
        roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
        
        roc_chart = alt.Chart(roc_df).mark_line(color="#d33682").encode(
            x=alt.X("FPR", title="False Positive Rate"),
            y=alt.Y("TPR", title="True Positive Rate"),
            tooltip=["FPR", "TPR"]
        ).properties(title=f"AUC: {roc_auc:.3f}", height=350)
        
        # Diagonal line
        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[5, 5], color="gray").encode(
            x="x", y="y"
        )
        
        st.altair_chart(diag + roc_chart, use_container_width=True)
