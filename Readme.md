# Customer Churn Prediction System  
### Milestone 1 – Machine Learning Implementation

---

## Project Overview

Customer churn is a major business challenge in the telecommunications industry. Losing customers directly affects revenue, and acquiring new customers is significantly more expensive than retaining existing ones.

This project develops a supervised machine learning system to predict customer churn using historical telecom customer data. The system identifies high-risk customers, provides churn probability scores, and highlights key factors influencing churn behavior.

---

##  Objective

The main objectives of this project are:

- Build a supervised machine learning model to predict churn
- Estimate churn probability for each customer
- Identify key drivers influencing churn behavior
- Provide a simple web interface for prediction and analysis

---

##  Dataset Information

**Dataset:** Telco Customer Churn Dataset (Kaggle)  
**Total Records:** 7043 customers  
**Original Features:** 21  
**Target Variable:** `Churn` (Yes / No)

The dataset contains:

- Demographic information
- Service subscription details
- Internet and contract types
- Billing and payment methods

---

## Machine Learning Pipeline
Customer CSV
↓
Data Cleaning & Preprocessing
↓
Feature Encoding & Scaling
↓
Logistic Regression Model
↓
Churn Prediction + Probability
↓
Streamlit Web Application

---

##  Data Preprocessing

The following preprocessing steps were performed:

- Removed `customerID` column (non-predictive identifier)
- Converted `TotalCharges` to numeric format
- Handled missing values using median imputation
- Converted `Churn` into binary format (0 = No, 1 = Yes)
- Applied One-Hot Encoding to categorical features
- Split dataset into 80% training and 20% testing sets
- Applied StandardScaler to normalize numerical features

These steps ensured clean, structured, and model-ready data.

---

##  Model Used

### Logistic Regression

Logistic Regression was selected because:

- It is suitable for binary classification
- It provides probability-based outputs
- It offers interpretability through feature coefficients
- It serves as a strong and efficient baseline model

---

##  Model Performance

| Metric        | Score   |
|--------------|---------|
| Accuracy     | 81.97%  |
| Precision    | 68.31%  |
| Recall       | 59.52%  |
| F1-Score     | 63.61%  |

 **Important:**  
Recall is particularly important in churn prediction because failing to identify churn customers can lead to revenue loss.

---

##  Key Insights (Feature Importance)

Feature importance analysis identified the following major churn drivers:

- **Tenure** – New customers are significantly more likely to churn.
- **Monthly Charges** – Higher monthly costs increase churn probability.
- **Total Charges** – High-paying customers show greater churn risk.
- **Fiber Optic Internet** – Users show relatively higher churn.
- **Two-Year Contracts** – Long-term contracts reduce churn likelihood.

---

## Business Recommendations

Based on model insights:

- Focus on onboarding and engagement during the first few months.
- Offer incentives to convert month-to-month customers to long-term contracts.
- Review pricing and quality of Fiber Optic services.
- Provide bundled service discounts for high-paying customers.

---

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn
- Streamlit

---

## Running the Project Locally

### 1️Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction