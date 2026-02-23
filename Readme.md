Customer Churn Prediction – Milestone 1
Project Overview

This project builds a machine learning system to predict customer churn using historical telecom customer data. The system identifies customers at risk of leaving and provides churn probability scores.

Objective

To develop a supervised machine learning model that predicts churn and identifies key contributing factors.

Dataset

Telco Customer Churn Dataset

7043 customer records

Target variable: Churn

Features Implemented

Data preprocessing and cleaning

Feature encoding and scaling

Logistic Regression model

Churn probability prediction

Feature importance analysis

Streamlit UI for predictions

📈 Model Performance

(Insert accuracy, precision, recall after model finishes)

🛠 Tech Stack

Python

pandas

NumPy

scikit-learn

Streamlit

🚀 How to Run
pip install -r requirements.txt
streamlit run app.py


Data Preprocessing Summary
The dataset was cleaned by removing the customerID column, converting the TotalCharges column to numeric format, and handling missing values using median imputation. The Churn column was converted into binary format (0 for No, 1 for Yes). All categorical variables were encoded using one-hot encoding. The dataset was then split into 80% training and 20% testing sets. Feature scaling was applied using StandardScaler to normalize the data before model training.


