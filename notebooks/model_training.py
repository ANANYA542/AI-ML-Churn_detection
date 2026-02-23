import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Reconstruct feature names since scaler converted X_train to a numpy array
print("Reconstructing feature names...")
df = pd.read_csv("telco.csv")
df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df = pd.get_dummies(df, drop_first=True)
X = df.drop("Churn", axis=1)
feature_names = X.columns

print("Loading preprocessed data...")
with open("processed_data.pkl", "rb") as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# Train the model
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("\n--- Model Evaluation (Logistic Regression) ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n--- Business Insight on Model Performance ---")
print(f"The model's Recall is {recall_score(y_test, y_pred):.2f}. In churn prediction, Recall is very important")
print("because it measures the percentage of actual churners we successfully identified. Missing a churn customer is costly.")

# Save the model and feature names
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)
print("\nModel and feature names saved respectively to 'model.pkl' and 'feature_names.pkl'")

