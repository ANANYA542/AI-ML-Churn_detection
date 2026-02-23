Data Preprocessing Summary
The dataset was cleaned by removing the customerID column, converting the TotalCharges column to numeric format, and handling missing values using median imputation. The Churn column was converted into binary format (0 for No, 1 for Yes). All categorical variables were encoded using one-hot encoding. The dataset was then split into 80% training and 20% testing sets. Feature scaling was applied using StandardScaler to normalize the data before model training.

