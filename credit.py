

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv(r"C:\Users\sarthak mohapatra\Downloads\archive\creditcard.csv")

# Separate features and target
X = df.drop(columns=["Class", "Time"])  # Drop 'Time' as it's usually irrelevant
y = df["Class"]  # Target variable

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 1: Train Isolation Forest (Unsupervised)
isolation_forest = IsolationForest(n_estimators=100, contamination=0.002, random_state=42)
isolation_forest.fit(X_scaled)

# Generate anomaly scores (lower scores indicate anomalies)
anomaly_scores = isolation_forest.decision_function(X_scaled)
print(anomaly_scores)
# Add anomaly scores as a feature
X_with_anomaly = np.hstack((X_scaled, anomaly_scores.reshape(-1, 1)))

# Step 2: Handle class imbalance using SMOTE (Supervised)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_with_anomaly, y)

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Step 4: Train Random Forest Classifier (Supervised)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = clf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# # Step 6: Real-Time Fraud Detection Function
# def detect_fraud(transaction):
#     """
#     Function to detect fraud in real-time.

#     Parameters:
#         transaction (list): A new transaction's features (unscaled).

#     Returns:
#         str: 'Fraud Detected' or 'Transaction Approved'
#         float: Anomaly score from Isolation Forest
#     """
#     # Preprocess the transaction
#     transaction_scaled = scaler.transform([transaction])  # Scale the features
    
#     anomaly_score = isolation_forest.decision_function(transaction_scaled)
    
#     transaction_with_anomaly = np.hstack((transaction_scaled, anomaly_score))
   
#     fraud_prediction = clf.predict(transaction_with_anomaly.reshape(1, -1))
    
#     if fraud_prediction == 1:
#         return "Fraud Detected", anomaly_score
#     else:
#         return "Transaction Approved", anomaly_score

# # Example of real-time fraud detection
# example_transaction = X.iloc[0].tolist() 
# result, anomaly_score = detect_fraud(example_transaction)
# print(f"Result: {result}, Anomaly Score: {anomaly_score}")
