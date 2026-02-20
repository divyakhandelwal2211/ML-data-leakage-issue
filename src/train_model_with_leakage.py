# =====================================
# LOGISTIC REGRESSION (WITH LEAKAGE)
# =====================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("data/customer_churn_realistic_with_leakage.csv")

# Encode categorical columns
le = LabelEncoder()
df["future_account_status"] = le.fit_transform(df["future_account_status"])

# Convert date to numeric
df["cancellation_date"] = df["cancellation_date"].astype("category").cat.codes

# Split features & target
X = df.drop("churn", axis=1)
y = df["churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling (Important for Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("Accuracy WITH Leakage:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))