🚨 Data Leakage Detection in Customer Churn Prediction

⚠️ A practical Machine Learning case study demonstrating how future information leakage can artificially inflate model performance — and how to properly detect and eliminate it.

📌 Overview

Machine Learning models sometimes show extremely high accuracy during validation but fail in real-world deployment.

One major reason behind this issue is:

Data Leakage (especially Future Information Leakage)

This project demonstrates:

🔍 How leakage occurs

📈 Why it inflates model performance

🛠 How to detect it

✅ How to remove it properly

🚀 How to build a production-ready ML pipeline

🎯 Problem Statement

Build a Customer Churn Prediction Model.

However, the initial dataset contained future-based features:

future_account_status

cancellation_date

These features are not available at prediction time.

Including them during model training leads to:

❌ Artificially high accuracy

❌ Misleading evaluation metrics

❌ Non-deployable model

🎯 Objective

Identify future information leakage

Remove leakage features

Retrain the model correctly

Compare performance before & after leakage removal

Build a clean modular ML project

📊 Dataset Description
✅ Valid Features (Available at Prediction Time)

monthly_usage

num_complaints

tenure_months

🚫 Leakage Features (Removed During Preprocessing)

future_account_status

cancellation_date

🎯 Target

churn (1 = Churned, 0 = Retained)

🛠 Approach
1️⃣ Leakage Identification

Analyzed feature availability and detected post-event information.

2️⃣ Leakage Removal

Explicitly dropped leakage columns in preprocessing.

3️⃣ Modular Pipeline Design

Separated logic into:

Data loading

Preprocessing

Model building

Evaluation

4️⃣ Model Used

📌 Logistic Regression

📌 Feature Scaling (StandardScaler)

📈 Results Comparison
Scenario	Accuracy
With Leakage	🔥 Artificially High
Without Leakage	✅ Realistic & Reliable

This confirms the importance of proper feature validation before model training.

📂 Project Structure
Data_Leakage_Detection_Case_Study/
│
├── data/
│       └── customer_churn_realistic_with_leakage.csv
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── evaluate.py
│   └── train.py
│
├── requirements.txt
└── README.md

✔ Clean separation of concerns
✔ Easy to extend
✔ Industry-style structure

🧰 Tech Stack

🐍 Python

📊 Pandas

🤖 Scikit-Learn

📈 Logistic Regression

🏗 Modular ML Architecture

🚀 How to Run
1️⃣ Install Dependencies
uv add -r requirements.txt
2️⃣ Run Training
python src/train.py
🧠 Key Learnings

🚨 Extremely high accuracy should raise suspicion

📌 Future information leakage is a serious ML issue

🧩 Proper preprocessing is critical

🏗 Modular coding improves maintainability

🎯 Production-ready models require realistic validation

