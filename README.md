Data Leakage Detection in Customer Churn Prediction
1. Introduction

Machine Learning models sometimes show extremely high accuracy during validation but fail in real-world deployment. One major reason behind this issue is Data Leakage, particularly Future Information Leakage.

This project demonstrates how future-based features can artificially inflate model performance and how to properly identify, remove, and validate against leakage to build a production-ready model.

2. Problem Statement

The objective of this project was to build a Customer Churn Prediction Model.

However, the initial dataset contained features that were not available at prediction time:

future_account_status

cancellation_date

These columns represent post-event information. Including them during model training leads to data leakage, resulting in misleading evaluation metrics and unrealistic model performance.

3. Objective

Identify future information leakage in the dataset

Remove leakage features

Train a model using only valid historical data

Compare model performance before and after leakage removal

Build a clean, modular, industry-ready ML pipeline

4. Dataset Description

The dataset includes:

Valid Features (Available at prediction time)

monthly_usage

num_complaints

tenure_months

Leakage Features (Removed during preprocessing)

future_account_status

cancellation_date

Target Variable

churn (1 = Customer churned, 0 = Customer retained)

5. Approach
Step 1: Leakage Identification

Analyzed feature availability and detected future-based attributes.

Step 2: Leakage Removal

Removed leakage columns explicitly in preprocessing module to ensure clean training data.

Step 3: Modular Architecture

Structured the project into:

Data Loading

Preprocessing

Model Building

Evaluation

Step 4: Model Training

Used Logistic Regression as the baseline classification model.

Step 5: Evaluation

Compared model performance:

With leakage

Without leakage

6. Results
Scenario	Performance
With Leakage	Artificially High Accuracy
Without Leakage	Realistic and Reliable Accuracy

This confirms the negative impact of data leakage on model validation.

7. Project Structure
Data_Leakage_Detection_Case_Study/
│
├── data/
│   └── raw/
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
8. Tech Stack

Python

Pandas

Scikit-learn

Logistic Regression

Modular ML Pipeline Design

9. Key Learnings

Importance of feature validation before model training

Risks associated with future information leakage

Why extremely high accuracy should be critically evaluated

Importance of modular and clean ML architecture

Difference between experimental and production-ready models

10. How to Run the Project

Install dependencies:

uv add -r requirements.txt

Run training:

python src/train.py
11. Conclusion

This project highlights a critical real-world ML challenge — data leakage. By identifying and removing future-based features, the model becomes reliable and production-ready.

The focus of this project is not just building a churn model, but demonstrating strong debugging, validation, and ML engineering practices.
