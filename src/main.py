# src/train.py

from data_load import load_data
from preprocess import (
    remove_leakage,
    split_features_target,
    train_test_split_data,
    scale_data
)
from model import build_model, train_model
from evaluate import evaluate_model


def main():

    # Step 1: Load Data
    df = load_data("data/customer_churn_realistic_with_leakage.csv")

    # Step 2: Remove Leakage
    df = remove_leakage(df)

    # Step 3: Split Features & Target
    X, y = split_features_target(df)

    # Step 4: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # Step 5: Scaling
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    # Step 6: Build & Train Model
    model = build_model()
    model = train_model(model, X_train_scaled, y_train)

    # Step 7: Evaluate
    evaluate_model(model, X_test_scaled, y_test)


if __name__ == "__main__":
    main()