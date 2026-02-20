# src/evaluate.py

from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy WITHOUT Leakage:", accuracy)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    return accuracy