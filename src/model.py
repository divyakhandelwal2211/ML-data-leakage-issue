# src/model.py

from sklearn.linear_model import LogisticRegression

def build_model():
    """
    Create Logistic Regression model
    """
    return LogisticRegression()


def train_model(model, X_train, y_train):
    """
    Train model
    """
    model.fit(X_train, y_train)
    return model