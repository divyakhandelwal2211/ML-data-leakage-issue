
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def remove_leakage(df):
    """
    Drop future information columns
    """
    leakage_cols = ["future_account_status", "cancellation_date"]
    return df.drop(columns=leakage_cols)


def split_features_target(df):
    """
    Split X and y
    """
    X = df.drop("churn", axis=1)
    y = df["churn"]
    return X, y


def train_test_split_data(X, y):
    """
    Perform train-test split
    """
    return train_test_split(X, y, test_size=0.2, random_state=42)


def scale_data(X_train, X_test):
    """
    Scale numeric features (important for Logistic Regression)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler