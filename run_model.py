# run_model.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

def train_model():
    # Sample dataset
    data = {
        "age": [22, 25, 47, 52, 46, 56, 55, 60],
        "income": [25000, 32000, 80000, 110000, 98000, 120000, 115000, 130000],
        "approved": [0, 0, 1, 1, 1, 1, 1, 1]
    }

    df = pd.DataFrame(data)

    X = df[["age", "income"]]
    y = df["approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "model.joblib")
    print("✅ Model trained and saved as model.joblib")

if __name__ == "__main__":
    train_model()
