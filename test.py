# test.py

import joblib
import pandas as pd

def run_prediction():
    # Load model
    model = joblib.load("model.joblib")

    # New data for prediction
    test_data = pd.DataFrame({
        "age": [30, 58],
        "income": [40000, 125000]
    })

    predictions = model.predict(test_data)
    probabilities = model.predict_proba(test_data)

    for i in range(len(test_data)):
        print(f"Input: {test_data.iloc[i].to_dict()}")
        print(f"Prediction (Approved=1, Not Approved=0): {predictions[i]}")
        print(f"Confidence: {probabilities[i]}")
        print("-" * 40)

if __name__ == "__main__":
    run_prediction()
