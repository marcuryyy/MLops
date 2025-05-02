import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import mlflow


def evaluate(model_path, data_path):
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)

    X = df.drop(columns=['price'])
    y = df['price']


    y_pred = model.predict(X)
    metrics = {
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        "mae": mean_absolute_error(y, y_pred),
        "r2": r2_score(y, y_pred)
    }

    with mlflow.start_run():
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(data_path)

    return metrics


if __name__ == "__main__":
    Path("reports").mkdir(exist_ok=True)

    metrics = evaluate("models/model.joblib", "data/diamonds_features.csv")

    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f)

    print("Metrics saved:", metrics)