import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDRegressor
from mlflow.models import infer_signature
import joblib
from pathlib import Path

if __name__ == "__main__":
    df = pd.read_csv("data/diamonds_features.csv")

    X = df.drop(columns=['price'])
    y = df['price']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    params = {
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
        'l1_ratio': [0.001, 0.05, 0.01, 0.2],
        "penalty": ["l1", "l2", "elasticnet"],
        "loss": ['squared_error', 'huber', 'epsilon_insensitive'],
        "fit_intercept": [False, True],
    }

    mlflow.set_experiment("diamonds_price_prediction")

    with mlflow.start_run():
        lr = SGDRegressor(random_state=42)
        clf = GridSearchCV(lr, params, cv=3, n_jobs=4)
        clf.fit(X_train, y_train)
        best = clf.best_estimator_

        mlflow.log_params(best.get_params())

        signature = infer_signature(X_train, best.predict(X_train))
        mlflow.sklearn.log_model(
            best,
            "model",
            signature=signature,
            registered_model_name="DiamondsPricePredictor"
        )

        Path("models").mkdir(exist_ok=True)
        joblib.dump(best, "models/model.joblib")