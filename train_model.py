from os import name
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
import joblib


# Убрал scale, так как уже делал в предыдущем шаге
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    df = pd.read_csv("df_clear.csv")

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

    # Настройка эксперимента MLflow
    mlflow.set_experiment("diamonds_price_prediction_clean")

    with mlflow.start_run():
        # Инициализация и обучение модели
        lr = SGDRegressor(random_state=42)
        clf = GridSearchCV(lr, params, cv=3, n_jobs=4)
        clf.fit(X_train, y_train)
        best = clf.best_estimator_

        # Предсказание и оценка
        y_pred = best.predict(X_val)
        rmse, mae, r2 = eval_metrics(y_val, y_pred)

        # Логирование параметров
        mlflow.log_params(best.get_params())

        # Логирование метрик
        mlflow.log_metrics({
            "rmse": rmse,
            "r2": r2,
            "mae": mae
        })

        signature = infer_signature(X_train, best.predict(X_train))
        mlflow.sklearn.log_model(
            best,
            "model",
            signature=signature,
            registered_model_name="DiamondsPricePredictorClean"
        )

        joblib.dump(best, 'diamonds_model_clean.pkl')

        print(f"R2: {r2:.4f}")

    df_runs = mlflow.search_runs()
    best_run = df_runs.sort_values("metrics.r2", ascending=False).iloc[0]
    print(f"\nЛучший запуск ID: {best_run.run_id}")
    print(f"Лучший R2 score: {best_run['metrics.r2']:.4f}")
