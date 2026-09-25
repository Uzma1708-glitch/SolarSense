import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)

    return {
        "MAE": round(float(mae), 4),
        "MSE": round(float(mse), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4)
    }


def train_single_tree_model(model_name, X_train, X_test, y_train, y_test):

    if model_name == "decision-tree":

        model = DecisionTreeRegressor(
            max_depth=10,
            random_state=42
        )

    elif model_name == "random-forest":

        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

    elif model_name == "adaboost":

        model = AdaBoostRegressor(
            n_estimators=50,
            learning_rate=0.1,
            random_state=42
        )

    elif model_name == "gradient-boosting":

        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=42
        )

    elif model_name == "xgboost":

        model = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror"
        )

    elif model_name == "lightgbm":

        model = LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )

    else:
        raise ValueError("Unknown model")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = calculate_metrics(
        y_test,
        predictions
    )

    return metrics