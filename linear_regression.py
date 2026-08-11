import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


def train_linear_regression(X_train, X_test, y_train, y_test):

    # Without Regularisation
    linear_model = LinearRegression()

    linear_model.fit(X_train, y_train)

    linear_prediction = linear_model.predict(X_test)

    linear_metrics = calculate_metrics(
        y_test,
        linear_prediction
    )

    # With Regularisation - Ridge
    ridge_model = Ridge(alpha=1.0)

    ridge_model.fit(X_train, y_train)

    ridge_prediction = ridge_model.predict(X_test)

    ridge_metrics = calculate_metrics(
        y_test,
        ridge_prediction
    )

    # With Regularisation - Lasso
    lasso_model = Lasso(
        alpha=0.1,
        max_iter=10000
    )

    lasso_model.fit(X_train, y_train)

    lasso_prediction = lasso_model.predict(X_test)

    lasso_metrics = calculate_metrics(
        y_test,
        lasso_prediction
    )

    return {
        "linear": linear_metrics,
        "ridge": ridge_metrics,
        "lasso": lasso_metrics
    }