import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------
# Model Training
# ---------------------------------------------------------

def train_models(X_train, X_test, y_train, y_test):

    # =====================================================
    # 1. LINEAR REGRESSION
    # =====================================================

    linear_model = LinearRegression()

    linear_model.fit(
        X_train,
        y_train
    )

    linear_predictions = linear_model.predict(
        X_test
    )

    # =====================================================
    # 2. RIDGE REGRESSION
    # =====================================================

    ridge_model = Ridge(
        alpha=1.0
    )

    ridge_model.fit(
        X_train,
        y_train
    )

    ridge_predictions = ridge_model.predict(
        X_test
    )

    # =====================================================
    # 3. LASSO REGRESSION
    # =====================================================

    lasso_model = Lasso(
        alpha=0.1,
        max_iter=10000
    )

    lasso_model.fit(
        X_train,
        y_train
    )

    lasso_predictions = lasso_model.predict(
        X_test
    )

    # =====================================================
    # METRICS FUNCTION
    # =====================================================

    def calculate_metrics(actual, predicted):

        mae = mean_absolute_error(
            actual,
            predicted
        )

        mse = mean_squared_error(
            actual,
            predicted
        )

        rmse = np.sqrt(mse)

        r2 = r2_score(
            actual,
            predicted
        )

        return {
            "MAE": round(float(mae), 4),
            "MSE": round(float(mse), 4),
            "RMSE": round(float(rmse), 4),
            "R2": round(float(r2), 4)
        }

    # =====================================================
    # MODEL METRICS
    # =====================================================

    linear_metrics = calculate_metrics(
        y_test,
        linear_predictions
    )

    ridge_metrics = calculate_metrics(
        y_test,
        ridge_predictions
    )

    lasso_metrics = calculate_metrics(
        y_test,
        lasso_predictions
    )

    # =====================================================
    # MODEL COMPARISON
    # =====================================================

    comparison = pd.DataFrame({

        "Model": [
            "Linear Regression",
            "Ridge Regression",
            "Lasso Regression"
        ],

        "MAE": [
            linear_metrics["MAE"],
            ridge_metrics["MAE"],
            lasso_metrics["MAE"]
        ],

        "MSE": [
            linear_metrics["MSE"],
            ridge_metrics["MSE"],
            lasso_metrics["MSE"]
        ],

        "RMSE": [
            linear_metrics["RMSE"],
            ridge_metrics["RMSE"],
            lasso_metrics["RMSE"]
        ],

        "R2 Score": [
            linear_metrics["R2"],
            ridge_metrics["R2"],
            lasso_metrics["R2"]
        ]
    })

    # =====================================================
    # FEATURE COEFFICIENTS
    # =====================================================

    feature_names = X_train.columns.tolist()

    coefficients = pd.DataFrame({

        "Feature": feature_names,

        "Linear Regression": linear_model.coef_,

        "Ridge Regression": ridge_model.coef_,

        "Lasso Regression": lasso_model.coef_
    })

    coefficients = coefficients.round(4)

    # =====================================================
    # FINAL RESULTS
    # =====================================================

    results = {

        "linear": linear_metrics,

        "ridge": ridge_metrics,

        "lasso": lasso_metrics,

        "comparison": comparison,

        "comparison_html": comparison.to_html(
            classes="styled-table",
            index=False
        ),

        "coefficients": coefficients,

        "coefficients_html": coefficients.to_html(
            classes="styled-table",
            index=False
        )
    }

    return results


# ---------------------------------------------------------
# Prediction Function
# ---------------------------------------------------------

def predict_ac_power(
        model,
        input_data
):

    prediction = model.predict(
        input_data
    )

    return float(prediction[0])