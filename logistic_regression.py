from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def get_metrics(y_test, prediction):

    return {
        "Accuracy": round(accuracy_score(y_test, prediction), 4),
        "Precision": round(
            precision_score(y_test, prediction, zero_division=0), 4
        ),
        "Recall": round(
            recall_score(y_test, prediction, zero_division=0), 4
        ),
        "F1 Score": round(
            f1_score(y_test, prediction, zero_division=0), 4
        )
    }


def train_logistic_regression(
        X_train,
        X_test,
        y_train,
        y_test
):

    # WITHOUT REGULARISATION
    model_without = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=50
    )

    model_without.fit(X_train, y_train)

    prediction_without = model_without.predict(X_test)

    without_results = get_metrics(
        y_test,
        prediction_without
    )


    # WITH L2 REGULARISATION
    model_l2 = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=50
    )

    model_l2.fit(X_train, y_train)

    prediction_l2 = model_l2.predict(X_test)

    l2_results = get_metrics(
        y_test,
        prediction_l2
    )


    # WITH L1 REGULARISATION
    model_l1 = LogisticRegression(
        penalty="l1",
        C=1.0,
        solver="liblinear",
        max_iter=50
    )

    model_l1.fit(X_train, y_train)

    prediction_l1 = model_l1.predict(X_test)

    l1_results = get_metrics(
        y_test,
        prediction_l1
    )


    return {
        "without": without_results,
        "l2": l2_results,
        "l1": l1_results
    }