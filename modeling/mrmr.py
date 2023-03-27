import mrmr
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def perform_mrmr(X: pd.DataFrame, y: pd.Series, k: int, scr):
    """
    Performs MRMR on the input data, keeping k features.

    Args:
        X: The data matrix
        y: The label vector
        k: The number of features to keep
        scr: Whether to return scores

    Returns:
        The k selected features
    """
    selected_features = mrmr.mrmr_classif(X=X, y=y, K=k, return_scores=scr)
    print(selected_features)
    return selected_features

def test_mrmr_model(df: pd.DataFrame):
    """
    Runs an elastic-net model on the input dataframe, using "P1_PT_TYPE" as the label.

    Args:
        df: The cleaned and encoded TARCC dataset.

    Returns:
        None
    """
    LABEL = 'P1_PT_TYPE'
    X = df.drop(LABEL, axis=1).values
    y = df[LABEL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    logistic_regression_model = LogisticRegressionCV(
        penalty="elasticnet",
        Cs=[0.001, 0.01, 0.1, 1, 10, 100, 1000],
        l1_ratios=[0, 0.2, 0.4, 0.6, 0.8, 1],
        solver="saga",
        n_jobs=-1
    )
    logistic_regression_model.fit(X_train, y_train)

    predictions = logistic_regression_model.predict(X_test)

    return f1_score(y_test, predictions, average="micro")
