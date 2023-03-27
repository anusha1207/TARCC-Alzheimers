import mrmr
import mrmr
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

def perform_mrmr(X: pd.DataFrame, y: pd.Series, k: int, scr):
    selected_features = mrmr.mrmr_classif(X=X, y=y, K=k, return_scores= scr)
    return selected_features

def test_mrmr_model(df: pd.DataFrame):
    """
    Runs an elastic-net model on the input dataframe, using "P1_PT_TYPE" as the label.

    Args:
        df: The cleaned and encoded TARCC dataset.

    Returns:

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
