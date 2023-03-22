import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score


LABEL = "P1_PT_TYPE"


def run_elastic_net(df: pd.DataFrame, num_iters: int = 1):
    """
    Runs an elastic-net model on the input dataframe, using "P1_PT_TYPE" as the label.

    Args:
        df: The cleaned and encoded TARCC dataset.
        num_iters: The number of elastic-net iterations to perform.

    Returns:

    """

    data = df[df[LABEL] != 3]
    if "PATID" in df.columns:
        data = data.drop("PATID", axis=1)
    if "RBM_TARC_PID" in df.columns:
        data = data.drop("RBM_TARC_PID", axis=1)

    features = data.drop(LABEL, axis=1).columns

    X = data.drop(LABEL, axis=1).values
    y = data[LABEL]

    imputer = KNNImputer(keep_empty_features=True)
    X = imputer.fit_transform(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    best_cs = np.zeros(num_iters)
    best_l1_ratios = np.zeros(num_iters)
    micro_f1_scores = np.zeros(num_iters)
    feature_importances = []
    confusion_matrices = []

    for i in range(num_iters):
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

        best_cs[i] = logistic_regression_model.C_[0]
        best_l1_ratios[i] = logistic_regression_model.l1_ratio_[0]
        micro_f1_scores[i] = f1_score(y_test, predictions, average="micro")

        # Feature importance
        r = permutation_importance(
            logistic_regression_model, X_test, y_test,
            scoring="f1_micro",
            n_repeats=10,
            random_state=0
        )
        importance_indices = np.argsort(r["importances_mean"])[::-1]
        feature_importances.append(features[importance_indices])

        confusion_matrices.append(confusion_matrix(y_test, predictions))

    return best_cs, best_l1_ratios, micro_f1_scores, feature_importances, confusion_matrices
