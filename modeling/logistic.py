"""
Defines functions for running and evaluating logistic net models with the elastic net penalty.
"""

import pickle as pk

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score


LABEL = "P1_PT_TYPE"


def run_elastic_net(df: pd.DataFrame, num_iters: int = 1, pickle: str = None):
    """
    Runs an elastic-net model on the input dataframe, using "P1_PT_TYPE" as the label.

    Args:
        df: The cleaned and encoded TARCC dataset.
        num_iters: The number of elastic-net iterations to perform.

    Returns:
        A tuple containing the following:
            best_cs: the best C values in predicting AD in each iteration.
            best_l1_ratios: the best l1 ratios in predicting AD in each iteration.
            micro_f1_scores: the micro-f1 score of each iteration
            feature_importances: the sklearn feature_importances (means) of each iteration
            confusion_matrices: the confusion matrix of each iteration
    """

    data = df
    if "PATID" in df.columns:
        data = data.drop("PATID", axis=1)
    if "RBM_TARC_PID" in df.columns:
        data = data.drop("RBM_TARC_PID", axis=1)
    if "STUDYID" in df.columns:
        data = data.drop("STUDYID", axis=1)
    if "VISIT" in df.columns:
        data = data.drop("VISIT", axis=1)

    features = data.drop(LABEL, axis=1).columns
    X = data.drop(LABEL, axis=1).values
    y = data[LABEL]

    imputer = KNNImputer(keep_empty_features=True)
    X = imputer.fit_transform(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    training_data = []
    testing_data = []
    models = []

    for i in range(num_iters):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        training_data.append((X_train, y_train))
        testing_data.append((X_test, y_test))

        logistic_regression_model = LogisticRegressionCV(
            penalty="elasticnet",
            Cs=[0.001, 0.01, 0.1, 1, 10, 100, 1000],
            l1_ratios=[0, 0.2, 0.4, 0.6, 0.8, 1],
            solver="saga",
            n_jobs=-1
        )

        logistic_regression_model.fit(X_train, y_train)

        # predictions = logistic_regression_model.predict(X_test)
        #
        # best_cs[i] = logistic_regression_model.C_[0]
        # best_l1_ratios[i] = logistic_regression_model.l1_ratio_[0]
        # micro_f1_scores[i] = f1_score(y_test, predictions, average="micro")
        #
        # # Feature importance
        # r = permutation_importance(
        #     logistic_regression_model, X_test, y_test,
        #     scoring="f1_micro",
        #     n_repeats=10,
        #     random_state=0
        # )
        # importance_indices = np.argsort(r["importances_mean"])[::-1]
        # feature_importances.append(features[importance_indices])
        #
        # confusion_matrices.append(confusion_matrix(y_test, predictions))

        models.append(logistic_regression_model)

    if pickle:
        with open(f"{pickle}.pickle", "wb") as handle:
            pk.dump(
                {
                    "features": features,
                    "models": models,
                    "training_data": training_data,
                    "testing_data": testing_data
                },
                handle,
                protocol=pk.HIGHEST_PROTOCOL
            )

    return {
        "features": features,
        "models": models,
        "training_data": training_data,
        "testing_data": testing_data
    }


def evaluate_results(pickle: str):

    with open(f"{pickle}.pickle", "rb") as handle:
        data = pk.load(handle)
        features = data["features"]
        models = data["models"]
        testing_data = data["testing_data"]

    micro_f1_scores = []
    confusions = []

    for i in range(len(models)):

        logistic_regression_model = models[i]
        X_test, y_test = testing_data[i]

        predictions = logistic_regression_model.predict(X_test)

        best_C = logistic_regression_model.C_[0]
        best_l1_ratio = logistic_regression_model.l1_ratio_[0]
        micro_f1_score = f1_score(y_test, predictions, average="micro")
        micro_f1_scores.append(micro_f1_score)

        r = permutation_importance(
            logistic_regression_model, X_test, y_test,
            scoring="f1_micro",
            n_repeats=10,
            random_state=0
        )
        importance_indices = np.argsort(r["importances_mean"])[::-1]
        sorted_important_features = features[importance_indices]

        confusion = confusion_matrix(y_test, predictions)
        confusions.append(confusion)

        print(f"Iteration {i}")
        print(f"Best C: {best_C}")
        print(f"Best l1 ratio: {best_l1_ratio}")
        print(f"Micro-F1 score: {micro_f1_score}")
        print(f"Feature importances: {sorted_important_features}")
        print(f"Confusion matrix:\n{confusion}")
        print()

    print(f"Average micro-F1 score: {sum(micro_f1_scores) / len(micro_f1_scores)}")
    print(f"Average confusion matrix:\n{sum(confusions) / len(confusions)}")
