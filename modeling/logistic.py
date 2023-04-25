"""
Defines functions for running and evaluating logistic net models with the elastic net penalty.
"""
import pickle
from typing import Any

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

from utils.utils import remove_bookkeeping_features

LABEL = "P1_PT_TYPE"


def run_logistic_regression(df: pd.DataFrame, num_iters: int = 1, pkl: str = None) -> dict[str, Any]:
    """
    Runs a logistic regression model for num_iters train-test splits on the input dataframe, using "P1_PT_TYPE" as the
    label. Output the results to a pickle file if the pkl option is provided.

    Args:
        df: The cleaned and encoded TARCC dataset.
        num_iters: The number of logistic regression iterations to perform.
        pkl: The name of the pickle file to cache the results in.

    Returns: A dictionary of model results with the following keys and values:
        - features: A list of feature used in the logistic regression model.
        - models: A list containing the LogisticRegressionCV object after each iteration.
        - training_data: A list of tuples, where each tuple is an (X, y) pair of training data.
        - testing_data: A list of tuples, where each tuple is an (X, y) pair of testing data.
    """

    # Remove bookkeeping information before modeling.
    df = remove_bookkeeping_features(df)

    # Obtain the features, the data matrix, and the label vector.
    features = df.drop(LABEL, axis=1).columns
    X = df.drop(LABEL, axis=1).values
    y = df[LABEL]

    # Impute the data using KNN imputing.
    imputer = KNNImputer(keep_empty_features=True)
    X = imputer.fit_transform(X)

    # Scale the data.
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Keep track of the models, the training data, and the testing data of each iteration.
    models = []
    training_data = []
    testing_data = []

    # Run the logistic regression model multiple times.
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

        models.append(logistic_regression_model)
        training_data.append((X_train, y_train))
        testing_data.append((X_test, y_test))

    output = {
        "features": features,
        "models": models,
        "training_data": training_data,
        "testing_data": testing_data
    }

    # Cache the return value if the pkl option has been provided.
    if pkl:
        with open(pkl, "wb") as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return output


def evaluate_logistic_regression(pkl: str, verbose: bool = False) -> dict[str, list]:
    """
    Evaluates the results of the logistic regression models stored in the input pickle file. For each train-test split,
    this model prints the optimal hyperparameters, the micro-F1 score, the feature importances, and the confusion
    matrix if the verbose option is True. After all iterations, this function prints the mean micro-F1 score and the
    mean confusion matrix.

    Args:
        pkl: The name of the pickle file which stores the output of the logistic regression model to evaluate. The
        object stored in this file should be a dictionary returned by the run_logistic_regression function.
        verbose: If True, prints the results after each iteration; otherwise, prints only the aggregate results

    Returns: A dictionary of model results with the following keys and values:
        - f1: A list of micro-F1 scores for each model.
        - confusion: A list of confusion matrices for each model.

    """

    with open(pkl, "rb") as handle:
        data = pickle.load(handle)
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

        if verbose:
            print(f"Iteration {i}")
            print(f"Best C: {best_C}")
            print(f"Best l1 ratio: {best_l1_ratio}")
            print(f"Micro-F1 score: {micro_f1_score}")
            print(f"Feature importances: {sorted_important_features}")
            print(f"Confusion matrix:\n{confusion}")
            print()

    print(f"Average micro-F1 score: {sum(micro_f1_scores) / len(micro_f1_scores)}")
    print(f"Average confusion matrix:\n{sum(confusions) / len(confusions)}")

    return {
        "f1": micro_f1_scores,
        "confusion": confusions
    }
