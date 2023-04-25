"""
Defines functions for running and evaluating random forest models.
"""
import pickle
from typing import Any

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

from utils.utils import remove_bookkeeping_features

LABEL = "P1_PT_TYPE"


def run_random_forest(df: pd.DataFrame, num_iters: int = 1, pkl: str = None) -> dict[str, Any]:
    """
    Runs a random forest model for num_iters train-test splits on the input dataframe, using "P1_PT_TYPE" as the label.
    Output the results to a pickle file if the pkl option is provided.

    Args:
        df: The cleaned and encoded TARCC dataset.
        num_iters: The number of random forest iterations to perform.
        pkl: The name of the pickle file to cache the results in.

    Returns: A dictionary of model results with the following keys and values:
        - features: A list of feature used in the random forest model.
        - models: A list containing the RandomForestClassifier object after each iteration.
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

    # Run the random forest model multiple times.
    for i in range(num_iters):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        random_forest_model = RandomForestClassifier()
        random_forest_model.fit(X_train, y_train)

        models.append(random_forest_model)
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


def evaluate_random_forest(pkl: str, verbose: bool = False) -> dict[str, list]:
    """
    Evaluates the results of the random forest models stored in the input pickle file. For each train-test split,
    this model prints the optimal hyperparameters, the micro-F1 score, the feature importances, and the confusion
    matrix if the verbose option is True. After all iterations, this function prints the mean micro-F1 score and the
    mean confusion matrix.

    Args:
        pkl: The name of the pickle file which stores the output of the random forest model to evaluate. The object
        stored in this file should be a dictionary returned by the run_random_forest function.
        verbose: If True, prints the results after each iteration; otherwise, prints only the aggregate results

    Returns:
        None
    """

    with open(pkl, "rb") as handle:
        data = pickle.load(handle)
        features = data["features"]
        models = data["models"]
        testing_data = data["testing_data"]

    micro_f1_scores = []
    confusions = []
    r = None

    for i in range(len(models)):

        random_forest_model = models[i]
        X_test, y_test = testing_data[i]

        predictions = random_forest_model.predict(X_test)
        micro_f1_score = f1_score(y_test, predictions, average="micro")
        micro_f1_scores.append(micro_f1_score)

        r = permutation_importance(
            random_forest_model, X_test, y_test,
            scoring="f1_micro",
            n_repeats=10,
            random_state=0,
            n_jobs=-1
        )

        importance_indices = np.argsort(r["importances_mean"])[::-1]
        sorted_important_features = features[importance_indices]

        confusion = confusion_matrix(y_test, predictions)
        confusions.append(confusion)

        if verbose:
            print(f"Iteration {i}")
            print(f"Micro-F1 score: {micro_f1_score}")
            print(f"Feature importances: {sorted_important_features}")
            print(f"Confusion matrix:\n{confusion}")
            print()

    print(f"Average micro-F1 score: {sum(micro_f1_scores) / len(micro_f1_scores)}")
    print(f"Average confusion matrix:\n{sum(confusions) / len(confusions)}")

    return {
        "f1": micro_f1_scores,
        "confusion": confusions,
        "importance": r,
        "features": features
    }
