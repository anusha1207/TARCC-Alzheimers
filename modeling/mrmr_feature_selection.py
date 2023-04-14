import numpy as np

import mrmr
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

from utils.utils import remove_bookkeeping_features, get_features_label


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

    # Remove bookkeeping information before feature selection.
    X = remove_bookkeeping_features(X)

    selected_features = mrmr.mrmr_classif(X=X, y=y, K=k, return_scores=scr)
    return selected_features


def plot_accuracy_with_features(X: pd.DataFrame, y: pd.Series):
    """
    Uses MRMR to plot changing accuracy with number of features over the entire feature set

    Args:
        X: The data matrix
        y: The label vector
    """
    features, score, _ = perform_mrmr(X, y, X.shape[1], True)
    score = score.sort_values(ascending=False)

    cdf = 0
    scores = []
    for feature in features:
        cdf += score[feature]
        scores.append(cdf)

    plt.plot(range(len(features)), scores)
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative Accuracy Score')
    plt.show()


def plot_cutoffs(
        blood_only: pd.DataFrame,
        clinical_only: pd.DataFrame,
        combined: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Runs MRMR feature selection on the blood-only, clinical-only, and combined dataframes. This function plots the
    cumulative relevance scores of each feature set, as well as the points at which the percent change of the scores
    dips below 0.5%.

    Args:
        blood_only: The cleaned and encoded blood-only dataset.
        clinical_only: The cleaned and encoded clinical-only dataset.
        combined: The combined dataset.

    Returns:
        A tuple of three lists, each representing the selected features of the input dataframes.
    """

    def plot_line(data: pd.DataFrame, label: str, color: str) -> pd.DataFrame:

        y, X = get_features_label(data)
        y = pd.Series(y)
        features, relevances, _ = perform_mrmr(X, y, len(X.columns), True)
        relevances /= np.sum(relevances)
        cumulative_relevances = np.cumsum(relevances.sort_values(ascending=False))
        cutoff = np.where(np.abs(np.diff(cumulative_relevances)) / cumulative_relevances[:-1] < 0.005)[0][0]

        plt.plot(range(len(features)), cumulative_relevances, label=label, c=color)
        plt.vlines(cutoff, 0, cumulative_relevances[cutoff], colors=color)

        return features[:cutoff]

    plt.figure(figsize=(8, 6))

    selected_blood_features = plot_line(blood_only, "Blood Only", "orange")
    selected_clinical_features = plot_line(clinical_only, "Clinical Only", "green")
    selected_combined_features = plot_line(combined, "Combined", "blue")

    plt.xlabel("Number of Features")
    plt.ylabel("Scaled Cumulative Accuracy Score")
    plt.legend()
    plt.show()

    return selected_blood_features, selected_clinical_features, selected_combined_features
