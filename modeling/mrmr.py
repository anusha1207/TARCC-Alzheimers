import mrmr
import pandas as pd
import matplotlib.pyplot as plt

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
    score = score.reset_index()

    cdf = 0
    scores = []
    for i in range(len(features)):
        cdf += score[features[i]]
        scores.append(cdf)

    plt.plot(range(len(features)), scores)
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative Accuracy Score')
    plt.show()
