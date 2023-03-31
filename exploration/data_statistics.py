"""
Functions for computing statistics on the cleaned dataset.
"""
from typing import List

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


def suppressed_nanmean(x: np.ndarray) -> float:
    """
    Performs the same operation as np.nanmean but suppresses warnings.

    Args:
        x: The array to take the mean over.

    Returns:
        None.
    """
    if np.all(np.isnan(x)):
        return np.nan
    return float(np.nanmean(x))


def suppressed_nanstd(x: np.ndarray) -> float:
    """
    Performs the same operation as np.nanstd but suppresses warnings.

    Args:
        x: The array to take the standard deviation over.

    Returns:
        None.
    """
    if np.all(np.isnan(x)):
        return np.nan
    return float(np.nanstd(x))


def count_proportion_missing(x: np.ndarray) -> float:
    """
    Counts the proportion of NaN values in an array.

    Args:
        x: The array over which to count NaN values.

    Returns:
        The proportion of NaN values in an array.
    """
    return np.count_nonzero(np.isnan(x)) / x.size


def plot_blood_draw_statistics(df: pd.DataFrame, png: str = None) -> None:
    """
    Plots the proportions of patients who have and have not drawn blood in any of their visits.

    Args:
        df: The cleaned and encoded dataframe representing the TARCC dataset.
        png: The title of the png file (or None to not output a png).

    Returns:
        None
    """

    # Count the number of unique patients who did and did not draw blood.
    blood_data = df[["PATID", "RBM_Rule_Based_Medicine_1.0"]]
    blood_draws = blood_data.groupby("PATID").agg(np.nansum).value_counts()
    drew_blood = sum(blood_draws) - blood_draws[0]
    did_not_draw_blood = blood_draws[0]

    plt.rcParams["figure.figsize"] = (6, 8)
    plt.bar(["Drew blood", "Did not draw blood"], [drew_blood, did_not_draw_blood], color=["red", "blue"])
    plt.title("Number of patients who drew blood / did not draw blood")
    plt.ylabel("Number of patients (duplicates not counted)")
    if png:
        plt.savefig(f"{png}.png", dpi=100)
    plt.show()


def plot_labels_pie_chart(df: pd.DataFrame, png: str = None) -> None:
    """
    Plots a pie chart of the labels (Control, AD, MCI and Other).

    Args:
        df: The cleaned and encoded dataframe representing the TARCC dataset.
        png: The title of the png file (or None to not output a png).

    Returns:
    """
    classes = df["P1_PT_TYPE"].value_counts()
    classes.index = ["Control", "AD", "MCI"]
    classes.plot(kind="pie")
    plt.legend(labels=classes.index)
    if png:
        plt.savefig(f"{png}.png", dpi=100)
    plt.show()


def plot_patientwise_errors(
        df: pd.DataFrame,
        features: List[str],
        x_labels: List[str],
        png: str = None
) -> None:
    """
    Plots box plots of the patient-wise standard deviations for the provided features.

    Args:
        df: The cleaned and encoded dataframe representing the TARCC dataset.
        features: The subset of features to plot.
        x_labels: The x-axis labels of each of the features.
        png: The title of the png file (or None to not output a png).

    Returns:
        None
    """

    df_copy = df.copy()
    df_copy = df_copy[features + ["PATID"]]

    for feature in features:
        df_copy[feature] = (df_copy[feature] - np.nanmean(df_copy[feature])) / np.nanstd(df_copy[feature])

    # Compute patient-wise variances
    grouped_sigmas = df_copy.groupby("PATID").agg(suppressed_nanstd)

    to_plot = {feature: grouped_sigmas[feature][~np.isnan(grouped_sigmas[feature])] for feature in features}
    plt.boxplot(to_plot.values(), showfliers=False)
    plt.xticks(range(1, len(features) + 1), x_labels)
    plt.title(f"Box plots of patient-wise standard deviations for {len(features)} features")
    plt.ylabel("Patient-wise SD")
    if png:
        plt.savefig(f"{png}.png", dpi=100)
    plt.show()
