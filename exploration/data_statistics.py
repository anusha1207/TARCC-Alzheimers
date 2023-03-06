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
        df: The dataframe representing the TARCC dataset.
        png: The title of the png file (or None to not output a png).

    Returns:
        None
    """

    # Count the number of unique patients who did and did not draw blood.
    blood_data = df[["PATID", "RBM_Rule_Based_Medicine"]]
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
        df: The dataframe representing the TARCC dataset.
        png: The title of the png file (or None to not output a png).

    Returns:
    """
    classes = df["P1_PT_TYPE"].value_counts()
    classes.index = ["Control", "AD", "MCI", "Other"]
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
        df: The dataframe representing the TARCC dataset.
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











# TODO: Delete
# OLD CODE #

# df = get_cleaned_data()
#
# # Take the standard deviation of each feature, grouping by patient ID.
# grouped_sigmas = df.groupby("PATID").agg(suppressed_nanstd)
# sigma_means = grouped_sigmas.agg(suppressed_nanmean)
# sigma_standard_deviations = grouped_sigmas.agg(suppressed_nanstd)
# high_variance_features = sigma_means[np.logical_not(np.logical_or(np.isnan(sigma_means), sigma_means < 0.5))].index
#
# with open("high_variance_features.log", "w") as f:
#     for high_variance_feature in high_variance_features:
#         f.write(f"{high_variance_feature}\t{sigma_means[high_variance_feature]}\n")

# # Plot the error bars of features which have "significant" variance.
# plt.rcParams["figure.figsize"] = (100, 10)
# plt.errorbar(
#     high_variance_features,
#     sigma_means[high_variance_features],
#     sigma_standard_deviations[high_variance_features],
#     linestyle="None",
#     marker="o",
# )
# plt.yscale("log")
# plt.xticks(rotation=90)
# plt.title("Patient-wise feature means and standard deviations (log scale)")
# plt.xlabel("Feature Name")
# plt.ylabel("Patient-Wise Mean and SD")
# plt.show()
#
# # Plot the error bars of the 10 features with the most variance.
# plt.rcParams["figure.figsize"] = (12, 4)
# top_10_errors_indices = np.argsort(-sigma_means.values)[:10]
# top_10_highest_variance_features = sigma_means[top_10_errors_indices].index
# plt.errorbar(
#     [
#         "Anti-dementia Drug Hx A: Strength",
#         "Prescription A: Strength",
#         "Vitamin E Hx A: Strength",
#         "Anti-dementia Drug Hx B: Strength",
#         "Anti-dementia Drug Hx C: Strength",
#         "Anti-dementia Drug Hx D: Strength",
#         "Systemic Steroids Hx A: Strength"
#     ],
#     # top_10_highest_variance_features,
#     sigma_means[top_10_highest_variance_features][:7],
#     sigma_standard_deviations[top_10_highest_variance_features][:7],
#     linestyle="None",
#     marker="o"
# )
# plt.yscale("log")
# plt.xticks(rotation=20)
# plt.title("Top 7 patient-wise feature means and standard deviations (log scale)")
# plt.xlabel("Feature Name")
# plt.ylabel("Patient-Wise Mean and SD")
# plt.subplots_adjust(bottom=0.35)
# plt.show()
#
# # Count the proportion of missing values for each feature.
# plt.rcParams["figure.figsize"] = (12, 4)
# proportions_missing = df.agg(count_proportion_missing)
# sorted_indices = np.argsort(proportions_missing)[::-1]
# plt.bar(df.columns[sorted_indices], proportions_missing[sorted_indices])
# plt.title("Proportion of missing values by feature")
# plt.xlabel("Feature Name")
# plt.ylabel("Proportion of Missing Values")
# plt.show()


# temp = get_cleaned_data()
# features = ["A1_RACE", "C1_WAIS3_DIGTOT", "B1_BMI", "RBM_Insulin"]
# # temp = temp[["PATID"] + features]
#
# # Normalize each of these features
# for f in features:
#     temp[f] = (temp[f] - np.nanmean(temp[f])) / np.nanstd(temp[f])
#
# # Compute patient-wise variances
# grouped_sigmas = temp.groupby("PATID").agg(suppressed_nanstd)
# sigma_means = grouped_sigmas.agg(suppressed_nanmean)
# sigma_standard_deviations = grouped_sigmas.agg(suppressed_nanstd)
#
# to_plot = {f: grouped_sigmas[f][~np.isnan(grouped_sigmas[f])] for f in features}
#
# plt.boxplot(to_plot.values(), showfliers=False)
# plt.xticks([1, 2, 3, 4], ["Race", "WAIS3 Digits Score", "BMI", "Insulin"])
# plt.title("Boxplots of patient-wise standard deviations for 4 features")
# plt.ylabel("Patient-wise SD")
# plt.show(dpi=2000)
