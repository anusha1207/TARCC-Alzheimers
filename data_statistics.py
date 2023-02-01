"""
Computes statistics for the cleaned data.
"""

from matplotlib import pyplot as plt
import numpy as np

from data_cleaning import get_cleaned_data


def suppressed_nanmean(x):
    """
    Same as np.nanmean, but with suppressed warnings.
    """
    if np.all(np.isnan(x)):
        return np.nan
    return np.nanmean(x)


def suppressed_nanstd(x):
    """
    Same as np.nanstd, but with suppressed warnings.
    """
    if np.all(np.isnan(x)):
        return np.nan
    return np.nanstd(x)


def count_proportion_missing(x):
    """
    Returns the proportion of NaN values in an array.
    """
    return np.count_nonzero(np.isnan(x)) / x.size


df = get_cleaned_data()

features = df.columns.drop("PATID")

# Take the standard deviation of each feature, grouping by patient ID.
grouped_sigmas = df.groupby("PATID").agg(lambda x: suppressed_nanstd(x))
normalized_grouped_sigmas = (grouped_sigmas - grouped_sigmas.mean()) / grouped_sigmas.std()
sigma_means = normalized_grouped_sigmas.agg(lambda x: suppressed_nanmean(x))
sigma_standard_deviations = normalized_grouped_sigmas.agg(lambda x: suppressed_nanstd(x))
features_with_high_variance = sigma_standard_deviations[sigma_standard_deviations != 0]


# Count the proportion of missing values for each feature, grouping by patient ID.
grouped_proportions_missing = df.groupby("PATID").agg(lambda x: count_proportion_missing(x))
proportions_missing_means = grouped_proportions_missing.agg(lambda x: suppressed_nanmean(x))
proportions_missing_standard_deviations = grouped_proportions_missing.agg(lambda x: suppressed_nanstd(x))


def plot_errors(x, y, errors, start, end):
    """
    Plots a subset of sigmas.
    """
    plt.rcParams["figure.figsize"] = (100, 10)
    plt.errorbar(x[start:end], y[start:end], errors[start:end], linestyle="None", marker="^")
    plt.xticks(rotation=90)
    plt.show()
