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
grouped_sigmas = df.groupby("PATID").agg(suppressed_nanstd)
sigma_means = grouped_sigmas.agg(suppressed_nanmean)
sigma_standard_deviations = grouped_sigmas.agg(suppressed_nanstd)
high_variance_features = sigma_means[np.logical_not(np.logical_or(np.isnan(sigma_means), sigma_means < 0.5))].index

with open("high_variance_features.log", "w") as f:
    for high_variance_feature in high_variance_features:
        f.write(f"{high_variance_feature}\t{sigma_means[high_variance_feature]}\n")

# Plot the error bars of features which have "significant" variance.
plt.rcParams["figure.figsize"] = (40, 10)
plt.errorbar(
    high_variance_features,
    sigma_means[high_variance_features],
    sigma_standard_deviations[high_variance_features],
    linestyle="None",
    marker="^"
)
plt.xticks(rotation=90)
plt.show()

# Count the proportion of missing values for each feature, grouping by patient ID.
grouped_proportions_missing = df.groupby("PATID").agg(count_proportion_missing)
proportions_missing_means = grouped_proportions_missing.agg(suppressed_nanmean)
proportions_missing_standard_deviations = grouped_proportions_missing.agg(suppressed_nanstd)
