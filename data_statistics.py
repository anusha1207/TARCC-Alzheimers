"""
Computes statistics for the cleaned data.
"""

from matplotlib import pyplot as plt
import numpy as np

from data_cleaning import get_cleaned_data
from data_cleaning import split_csv


def suppressed_nanmean(x: np.ndarray, log: bool = False):
    """
    Performs the same operation as np.nanmean but suppresses warnings.

    Args:
        x: The array to take the mean over.
        log: Whether to use a log scale.

    Returns:
        None.
    """
    if np.all(np.isnan(x)):
        return np.nan
    nanmean = np.nanmean(x)
    if log:
        return np.nan if nanmean == 0.0 else np.log(nanmean)
    else:
        return nanmean


def suppressed_nanstd(x: np.ndarray, log: bool = False):
    """
    Performs the same operation as np.nanstd but suppresses warnings.

    Args:
        x: The array to take the standard deviation over.
        log: Whether to use a log scale.

    Returns:
        None.
    """
    if np.all(np.isnan(x)):
        return np.nan
    nanstd = np.nanstd(x)
    if log:
        return np.nan if nanstd == 0.0 else np.log(nanstd)
    else:
        return nanstd


def count_proportion_missing(x: np.ndarray) -> float:
    """
    Counts the proportion of NaN values in an array.

    Args:
        x: The array over which to count NaN values.

    Returns:
        The proportion of NaN values in an array.
    """
    return np.count_nonzero(np.isnan(x)) / x.size


df = get_cleaned_data()
blood_df, clinical_df = split_csv(df)

# Take the standard deviation of each feature, grouping by patient ID.
grouped_sigmas = clinical_df.groupby("PATID").agg(lambda x: suppressed_nanstd(x, log=True))
sigma_means = grouped_sigmas.agg(suppressed_nanmean)
sigma_standard_deviations = grouped_sigmas.agg(suppressed_nanstd)
high_variance_features = sigma_means[np.logical_not(np.logical_or(np.isnan(sigma_means), sigma_means < 0.5))].index

with open("high_variance_features.log", "w") as f:
    for high_variance_feature in high_variance_features:
        f.write(f"{high_variance_feature}\t{sigma_means[high_variance_feature]}\n")

# Plot the error bars of features which have "significant" variance.
plt.rcParams["figure.figsize"] = (100, 10)
plt.errorbar(
    high_variance_features,
    sigma_means[high_variance_features],
    sigma_standard_deviations[high_variance_features],
    linestyle="None",
    marker="^"
)
plt.xticks(rotation=90)
# plt.savefig("BloodStandardDPlot.pdf", format="pdf", bbox_inches="tight")
# plt.savefig("ClinicalStandardDPlot.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Count the proportion of missing values for each feature, grouping by patient ID.
# grouped_proportions_missing = df.groupby("PATID").agg(count_proportion_missing)
# proportions_missing_means = grouped_proportions_missing.agg(suppressed_nanmean)
# proportions_missing_standard_deviations = grouped_proportions_missing.agg(suppressed_nanstd)


# plot on log scale
# find out how many are 0
# take a look at medians too
# violin plot or boxplot
