"""
Computes statistics for the cleaned data.

TODO:
plot on log scale - make y-axis log scale
encoding
normalize data by range: for each patient, divide the SD by the range of the total feature.
find out how many are 0
take a look at medians too
violin plot or boxplot
"""

from matplotlib import pyplot as plt
import numpy as np

from data_cleaning import get_cleaned_data


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


df = get_cleaned_data()

# Take the standard deviation of each feature, grouping by patient ID.
grouped_sigmas = df.groupby("PATID").agg(suppressed_nanstd)
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
    marker="o",
)
plt.yscale("log")
plt.xticks(rotation=90)
plt.title("Patient-wise feature means and standard deviations (log scale)")
plt.xlabel("Feature Name")
plt.ylabel("Patient-Wise Mean and SD")
plt.show()

# Plot the error bars of the 10 features with the most variance.
plt.rcParams["figure.figsize"] = (12, 4)
top_10_errors_indices = np.argsort(-sigma_means.values)[:10]
top_10_highest_variance_features = sigma_means[top_10_errors_indices].index
plt.errorbar(
    [
        "Anti-dementia Drug Hx A: Strength",
        "Prescription A: Strength",
        "Vitamin E Hx A: Strength",
        "Anti-dementia Drug Hx B: Strength",
        "Anti-dementia Drug Hx C: Strength",
        "Anti-dementia Drug Hx D: Strength",
        "Systemic Steroids Hx A: Strength"
    ],
    # top_10_highest_variance_features,
    sigma_means[top_10_highest_variance_features][:7],
    sigma_standard_deviations[top_10_highest_variance_features][:7],
    linestyle="None",
    marker="o"
)
plt.yscale("log")
plt.xticks(rotation=20)
plt.title("Top 7 patient-wise feature means and standard deviations (log scale)")
plt.xlabel("Feature Name")
plt.ylabel("Patient-Wise Mean and SD")
plt.subplots_adjust(bottom=0.35)
plt.show()

# Count the proportion of missing values for each feature.
plt.rcParams["figure.figsize"] = (12, 4)
proportions_missing = df.agg(count_proportion_missing)
sorted_indices = np.argsort(proportions_missing)[::-1]
plt.bar(df.columns[sorted_indices], proportions_missing[sorted_indices])
plt.title("Proportion of missing values by feature")
plt.xlabel("Feature Name")
plt.ylabel("Proportion of Missing Values")
plt.show()




# Number of unique patients that drew blood: 594
# Number of unique patients that did not draw blood: 3076
plt.rcParams["figure.figsize"] = (6, 8)
plt.bar(["Drew blood", "Did not draw blood"], [594, 3076], color=["red", "blue"])
plt.title("Number of patients who drew blood / did not draw blood")
plt.ylabel("Number of patients (duplicates not counted)")
plt.show()


blood = df[""]


classes = df["P1_PT_TYPE"].value_counts()
classes.index = ["Control", "AD", "MCI", "Other"]
# classes.plot(kind="pie")
sizes = classes.values
pie = plt.pie(sizes, autopct='%1.1f%%', startangle=90)
plt.legend(labels=classes.index)
plt.savefig("PTTYPE.png", dpi=10000)
plt.show()


# For report revision: Just taking a few features to demonstrate patient-wise feature variances.

import matplotlib.pyplot as plt

temp = get_cleaned_data()
features = ["A1_RACE", "C1_WAIS3_DIGTOT", "B1_BMI", "RBM_Insulin"]
temp = temp[["PATID"] + features]

# Normalize each of these features
for f in features:
    temp[f] = (temp[f] - np.nanmean(temp[f])) / np.nanstd(temp[f])

# Compute patient-wise variances
grouped_sigmas = temp.groupby("PATID").agg(suppressed_nanstd)
sigma_means = grouped_sigmas.agg(suppressed_nanmean)
sigma_standard_deviations = grouped_sigmas.agg(suppressed_nanstd)

to_plot = {f: grouped_sigmas[f][~np.isnan(grouped_sigmas[f])] for f in features}

plt.boxplot(to_plot.values(), showfliers=False)
plt.xticks([1, 2, 3, 4], ["Race", "WAIS3 Digits Score", "BMI", "Insulin"])
plt.title("Boxplots of patient-wise standard deviations for 4 features")
plt.ylabel("Patient-wise SD")
plt.show(dpi=2000)
