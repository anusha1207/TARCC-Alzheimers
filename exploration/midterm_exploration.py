"""
Provides functions for creating plots for the midterm presentation/report.
"""
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def plot_feature_against_diagnosis(
        data: pd.DataFrame,
        feature: str,
        title: str = None,
        ylabel: str = None,
        png: str = None
) -> None:
    """
    Plots a boxplot of the input feature for each diagnosis level (Control, MCI, and AD).

    Args:
        data: The dataframe containing at least the input feature and the P1_PT_TYPE label.
        feature: The name of the feature to plot.
        title: The title of the plot.
        ylabel: The label of tye y-axis.
        png: The name of the png file to save.

    Returns:
        None
    """
    sns.set_palette("Set2")
    boxplot = sns.boxplot(
        data,
        y=feature,
        x="P1_PT_TYPE",
        showfliers=False,
        order=[2, 4, 1]
    )
    boxplot.set_title(title, fontsize=18)
    boxplot.set_xlabel("Diagnosis", fontsize=14)
    boxplot.set_ylabel(ylabel, fontsize=14)
    boxplot.set_xticklabels(["Control", "MCI", "AD"], fontsize=12)
    if png:
        plt.savefig(f"{png}.png", dpi=1000)
    plt.show()
