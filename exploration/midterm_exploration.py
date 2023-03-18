"""
Provides functions for creating plots for the midterm presentation/report.
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
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


def plot_scatterplot(
        data: pd.DataFrame,
        x: str,
        y: str,
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        png: str = None
) -> None:
    """
    Plots a scatterplot of two features from the input dataframe, coloring the points by diagnosis.

    Args:
        data: The dataframe containing at least the features to be plotted.
        x: The name of the features to plot on the x-axis.
        y: The name of the features to plot on the y-axis.
        title: The title of the plot.
        xlabel: The label of the x-axis.
        ylabel: The label of tye y-axis.
        png: The name of the png file to save.

    Returns:
        None
    """
    scatterplot = sns.scatterplot(
        data,
        x=x,
        y=y,
        hue="P1_PT_TYPE",
        palette=["r", "g", "b"]
    )
    scatterplot.set_title(title, fontsize=18)
    scatterplot.set_xlabel(xlabel, fontsize=14)
    scatterplot.set_ylabel(ylabel, fontsize=14)
    if png:
        plt.savefig(f"{png}.png", dpi=1000)
    plt.show()


def plot_grouped_barplot(
        data: pd.DataFrame,
        x: str,
        y: str,
        hue: str
):
    sns.set_theme(style="whitegrid")
    barplot = sns.catplot(
        data=data,
        kind="bar",
        x=x,
        y=y,
        hue=hue,
        # palette="dark", alpha=.6, height=6
    )
    barplot.despine(left=True)
    barplot.set_axis_labels("", "Body mass (g)")
    barplot.legend.set_title("")
    plt.show()


def plot_proteomics_histograms(data: pd.DataFrame, feature: str) -> None:

    x = data[[feature, "P1_PT_TYPE"]]
    x = x[~np.isnan(x[feature])]
    x = x[x != 999999999]

    print(np.isnan(x[feature]))
    xlim = (np.percentile(x[feature], 10), np.percentile(x[feature], 90))
    print(xlim)

    alzheimers_x = x[x["P1_PT_TYPE"] == 1]
    histplot = sns.histplot(x=alzheimers_x[feature])
    histplot.set(xlim=xlim)
    plt.show()

    mci_x = x[x["P1_PT_TYPE"] == 4]
    histplot = sns.histplot(x=mci_x[feature])
    histplot.set(xlim=xlim)
    plt.show()

    control_x = x[x["P1_PT_TYPE"] == 2]
    histplot = sns.histplot(x=control_x[feature])
    histplot.set(xlim=xlim)
    plt.show()
