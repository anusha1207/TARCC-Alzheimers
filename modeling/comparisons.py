"""
Provides functions for visualizing and analyzing model results.
"""
from typing import Any

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def plot_f1_scores(
        f1_scores: list[np.array],
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        models: list[str] = None,
        pad: int = None,
        png: str = None
) -> None:
    """
    Plots the micro-F1 scores of the input models as violin plots, with models on the x-axis and scores on the y-axis.

    Args:
        f1_scores: A list of numpy arrays containing the micro-F1 scores of the models.
        title: The title of the plot
        xlabel: The label of the x-axis.
        ylabel: The label of the y-axis.
        models: A list of model names to plot. This list must divide the length of the micro-F1 scores list.
        pad: The number of groups in the final plot, where each group is a collection of related models.
        png: The name of the png file to save.

    Returns:
        None
    """

    # Check that len(models) divides len(f1_scores).
    if len(f1_scores) % len(models) != 0:
        print("len(f1_scores) should be an integral multiple of len(models)")
        return

    # Pad the f1 scores if necessary.
    padded_f1_scores = f1_scores if not pad else insert_value(f1_scores, [np.nan], pad)

    # Pad the model names if necessary.
    complete_models = models
    quotient = len(f1_scores) // len(models)
    if quotient != 1:
        complete_models = []
        for i in range(quotient):
            complete_models += ["" for _ in range(quotient // 2)] + [models[i]] + ["" for _ in range(quotient // 2)]
    padded_models = complete_models if not pad else insert_value(complete_models, "", pad)

    # Pad the colors if necessary.
    colors = ["orange", "green", "blue"]
    padded_colors = colors if not pad else insert_value(colors, "grey", pad)[:-1]

    # Plot the violin plot.
    violin_plot = sns.violinplot(
        padded_f1_scores,
        orient="v",
        scale="width",
        palette=padded_colors
    )
    violin_plot.set_title(title, fontsize=18)
    violin_plot.set_xlabel(xlabel, fontsize=14)
    violin_plot.set_ylabel(ylabel, fontsize=14)
    violin_plot.set_xticklabels(padded_models)
    violin_plot.tick_params(bottom=False)

    # Add vertical lines to separate the models.
    if pad:
        line_indices = [(pad + 1) * i for i in range(1, pad)]
        for line_index in line_indices:
            violin_plot.axvline(x=line_index, linestyle="--", c="grey")

    # Add a legend for the colors.
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in ["orange", "green", "blue"]]
    plt.legend(legend_handles, ["Blood", "Clinical", "Combined"], title="Dataset", loc="lower right", prop={"size": 8})

    if png:
        plt.savefig(f"{png}.png", dpi=100)
    plt.show()


def plot_mci_f1_scores(
        f1_scores: list[np.array],
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        models: list[str] = None,
        pad: int = None,
        png: str = None
) -> None:

    # Check that len(models) divides len(f1_scores).
    if len(f1_scores) % len(models) != 0:
        print("len(f1_scores) should be an integral multiple of len(models)")
        return

    # Pad the f1 scores if necessary.
    padded_f1_scores = f1_scores if not pad else insert_value(f1_scores, [np.nan], pad)

    # Pad the model names if necessary.
    complete_models = models
    quotient = len(f1_scores) // len(models)
    if quotient != 1:
        complete_models = []
        for i in range(quotient):
            complete_models += ["" for _ in range(quotient // 2)] + [models[i]] + ["" for _ in range(quotient // 2)]
    padded_models = complete_models if not pad else insert_value(complete_models, "", pad)

    # Pad the colors if necessary.
    colors = ["green", "green", "green"]
    padded_colors = colors if not pad else insert_value(colors, "grey", pad)[:-1]

    # Plot the violin plot.
    violin_plot = sns.violinplot(
        padded_f1_scores,
        orient="v",
        scale="width",
        palette=padded_colors
    )
    violin_plot.set_title(title, fontsize=18)
    violin_plot.set_xlabel(xlabel, fontsize=14)
    violin_plot.set_ylabel(ylabel, fontsize=14)
    violin_plot.set_xticklabels(padded_models)
    violin_plot.tick_params(bottom=False)

    # Add vertical lines to separate the models.
    if pad:
        line_indices = [(pad + 1) * i for i in range(1, pad)]
        for line_index in line_indices:
            violin_plot.axvline(x=line_index, linestyle="--", c="grey")

    if png:
        plt.savefig(f"{png}.png", dpi=100)
    plt.show()


def insert_value(original_list: list[Any], val: Any, padding: int) -> list[Any]:
    """
    Inserts a value in between elements of a list, including before the first item and after the last item.

    Args:
        original_list: The list where the value is being inserted.
        val: The value to insert.
        padding: The amount of items to skip after each insertion.

    Returns:
        A copy of the original list, modified with the insertions.
    """
    copy = original_list[:]
    for i in range(len(copy) - 1, 0, -1):
        if i % padding == 0:
            copy.insert(i, val)
    copy.insert(0, val)
    copy.append(val)
    return copy
