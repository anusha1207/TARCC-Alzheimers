import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import seaborn as sns

from utils.utils import get_features_label, split_csv

from preprocessing.cleaning import get_cleaned_data
from preprocessing.encoding import encode_data

from exploration.data_correlations import plot_correlations
from exploration.data_statistics import plot_labels_pie_chart, plot_blood_draw_statistics
from exploration.midterm_exploration import plot_feature_against_diagnosis

from modeling.mrmr_feature_selection import plot_accuracy_with_features, perform_mrmr
from modeling.logistic import run_elastic_net, evaluate_results
import numpy as np


def plot_mrmr_features(scores):
    scores = scores.sort_values(ascending=False)
    scores = scores.head(10)

    sns.set(style="whitegrid")
    ax = scores.plot(kind = 'bar')
    plt.xticks(rotation = 90)
    plt.ylabel("Accuracy Score")
    plt.show()

def plot_mrmr_features_scaled(scores):
    scores = scores.sort_values(ascending=False)
    scores = scores.head(10)
    max_score = max(scores)
    scores = scores.div(max_score)

    ax = scores.plot(kind='bar')
    plt.xticks(rotation=90)
    plt.ylabel("Accuracy Score")
    plt.show()


def plot_rf_features(result, features):
    sns.set(style="whitegrid")
    top_10_idx = np.argsort(result.importances_mean)[::-1][:10]
    top_10_features = features[top_10_idx]
    top_10_scores = result.importances_mean[top_10_idx]
    top_10_std = result.importances_std[top_10_idx]

    # Print the top 10 features and their scores
    print("Standard deviation of the top 10 features by permutation importance:")
    for feature, score in zip(top_10_features, top_10_std):
        print(f"{feature}: {score}")


    # Print the top 10 features and their scores
    print("Top 10 features by permutation importance:")
    for feature, score in zip(top_10_features, top_10_scores):
        print(f"{feature}: {score}")


    plt.bar(top_10_features, top_10_scores)
    plt.xticks(rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Permutation importance score")
    plt.title("Top 10 features by permutation importance")
    plt.show()

def plot_rf_features_scaled(result, features):
    top_10_idx = np.argsort(result.importances_mean)[::-1][:10]
    top_10_features = features[top_10_idx]
    top_10_scores = result.importances_mean[top_10_idx]
    max_score = max(top_10_scores)
    top_10_scores = np.divide(top_10_scores, max_score)
    # top_10_std = result.importances_std[top_10_idx]


    # Print the top 10 features and their scores
    print("Top 10 features by permutation importance:")
    for feature, score in zip(top_10_features, top_10_scores):
        print(f"{feature}: {score}")


    plt.bar(top_10_features, top_10_scores)
    plt.xticks(rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Permutation importance score")
    plt.title("Top 10 features by permutation importance")
    plt.show()

def plot_mrmr_and_rf(scores, result, features):
    scores = scores.sort_values(ascending=False)
    scores = scores.head(10)
    max_score = max(scores)
    scores = scores.div(max_score)

    top_10_idx = np.argsort(result.importances_mean)[::-1][:10]
    top_10_features = features[top_10_idx]
    top_10_scores = result.importances_mean[top_10_idx]
    max_score = max(top_10_scores)
    top_10_scores = np.divide(top_10_scores, max_score)


    # Set the style to 'whitegrid'
    sns.set_style('whitegrid')
    colors = {'MRMR': 'blue', 'Random Forest': 'orange'}

    # Create a new pandas series to combine the two sets of data
    all_scores = pd.concat([scores, pd.Series(data=top_10_scores, index=top_10_features)])

    # Set up a new dataframe to differentiate between the two sets of data
    all_scores_df = pd.DataFrame({'feature': all_scores.index, 'score': all_scores.values})
    all_scores_df.loc[:9, 'type'] = 'MRMR'
    all_scores_df.loc[10:19, 'type'] = 'Random Forest'

    # Find the overlapping features
    overlap = scores.index.intersection(top_10_features)

    # Create the bar plot using Seaborn
    ax = sns.barplot(x=all_scores_df.index, y='score', hue='type', data=all_scores_df, palette=colors, width=0.8)
    plt.xticks(rotation=90)

    # Add a legend
    ax.legend(loc='best', fontsize=18, fontweight='bold')

    # Add a title
    ax.set_xlabel('Features', fontsize=12)
    ax.set_xticklabels(all_scores_df['feature'], fontsize=12)
    ax.set_ylabel('Scaled Importance Scores', fontsize=12)
    ax.set_title('Top 10 Features of Both MRMR and Random Forest', fontweight='bold', fontsize=14)
    labels = ax.get_xticklabels()

    for label in labels:
        if label.get_text() in overlap:
            label.set_fontweight('bold')

    ax.set_xticklabels(labels)

    # Rotate the x-axis labels
    plt.show()

    # Create the stacked bar chart for overlapping features using seaborn
    sns.set(style="whitegrid")
    fig, ax = plt.subplots()
    sns.barplot(x=overlap, y=[scores[feature] for feature in overlap], color='blue', label='MRMR', ax=ax)
    sns.barplot(x=overlap, y=[top_10_scores[top_10_features.get_loc(feature)] for feature in overlap], color='orange',
                label='Random Forest', ax=ax, bottom=[scores[feature] for feature in overlap])

    ax.set_xlabel('Features')
    ax.set_ylabel('Importance Scores')
    ax.set_title('Overlapping Important Features of MRMR and Random Forest')
    ax.legend()

    ax.tick_params(axis='x', rotation=90)
    plt.show()

