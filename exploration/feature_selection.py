import pandas as pd
import matplotlib.pyplot as plt

from utils.utils import get_features_label, split_csv

from preprocessing.cleaning import get_cleaned_data
from preprocessing.encoding import encode_data

from exploration.data_correlations import plot_correlations
from exploration.data_statistics import plot_labels_pie_chart, plot_blood_draw_statistics
from exploration.midterm_exploration import plot_feature_against_diagnosis

from modeling.mrmr import plot_accuracy_with_features, perform_mrmr
from modeling.logistic import run_elastic_net, evaluate_results
import numpy as np


def plot_mrmr_features(scores):
    scores = scores.sort_values(ascending=False)
    scores = scores.head(10)

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

