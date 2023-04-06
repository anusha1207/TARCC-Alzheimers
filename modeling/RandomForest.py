import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt

# import sys
# sys.path.insert(0, "\xabbo\Desktop\TARCC\TARCC_F22\preprocessing")
from preprocessing.cleaning import get_cleaned_data
from preprocessing.encoding import encode_data
from utils.utils import get_features_label, split_csv

def run_random_forest(df: pd.DataFrame, num_iters: int = 1):
    """
    Runs a random forest model on the input dataframe, using "P1_PT_TYPE" as the label.

    Args:
        df: The cleaned and encoded TARCC dataset.
        num_iters: The number of random forest iterations to perform.

    Returns:

    """
    LABEL = "P1_PT_TYPE"
    # dropping patient ID columns. Removing rows that reflect "other" in response variable
    data = df[df[LABEL] != 3]
    if "PATID" in df.columns:
        data = data.drop("PATID", axis=1)
    if "RBM_TARC_PID" in df.columns:
        data = data.drop("RBM_TARC_PID", axis=1)

    features = data.drop(LABEL, axis=1).columns

    X = data.drop(LABEL, axis=1).values
    y = data[LABEL]

    # KNN Imputation to 
    imputer = KNNImputer()
    X = imputer.fit_transform(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    #best_cs = np.zeros(num_iters)
    #best_l1_ratios = np.zeros(num_iters)
    micro_f1_scores = np.zeros(num_iters)
    feature_importances = []
    confusion_matrices = []

    for i in range(num_iters):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        random_forest_model = RandomForestClassifier()
        random_forest_model.fit(X_train, y_train)

        predictions = random_forest_model.predict(X_test)

        micro_f1_scores[i] = f1_score(y_test, predictions, average="micro")

        # Feature importance
        r = permutation_importance(
            random_forest_model, X_test, y_test,
            scoring="f1_micro",
            n_repeats=10,
            random_state=0
        )
        importance_indices = np.argsort(r["importances_mean"])[::-1]

        top_10_idx = np.argsort(r.importances_mean)[::-1][:10]
        top_10_features = features[top_10_idx]
        top_10_scores = r.importances_mean[top_10_idx]

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

        feature_importances.append(features[importance_indices])

        confusion_matrices.append(confusion_matrix(y_test, predictions))

    return micro_f1_scores, feature_importances, confusion_matrices

df = encode_data(get_cleaned_data())
combined, blood, clinical = split_csv(df)

micro_f1_scores, feature_importances, confusion_matrices = run_random_forest(combined)