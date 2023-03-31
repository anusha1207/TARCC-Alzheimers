import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
# import sys
# sys.path.insert(0, "\xabbo\Desktop\TARCC\TARCC_F22\preprocessing")
from preprocessing.data_cleaning import get_cleaned_data




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
    imputer = KNNImputer(keep_empty_features=True)
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
        feature_importances.append(features[importance_indices])

        confusion_matrices.append(confusion_matrix(y_test, predictions))

    return micro_f1_scores, feature_importances, confusion_matrices

# df = encode_data(get_cleaned_data())
# df = df[df["P1_PT_TYPE"] != 3]
# blood, clinical = split_csv(df)

# micro_f1_scores, feature_importances, confusion_matrices = run_random_forest(clinical)