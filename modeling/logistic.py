import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from preprocessing.data_cleaning import get_cleaned_data, split_csv
from preprocessing.encoding import encode_data


def LR_model(df: pd.DataFrame):
    # TODO: filter df into X,y
    # scale X data


    clf = sklearn.LogisticRegressionCV(
        penalty = 'elasticnet', l1_ratio = 0.5, multi_class = 'multinomial', solver = 'saga').fit(X, y)


def elasticnet_model(df: pd.DataFrame) -> None:
    import numpy as np

    data = encode_data(get_cleaned_data())
    data = data[data["P1_PT_TYPE"] != 3]
    blood_data, _ = split_csv(data)
    blood_data_features = blood_data.drop("P1_PT_TYPE", axis=1).columns

    # TODO: KNN by group doesn't work yet because patients with all NaN values for a feature will be turned into 0s.
    ## Possible solution: After grouping, look for patients with features with all NaNs. Replace all these NaNs with some
    ## special constant (like 1234567890) and impute the values by group. Then replace the constants with NaN again, stack
    ## the patients back up to the original dataset, and impute the entire thing normally.
    # ungrouped_X = blood_data.drop("P1_PT_TYPE", axis=1)
    # grouped_data = ungrouped_X.groupby("PATID")
    # imputed_grouped_data = grouped_data.apply(lambda x: imputer.fit_transform(x.values))
    # X = np.vstack(imputed_grouped_data)
    # y = blood_data["P1_PT_TYPE"]

    X = blood_data.drop("P1_PT_TYPE", axis=1).values
    y = blood_data["P1_PT_TYPE"]

    imputer = KNNImputer(keep_empty_features=True)
    X = imputer.fit_transform(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    logistic_regression_model = LogisticRegressionCV(
        penalty="elasticnet",
        Cs=[0.001, 0.01, 0.1, 1, 10, 100, 1000],
        l1_ratios=[0, 0.2, 0.4, 0.6, 0.8, 1],
        solver="saga",
        max_iter=1000,
        n_jobs=-1
    )
    logistic_regression_fit = logistic_regression_model.fit(X_train, y_train)

    best_C = logistic_regression_model.C_[0]
    print(f"Best C: {best_C}")

    best_l1_ratio = logistic_regression_model.l1_ratio_[0]
    print(f"Best l1 Ratio: {best_l1_ratio}")

    predictions = logistic_regression_model.predict(X_test)
    micro_f1 = f1_score(y_test, predictions, average="micro")
    print(f"Micro-F1 Score: {micro_f1}")

    # TODO: Feature importance



#
# import numpy as np
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegressionCV
#
# # Local imports
# from preprocessing.data_cleaning import get_cleaned_data, split_csv
# from preprocessing.encoding import encode_data
#
#
# data = encode_data(get_cleaned_data())
# blood_data, _ = split_csv(data)
#
# X = blood_data.drop("P1_PT_TYPE", axis=1)
# y = blood_data["P1_PT_TYPE"]
#
# imputer = SimpleImputer()
# X = imputer.fit_transform(X)
#
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
#
# logistic_regression_model = LogisticRegressionCV(
#     penalty="elasticnet",
#     Cs=[0.01],
#     l1_ratios=[0],
#     multi_class="multinomial",
#     solver="saga",
#     max_iter=1000
# )
# logistic_regression_fit = logistic_regression_model.fit(X, y)
#
# ad_coef_magnitudes = np.abs(logistic_regression_fit.coef_[1])
# best_feature_indices = np.argsort(ad_coef_magnitudes)
