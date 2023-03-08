import pandas as pd
import numpy as np
import sklearn

def LR_model(df: pd.DataFrame):
    # TODO: filter df into X,y
    # scale X data


    clf = sklearn.LogisticRegressionCV(
        penalty = 'elasticnet', l1_ratio = 0.5, multi_class = 'multinomial', solver = 'saga').fit(X, y)



import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV

# Local imports
from preprocessing.data_cleaning import get_cleaned_data, split_csv
from preprocessing.encoding import encode_data


data = encode_data(get_cleaned_data())
blood_data, _ = split_csv(data)

X = blood_data.drop("P1_PT_TYPE", axis=1)
y = blood_data["P1_PT_TYPE"]

imputer = SimpleImputer()
X = imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

logistic_regression_model = LogisticRegressionCV(
    penalty="elasticnet",
    Cs=[0.01],
    l1_ratios=[0],
    multi_class="multinomial",
    solver="saga",
    max_iter=1000
)
logistic_regression_fit = logistic_regression_model.fit(X, y)

ad_coef_magnitudes = np.abs(logistic_regression_fit.coef_[1])
best_feature_indices = np.argsort(ad_coef_magnitudes)
