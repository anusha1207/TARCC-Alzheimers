import pandas as pd
import numpy as np
import sklearn

def LR_model(df: pd.DataFrame):
    # TODO: filter df into X,y
    # scale X data


    clf = sklearn.LogisticRegressionCV(
        penalty = 'elasticnet', l1_ratio = 0.5, multi_class = 'multinomial', solver = 'saga').fit(X, y)