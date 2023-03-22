import mrmr
import mrmr
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification

def perform_mrmr(X: pd.DataFrame, y: pd.Series, k: int, scr):
    selected_features = mrmr.mrmr_classif(X=X, y=y, K=k, return_scores= scr)
    return selected_features



