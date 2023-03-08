import mrmr
from mrmr import mrmr_classif
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification

def perform_mrmr(X: pd.DataFrame, y: pd.Series, k: int):
    selected_features = mrmr_classif(X=X, y=y, K=k)
    return selected_features



