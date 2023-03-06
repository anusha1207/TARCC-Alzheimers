"""
Computes correlations for the cleaned data
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def filter_corr(df: pd.DataFrame):
    """
    Function to delete rows and column with low correlations

    Input: df, a pandas dataframe modeling correlation values (from df.corr())
    Output: None, alters the correlations of df
    """
    np.fill_diagonal(df.values, 0)

    # Scan each row to see if it has any correlations over 1 or under -1. If not, add to list to delete
    for column in df:
        df[column].mask(df[column] >= 1.0, np.nan, inplace=True)
        df[column].mask(df[column] <= -1.0, np.nan, inplace=True)
    
    rows_delete = []
    for index, row in df.iterrows():
        delete = True
        for value in row: 
            if (value >= 0.5 or value <= -0.5):
                delete = False
        rows_delete.append(tuple([index, delete]))

    # Now, delete row/columns that we've identified as True
    for row in rows_delete:
        if row[1]:
            df = df.iloc[df.index!=row[0], df.columns!=row[0]]

def mask_(elem):
    """
    masking correlations < |.5| to be nan
    """
    if elem < 0.5 and elem > -0.5 and type(elem) != str:
        return np.nan
    else:
        return elem


# following two functions identify top correlation pairs

def get_redundant_pairs(df):
    """
    Get diagonal and lower triangular pairs of correlation matrix
    Inputs: df, a correlation matrix
    Outputs: set of redundant column pairs to drop
    """
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    """
    Returns the top correlations of a given correlation matrix
    Inputs: df, a correlation matrix
    Outputs: top n features ranked by correlation values
    """
    au_corr = df.abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

def filter_top_features(df):
    """
    Filters Correlation Matrix to contain only the columns containing top correlations
    Inputs: df, a correlation matrix
    Outputs: None, alters the correlations of df
    """
    # isolating top 25 pairs to optimize feature visibility
    top = get_top_abs_correlations(df, 25)

    # recording all rows + columns of interest
    topdf = top.unstack(level=-1)
    features = set(list(topdf.columns) + topdf.axes[0].tolist())

    # filtering out nonimportant features
    for col in df.columns:
        if col not in features:
            df = df.iloc[df.index != col, df.columns != col]

def plot_correlations(df: pd.DataFrame):
    """
    Plots Correlation Matrix based on above filtrations and specifications
    """
    # calling above functions
    corr_values = df.corr()
    filter_corr(corr_values)
    corr_values = corr_values.applymap(mask_)
    filter_top_features(corr_values)

    # Fill diagonal and upper half with NaNs
    mask = np.zeros_like(corr_values, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    corr_values[mask] = np.nan

    plt.figure(figsize=(19,15))
    sns.heatmap(corr_values, cmap="coolwarm")
    plt.show()
