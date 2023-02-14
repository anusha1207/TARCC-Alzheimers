"""
Computes correlations for the cleaned data.
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from data_cleaning import get_cleaned_data

df = get_cleaned_data()

# collect correlations, and eliminate diagonal of 1's
corr_values = df.corr()
np.fill_diagonal(corr_values.values, 0)

# Scan each row to see if it has any correlations over 0.75 or under -0.75. If not, add to list to delete
for column in corr_values:
    corr_values[column].mask(corr_values[column] >= 1.0, np.nan, inplace=True)
    corr_values[column].mask(corr_values[column] <= -1.0, np.nan, inplace=True)
        

rows_delete = []
for index, row in corr_values.iterrows():
    delete = True
    for value in row: 
        if (value >= 0.5 and value < 0.995) or (value <= -0.5 and value > -0.995):
            delete = False
    rows_delete.append(tuple([index, delete]))

# Now, delete row/columns that we've identified as True
# Initial size = 735x735, reduced is 664x664, 71 values deleted

for row in rows_delete:
    if row[1]:
        corr_values = corr_values.iloc[corr_values.index!=row[0], corr_values.columns!=row[0]]

corr_values.shape

# following two functions identify top correlation pairs

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

# isolating top 65 pairs to optimize feature visibility
top = get_top_abs_correlations(corr_values, 65)

# recording all rows + columns of interest
topdf = top.unstack(level=-1)
features = set(list(topdf.columns) + topdf.axes[0].tolist())
len(features)

# filtering out nonimportant features

for col in corr_values.columns:
    if col not in features:
        corr_values = corr_values.iloc[corr_values.index != col, corr_values.columns != col]

corr_values.shape

# Fill diagonal and upper half with NaNs
mask = np.zeros_like(corr_values, dtype=bool)
mask[np.triu_indices_from(mask)] = True
corr_values[mask] = np.nan
f = plt.figure(figsize=(19, 15))
plt.matshow(corr_values, fignum=f.number)
plt.xticks(range(corr_values.select_dtypes(['number']).shape[1]), corr_values.select_dtypes(['number']).columns, fontsize=14, rotation=90)
plt.yticks(range(corr_values.select_dtypes(['number']).shape[1]), corr_values.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)