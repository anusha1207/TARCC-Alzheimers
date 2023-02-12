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

# Scan each row to see if it has any correlations over 0.5 or under -0.5. If not, add to list to delete

rows_delete = []
for index, row in corr_values.iterrows():
    delete = True
    for value in row: 
        if value >= 0.5 or value <= -0.5:
            delete = False
    rows_delete.append(tuple([index, delete]))

# Now, delete row/columns that we've identified as True
# Initial size = 735x735, reduced is 664x664, 71 values deleted

for row in rows_delete:
    if row[1]:
        corr_values = corr_values.iloc[corr_values.index!=row[0], corr_values.columns!=row[0]]

# Fill diagonal and upper half with NaNs
mask = np.zeros_like(corr_values, dtype=bool)
mask[np.triu_indices_from(mask)] = True
corr_values[mask] = np.nan
f = plt.figure(figsize=(19, 15))
plt.matshow(corr_values, fignum=f.number)
plt.xticks(range(corr_values.select_dtypes(['number']).shape[1]), corr_values.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(corr_values.select_dtypes(['number']).shape[1]), corr_values.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)