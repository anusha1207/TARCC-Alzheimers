# TARCC_F22

1. Eliminate columns that may be unnecessary to what we are looking for in the dataset in accordance with the project goals
2. Investigate columns/features with null values (NaN) and drop columns/features that have > 10000 NaN values 
3. Investigate the rest of the columns/features with still NaN values and fill the missing values in accordance to the purpose of these columns/features as stated from the codebook (eg fill in missing values for features with "99" if 99 is supposed to be representing "unknown" for that certain column/feature according to the codebook; fill in missing values according to a formula of dependent columns/features for columns/features that depend on other certain columns/features that determine the sum; fill in missing values with "-1" for entries in columns/features that do not specify a missing value to be filled, but yet need to specify that the value is not applicable or missing other than "null" or "NaN"
4. Drop columns/features that indicate a unique value for each entry (eg. STUDYID) since they may be unnecessary to what we are looking for in the dataset in accordance with the project goals 
5. (Still in progress) Apply K-Modes clustering to the dataset to see how the dataset clusters
6. (Still in progress) Apply feature selection to the dataset to see most important features that can help with what we are looking for in the dataset in accordance with the project goals
7. (Still in progress) Apply Random Forests and XGBoost as models for our dataset. 
