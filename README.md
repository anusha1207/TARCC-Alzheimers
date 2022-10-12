# TARCC_F22

The EDA or Exploratory Data Analysis section has multiple parts. 

> **Part I**
1. Data Loading, Importing Libraries and Mounting the Drive

> **Part II**
A. Data preprocessing modular block
  1. This block consists of a data wrangling code.
  2. Columns which have over 55% of null values are removed.
  3. Unimportant columns and features are removed and the dataset is resized.
  4. Continuous variables are encoded into categorical variables.
  5. The preprocessing step brings down the number of features from 787 to 325
  
> **Part III**
 B. One Hot Encoding
  1. Using pandas "get_dummies" feature, all the 325 variables are one-hot-encoded.
  2. This function block takes time to execute due to the largeness of the dataset.
  3. One-hot-encoding produces multiple boolean variable columns based on the features
 
> **Part IV**
  1. K-modes clustering algorithm is run on a subset of the dataset to visualize the division of the data into different clusters.
   
> **Part V**
  1. (Under construction) Feature selection methods like recursive feature elimination and backwards feature selection using scikit library is run. 
  
