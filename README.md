# TARCC_F22

The EDA or Exploratory Data Analysis section has multiple parts. 

>> **Part I**
1. Data Loading, Importing Libraries and Mounting the Drive
>> **Part II**
1. Data preprocessing modular block
  > This block consists of a data wrangling code.
  > Columns which have over 55% of null values are removed.
  > Unimportant columns and features are removed and the dataset is resized.
  > Continuous variables are encoded into categorical variables.
  > The preprocessing step brings down the number of features from 787 to 325.
 >> **Part III**
 1. One Hot Encoding
  > Using pandas "get_dummies" feature, all the 325 variables are one-hot-encoded.
  > This function block takes time to execute due to the largeness of the dataset.
  > One-hot-encoding produces multiple boolean variable columns based on the features.
>>> **Part IV**
  > K-modes clustering algorithm is run on a subset of the dataset to visualize the division of the data into different clusters. 
>>> **Part V**
  > (Under construction) Feature selection methods like recursive feature elimination and backwards feature selection using scikit library is run. 
  
