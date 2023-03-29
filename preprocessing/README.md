# PREPROCESSING

## data_cleaning.py
This python file utilizes a function called get_cleaned_data(), which reads the provided .csv file into a Pandas DataFrame. Utilizing sum_D1() and map_value_D1() helper 
functions, the D1 prefixed features are transformed to reflect patient diagnostic weights. Then using the drop_features() helper function, a json of all feature names 
(found in config/data_codes.json) to be dropped is processed and inputted so that our DataFrame features correspond. get_cleaned_data() then continues to convert specific column values to reflect patient perscription levels and clean irregular null values given in the codebook.

## encoding.py
Given the json of column names (found in config/data_codes.json) to operate upon, this function performs dummy encoding on our categorical dataset features
