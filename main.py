###
#This main module is to run all dependent modules
#output: 
#- EDA graphs
#- graphs of each models feature selection
#- average top features 
###

import feature_selection as fs

combined_features = fs.combine_features()
combined_features

