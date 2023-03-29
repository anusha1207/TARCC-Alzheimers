# MODELING

## logistic.py
This python file executes a multinomial logistic regression model on a Pandas DataFrame. Utilization of an elastic net penalty, we use this model to predict classifications between AD (Alzheimer's), MCI (Mild Cognitive Impairment), and Control patients. This model is cross validated with a grid 
search of parameters to ensure an optimal combination of hyperparameters are chosen. After learning upon patient data, we calculate f1-micro scores on the testing
data to evaluate the score between the three diagnoses and represent the findings through a confusion matrix. The optimal parameters and best features via permutation 
feature importance are also recorded in the output.

## mrmr.py
This python file executes MRMR (minimum redundancy, maximum relevance) algorithm upon a Pandas DataFrame to identify the most important features. Given a specified integer k, the perform_mrmr function will return the top k features that are identified through this algorithm on a specified DataFrame.
