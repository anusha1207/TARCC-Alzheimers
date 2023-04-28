# MODELING

## logistic.py
This python file executes a multinomial logistic regression model on a Pandas DataFrame. Utilizing of an elastic net penalty, we use this model to predict classifications between AD (Alzheimer's), MCI (Mild Cognitive Impairment), and Control patients. This model is cross validated with a grid 
search of parameters to ensure an optimal combination of hyperparameters are chosen. After learning upon patient data, we calculate f1-micro scores on the testing
data to evaluate the score between the three diagnoses and represent the findings through a confusion matrix. The optimal parameters and best features via permutation 
feature importance are also recorded in the output. The results are pickled for reusability, and are printed after bootstrapping results over iterations.

## mlp.py
This python file executes a MultiLayer Perceptron Neural Net on a Pandas DataFrame. After learning upon patient data, we calculate f1-micro scores on the testing
data to evaluate the score between the three diagnoses and represent the findings through a confusion matrix. The optimal parameters and best features via permutation 
feature importance are also recorded in the output. The results are pickled for reusability, and are printed after bootstrapping results over iterations.

## RandomForest.py
This python file executes a Random Forest model on a Pandas DataFrame. After learning upon patient data, we calculate f1-micro scores on the testing
data to evaluate the score between the three diagnoses and represent the findings through a confusion matrix. The optimal parameters and best features via permutation 
feature importance are also recorded in the output. The results are pickled for reusability, and are printed after bootstrapping results over iterations.

## mrmr.py
This python file executes MRMR (minimum redundancy, maximum relevance) algorithm upon a Pandas DataFrame to identify the most important features. Given a specified integer k, the perform_mrmr function will return the top k features that are identified through this algorithm on a specified DataFrame.

## mrmr_feature_selections.py
This python file applies the MRMR algorithm on our three dataset partitions, producing a cumulative relevance score per additional feature included, ranked in descending order. Given a cutoff in diminishing returns of relevance, we plot the optimal number of features to keep per partition, and return the relevant feature labels

## comparisons.py
Given our logistic, MLP, and RF models, we bootstrap model results and produce a violin plot comparing how the three models score on the different data partitions through a side-by-side comparison
