# Texas Alzheimers' Research Care and Consortium: Team TARCC 'S23

## Table of Contents

1. [About the Project](#about-the-project)
2. [Prerequisites](#prerequisites)
3. [Folder Structure](#folder-structure)
4. [Installation and Usage Instructions](#installation-and-usage-instructions)
5. [Data](#data)
6. [Data Science pipeline](#data-science-pipeline)
- [Preprocessing](#data-science-pipeline)
- [Feature Selection](#data-science-pipeline)
- [Modeling](#data-science-pipeline)
- [Results and Interpretation](#data-science-pipeline)
7. [Contributors](#contributors)


## About the Project
Our main goal is to determine biological and non-biological risk factors for Alzheimer’s and analyze the weight of influence of these factors in developing or resisting the progress of Alzheimer’s Disease.

## Prerequisites
This project installs and uses the following packages.
- [Numpy](https://pypi.org/project/numpy/)
- [Pandas](https://pypi.org/project/pandas/)
- [Matplotlib](https://pypi.org/project/matplotlib/)
- [Seaborn](https://pypi.org/project/seaborn/)
- [Statsmodel](https://pypi.org/project/statsmodels/)
- [Scikit Learn](https://pypi.org/project/scikit-learn/)
- [Scipy Stats](https://pypi.org/project/scipy/)
- [MLXtend](https://pypi.org/project/mlxtend/)
- [LightGBM](https://pypi.org/project/lightgbm/)
- [XGBoost](https://pypi.org/project/xgboost/)
- [Boruta](https://pypi.org/project/Boruta/)
- [SHAP](https://pypi.org/project/shap/)
- [mca](https://pypi.org/project/mca/)
- [bayesian-optimization](https://pypi.org/project/bayesian-optimization/) 

## Folder Structure
* config: The config folder contains two files
  * config.py: sets global variables for parameters and settings for methods, and the data path for the data.
  * requirements.txt: contains all the dependencies and necessary packages/libraries
* data: The data folder contains pickled features from different models (LGBM, catboost, extra trees, random forest and XGBoost). It also contains the combined pickled features.
* dataset: The dataset contains a csv file of our raw input. This is present in the repository for our sponsors to easily reproduce and replicate results.
* results: This folder contains the results of all our tests, and the combined AUC-ROC plots.
* script: This folder consists of preprocessing, modeling, feature selection modules.
  * preprocessing: `preprocessing.py` contains the function block to preprocess the data and subset it to usable features from the blood biomarker and diseases data
  * modeling: `modeling.py` contains functions to run the different models
  * feature selection: `feature_selection.py`  contains functions to evaluate the importance of each feature in determining the most important biomarkers
* [TARCC_Demo.ipynb](TARCC_Demo.ipynb) orchestrates the run and pipes data between other functions






## Installation and Usage Instructions
1. Clone the repository
```
git clone https://github.com/RiceD2KLab/TARCC_F22.git
```
2. Install dependencies
```
pip install -r config/requirements.txt
```
3. Copy the TARCC.csv file from the dataset folder and paste it in the main directory. This will ensure proper configuration from which main.py accesses the data.

4. Run main.py
```
python scripts/main.py
```


## Data
The data we have is from Texas Alzheimer's Research Care and Consortium. It consists of clinical survey information of different patients. There are different types of columns pertaining to information about the patient, mental and physical disorders, blood test and protein data, as well as other statistical and phenotypical information. The dataset size is 14655 observations/rows and 787 features/columns. The 14655 observations are repetitive visits which are spread across 3670 unique patients.

## **Data Science Pipeline**

![Pipeline](https://user-images.githubusercontent.com/97485268/198148374-fc9760c7-bf3f-4b82-8a7c-b83b73d82556.png)

### PREPROCESSING <br />
The data pertaining to blood and protein bio-markers along with mental and physical attributes is subsetted to perform a line of analysis which helps us gauge the most important features. Upon doing this, the resulting data-set consisted of 563 patients with a total of 217 features. After preprocessing, the patient database is divided broadly into two classes based on the diagnosis, whether they have AD or not.
<br />

### FEATURE SELECTION <br />
The feature selection methods we've used are as follows. <br />
- **STAT BASED FEATURE SELECTION**
1. Chi Square Test
2. Mutual Information Test <br />
- **EMBEDDED FEATURE SELECTION** <br />
3. Random Forest <br />
- **WRAPPER BASED** <br />
4. Recursive Feature Elimination: Using Decision Tree Classifier 
5. Recursive Feature Elimination: Using Random Forest Classifier
6. Forward Feature Selection: Using Decision Tree Classifier
7. Forward Feature Selection: Using Random Forest Classifier 
8. Backwards Features Elimination: Using Logistic Regression Classifier
9. Backwards Feature Elimination: Using Decision Tree Classifer <br />
- **OTHER METHODS** <br />
10. Boruta Test 
11. Multiple Correspondence Analysis (MCA) <br />
 
### MODELING AND VALIDATION <br />
- **Classification Models** <br />
1. Light Gradient Boosting Method (LGBM)
2. Categorical Boosting (CatBoost)
3. eXtreme Gradient Boosting (XGB)
4. Gradient Boosting 
5. Random Forest 
6. Extra Trees
7. Decision Tree
8. Logistic Regression  <br />

- **Hyperparameter Tuning** <br />
1. Hyperoptimization (Hyperopt)
2. Bayesian Optimization (BayesSearchCV) <br />

- **Evaluation Metrics** <br />
1. **F-Beta Score** - The primary evaluation metric for our analysis
2. Specificity
3. Receiver Operating Characteristic - Area Under the Curve (ROC-AUC) 
4. Area Under the Precision Recall Curve (AUPRC)
5. Recall/Sensitivity 
6. F1 Score
7. Confusion Matrix
8. Classificaiton Report
9. Precision Score
10. Accuracy Score <br />

### INTERPRETATION AND RESULTS <br />
SHAP Interpretation
<br />

## Contributors
**Cao, Angela** <br />
**Muddapati, Anusha** <br />
**Gan, Wei Ren** <br />
**Prieto, Sophia** <br />
**Viswanathan, Tejeshwine** <br /> 

<hr style="border:2px">

*This project was a part of DSCI 535/COMP 449/COMP 549 as a part of D2K Lab, in Rice University.*
