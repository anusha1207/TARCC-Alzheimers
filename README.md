# Texas Alzheimers' Research Care and Consortium: Team TARCC 'S23

## Table of Contents

1. [About the Project](#about-the-project)
2. [Prerequisites](#prerequisites)
3. [Folder Structure](#folder-structure)
4. [Installation and Usage Instructions](#installation-and-usage-instructions)
5. [Data](#data)
6. [Data Science pipeline](#data-science-pipeline)
7. [Preprocessing](#data-science-pipeline)
8. [Feature Selection](#data-science-pipeline)
9. [Modeling](#data-science-pipeline)
- [Results and Interpretation](#data-science-pipeline)
10. [Contributors](#contributors)


## About the Project
Our main goal is to determine biological and non-biological risk factors for Alzheimer’s and analyze the weight of influence of these factors in developing or resisting the progress of Alzheimer’s Disease.

## Prerequisites
This project installs and uses the following packages.
- [Numpy](https://pypi.org/project/numpy/)
- [Pandas](https://pypi.org/project/pandas/)
- [Scikit Learn](https://pypi.org/project/scikit-learn/)
- [Scipy Stats](https://pypi.org/project/scipy/)
- [Seaborn](https://seaborn.pydata.org/)
- [mrmr](https://github.com/smazzanti/mrmr)
- [matplotlib](https://matplotlib.org/)


## Folder Structure
* config: This folder contains a file named requirements.txt which contains all the packages you need to install.
* data: The data folder contains the csv file of our raw input (TARCC_data.csv) 
* exploration: This contains initial data exploration files.
* preprocessing: This folder contains a preprocessing file which cleans our data's missing values and features.
* main.ipynb: The main notebook for running functions located in other modules. This notebook serves as a demo of what our code can do.


## Installation and Usage Instructions
1. Clone the repository
```
git clone https://github.com/RiceD2KLab/TARCC_F22.git
```
2. Install dependencies
```
pip install -r config/requirements.txt
```

## Data
Our data is provided by the Texas Alzheimer's Research Care and Consortium (TARCC) and contains 14,655 observations of clinical visit data spread acorss 3,670 unique patients. There are 787 features (columns) which are either clinical (mental/physical disorders, measurements, prescriptions, cognitive assesments, living habits) or biomarkers (blood tests and protein levels) and can be formatted as numerical, categorical, binary, date, or unstructured. 
The patients can have repetitive visits and are diagnosed each time as either having AD, MCI, or healthy. 

## **Data Science Pipeline**
Data Wrangling --> Data Exploration --> Modeling --> Validation 
1. Data Wrangling: Here, we merged data entries by patient to retain both clinical and biomarkers data. We also cleaned missing values and removed features which our sponsor deemed irrelevant. 
2. Data Exploration: We created a correlation matrix to analyze features which were highly correlated with each other. 
3. Modeling: We plan to use a two-model system; for patients who did not draw blood, we will use clinical data in the modeling phase, and for patients who do have biomarker data recorded, we will take into account both their blood-draw data and their clinical data. For now, we plan to use MRMR and Lasso/Ridge Regression. 
4. Evaluation: For now, we plan to use a simple train-validation-test split, k-folds cross-validation, and the use of an external dataset for testing. 


## Preprocessing <br />
Our preprocessing stage merged data by patient visits to minimze the number of missing values per row (for example, blood tests are only taken on the first visit for many patients). 
<br />

## Feature Selection
Our team conducted feature selection using a model agnostic approach: maximum relevancy, minimal redundancy (MRMR). MRMR is an iterative approach that selects the best features based on their highest correlation to the response variable (relevance), and lowest correlation between features (redundancy).

## Modeling
Our modeling stage is currently utilizing a multiclass logistic regression, Random Forest, and MultiLayer Perceptron Neural Net for prediction of disease class. We evaluate these results, bootstrapping our scores and plotting them side-by-side via a violin plot to compare performances across different models and data partitions. Here, we also calculate the points of diminishing returns of feature relevance through the MRMR algorithm, plotting the cutoff points at the optimal number of features.

## Contributors
**Lee, Michelle** <br />
**Derbabian, Kyle** <br />
**Abott, Alexander** <br />
**Montemayor, Roy** <br />
**Khan, Fadeel** <br /> 

<hr style="border:2px">

*This project was a part of DSCI 435/COMP 449/COMP 549 as a part of D2K Lab, in Rice University.*
