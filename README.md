# Texas Alzheimers' Research Care and Consortium: Team TARCC 'F22

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

## Folder Structure
* `main.py` orchestrates the run and pipes data between other functions
* `config.py` sets global variables for parameters and settings for methods
* `data_utils.py` contains functions to load the data
* `preprocessing.py` contains the function block to preprocess the data and subset it to usable features
* `feature_selection.py`  contains functions to evaluate the importance of each feature in determining the most important biomarkers
* `model.py` contains functions to run the different models
* `visualization.py` contains functions that output visualizations of the data
* `environment.yml` defines the project environment

## Installation and Usage Instructions
1. Clone the repository
```
git clone https://github.com/RiceD2KLab/TARCC_F22.git
```
2. Install dependencies
```
pip install requirements.txt
```
## Data


## Data Science Pipeline

<img src="/TARCC_F22/Pipeline.png" alt="Alt text" title="Optional title">

> Preprocessing <br />
- Details here

> Feature Selection <br />
- Details here

> Modeling <br />
- Details here

> Results and Interpretation <br />
- Details here

## Contributors
**Cao, Angela** <br />
**Mudappatti, Anusha** <br />
**Gan, Wei Ren** <br />
**Prieto, Sophia** <br />
**Viswanathan, Tejeshwine** <br /> 

<hr style="border:2px">

*This project was a part of DSCI 535/COMP 449/COMP 549 as a part of D2K Lab, in Rice University.*
