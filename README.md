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
- [Scikit Learn](https://pypi.org/project/scikit-learn/)
- [Scipy Stats](https://pypi.org/project/scipy/)


## Folder Structure
* data: The data folder contains the csv file of our raw input (TARC_data.csv) 
* exploration: This contains initial data exploration files.
* preprocessing: This folder contains a preprocessing file which cleans our data's missing values and features.


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


![Pipeline](https://user-images.githubusercontent.com/97485268/198148374-fc9760c7-bf3f-4b82-8a7c-b83b73d82556.png)

### PREPROCESSING <br />

The data pertaining to blood and protein bio-markers along with mental and physical attributes is subsetted to perform a line of analysis which helps us gauge the most important features. Upon doing this, the resulting data-set consisted of 563 patients with a total of 217 features. After preprocessing, the patient database is divided broadly into two classes based on the diagnosis, whether they have AD or not.
<br />


## Contributors
**Lee, Michelle** <br />
**Derbabian, Kyle** <br />
**Abott, Alexander** <br />
**Montemayor, Roy** <br />
**Khan, Fadeel** <br /> 

<hr style="border:2px">

*This project was a part of DSCI 435/COMP 449/COMP 549 as a part of D2K Lab, in Rice University.*
