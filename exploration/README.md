# Texas Alzheimers' Research Care and Consortium: Team TARCC 'S23

## Table of Contents

1. [Data Correlations](#data_correlations.py)
2. [Data Statistics](#data_statistics.py)
3. [Midterm Exploration](#midterm_exploration.py)

## data_correlations.py
This module has several functions for computing correlations between our features. The plot_correlations function is used to plot
the correlation matrix (like in main.ipynb). The dataset used must be cleaned before using this module.

## data_statistics.py
The data_statistics module performs initial exploration of our dataset. The plot_patientwise_errors function computes the patient-wise standard deviation
of the input features, and plots those standard deviations as boxplots.

## midterm_exploration.py
This module contains the plot_feature_against_diagnosis function, which was used for the plots in the midterm presentation/report.
This function plots three separate boxplots (one for each diagnosis level) of the input feature. The data should be cleaned and encoded before use.
