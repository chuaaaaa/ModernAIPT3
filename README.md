#### **Project Title: Predicting Abalone Age Using Machine Learning: Smarter Harvesting for a Sustainable Future**



This repository contains the workflow for predicting abalone age using physical measurements from the UCI Abalone dataset. Accurate age prediction supports fisheries and helps manage marine populations sustainably.



**Table of Contents**

1\. Importing required libraries

2\. Loading the data into a dataframe

3\. Exploring the data

4\. Data Visualization

5\. Pre-processing the data

6\. Select and train models

7\. Fine Tune the model/ Final evaluation



Dataset: UCI Abalone Dataset (https://archive.ics.uci.edu/dataset/1/abalone)

Features: Length, Diameter, Height, Whole weight, Shucked weight, Viscera weight, Shell weight

Target: Age (rings + 1.5 years)



**Usage**

* Clone the repository or download the ipynb file
* Open AI\_groupPT3.ipynb in Jupyter Notebook or Lab
* Run all cells sequentially



**Environment Setup \& Dependencies**

* Google Colab with Python 3.12
* Import/ install all the following libraries: pandas, numpy, io, matplotlib, seaborn, sklearn
* Important! Install the repository that contains the abalone dataset: pip install ucimlrepo



**Key highlights**

* Data visualisation: Using heatmap, scatter plots, histograms
* Pre-processing: Remove outliers, stratified sampling
* Models: Linear Regression, Decision Tree, Random Forest
* Using 5 fold cross validation, RandomizedSearchCV and finetuning of hyperparameters for modelling
* Metrics: RMSE, R^2
* Learning curves let us plot overfitting by varying model complexity



**Results**

This project explored 3 machine learning models to predict abalone age. While linear regression provided a baseline, it underperformed slightly compared to tree-based random forest. Random forest achieved the best overall performance with the lowest RMSE of 1.99 and best R^2 of 0.599. While abalone growth patterns are largely linear, modest non-linear interactions exist that ensemble models like Random Forest can better capture, making it the best for age prediction. It offers the most balanced trade-off between bias and variance. Future work could include expanding the dataset or applying advanced ensemble methods like XGBoost to enhance accuracy.





