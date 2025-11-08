# ModernAIPT3
Predicting Abalone Age Using Machine Learning

Our environment setup via google colab

Firstly we need to import libraries 
import pandas as pd
import numpy as np
import io
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin

// install package
pip install ucimlrepo

// Loading the data into a dataframe

from ucimlrepo import fetch_ucirepo
// get dataset import into google colab
fetch dataset from https://archive.ics.uci.edu/dataset/1/abalone
abalone = fetch_ucirepo(id=1)


X = abalone.data.features
y = abalone.data.targets
df = pd.concat([X, y], axis=1)
print(df.head())


df['Age'] = df.Rings + 1.5
df = df.drop(columns='Rings')
df.head()

//Exploring the data 

df.info()
df.dtypes.value_counts()
df.describe()
missing_values = df.isnull().sum()  
missing_ratio = (df.isnull().sum()/df.shape[0])*100
df.nunique().to_frame("# of unique values")

// Data Visualization
numerical_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(12, 6))
sns.heatmap(numerical_df.corr(),
            cmap = 'coolwarm',
            fmt = '.2f',
            linewidths = 2,
            annot = True)

df.hist(bins=50, figsize=(20,10), log=True)

outlier_rows = df[df["Height"] > 1]
print(outlier_rows)

// drop row by index
df = df.drop(2051)
df.shape[0] #expected 4177 - 1 = 4176

// check that there are no more outlier rows after removal
outlier_rows = df[df["Height"] > 1]
print(outlier_rows)


Pre-processing the data

// Preparing data by splitting into training and testing sets using stratified sampling
from sklearn.model_selection import StratifiedShuffleSplit

// bin continuous attribute price
df["age_bin"] = pd.qcut(df["Age"], q=5, labels=False)

// using 20% for testing, and random seed 42 for reproducability
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

// .split() generates indices / row numbers
for train_index, test_index in split.split(df, df["age_bin"]):
    strat_train = df.iloc[train_index].copy() #full training set
    strat_test = df.iloc[test_index].copy() #full test set

// drop age_bin
strat_train.drop(columns=["age_bin"], inplace=True)
strat_test.drop(columns=["age_bin"], inplace=True)

// split into features and class
X_train = strat_train.drop(columns=["Age"])
y_train = strat_train["Age"]
X_test = strat_test.drop(columns=["Age"])
y_test = strat_test["Age"]

print("X_train shape:", X_train.shape) # (7998,14) shows 7998 rows, 14 columns
print("X_test shape:", X_test.shape) # (2000,14) shows 2000 rows, 14 columns
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)




// Create preprocessing pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

num_attributes = ["Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight"] #numeric column names
cat_attributes = ["Sex"] #categorical column name

num_pipeline = Pipeline([('std_scaler', StandardScaler()) # normalisation using z-score ])

  cat_pipeline = Pipeline([ ('encoder', OneHotEncoder(handle_unknown="ignore", sparse_output=False)) # convert categorical to numeric ])

  pre_pipeline = ColumnTransformer([ ("num", num_pipeline, num_attributes), #applies to numerics ("cat", cat_pipeline, cat_attributes),]) #applies to categorical 













