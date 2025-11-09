# ModernAIPT3
Predicting Abalone Age Using Machine Learning

Our environment setup via google colab

## Firstly we need to import libraries 

import pandas as pd
import numpy as np
import io
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin

pip install ucimlrepo

##  Loading the data into a dataframe

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

# check proportions of bins
print("Original dataset age_bin distribution:")
print(df["age_bin"].value_counts(normalize=True).sort_index())

print("\nTraining set distribution:")
train_bins = pd.qcut(y_train, q=5, labels=False, duplicates='drop')
print(train_bins.value_counts(normalize=True).sort_index())

print("\nTesting set distribution:")
test_bins = pd.qcut(y_test, q=5, labels=False, duplicates='drop')
print(test_bins.value_counts(normalize=True).sort_index())

# results show that the distributions are about the same, so the stratified sampling worked.


// Create preprocessing pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

num_attributes = ["Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight"] #numeric column names
cat_attributes = ["Sex"] #categorical column name

num_pipeline = Pipeline([('std_scaler', StandardScaler()) # normalisation using z-score ])

  cat_pipeline = Pipeline([ ('encoder', OneHotEncoder(handle_unknown="ignore", sparse_output=False)) # convert categorical to numeric ])#applies to numerics

  pre_pipeline = ColumnTransformer([ ("num", num_pipeline, num_attributes), #applies to numerics ("cat", cat_pipeline, cat_attributes),]) #applies to categorical 

# for SGDRegressor, we list the column names to check against coefficients results later
feature_names = X_train.columns
print(feature_names)

# also, we plot scatter plots to explore the relationship between the features and age
# using the training set.
import matplotlib.pyplot as plt
import numpy as np

# select numeric columns only
numeric_cols = X_train.select_dtypes(include=np.number).columns

for col in numeric_cols:
    plt.figure()
    plt.scatter(X_train[col], y_train, alpha=0.5)

# fit a linear line
coef = np.polyfit(X_train[col], y_train, deg=1)
poly1d_fn = np.poly1d(coef)
plt.plot(X_train[col], poly1d_fn(X_train[col]), color='red', linewidth=2)

 plt.xlabel(col)
 plt.ylabel("Age")
 plt.title(f"{col} vs. Age")
 plt.show()

# SGD Linear Regression
# Decision Tree Regressor
# Random Forest Regressor
# run these 3 models on training set
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# build a pipeline with each model
def build_pipeline(model):
    return Pipeline([('preprocessor', pre_pipeline), # preprocessing pipeline('model', model)])

# create instance of model and build pipelines
lr_model = SGDRegressor()
lr_pipeline = build_pipeline(lr_model)

dt_model = DecisionTreeRegressor(random_state=42)
dt_pipeline = build_pipeline(dt_model)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_pipeline = build_pipeline(rf_model)

# evaluate models on training set using cross validation
# by splitting data into multiple train/ test sets and averaging results
# we get a more reliable estimate of performance
from sklearn.model_selection import cross_val_score

def evaluate_model_custom(model):
  # this function returns evaluation metrics for each model passed in
  model_name = model.named_steps['model'].__class__.__name__
  scores = cross_val_score(model, X_train, y_train, \
                          scoring="neg_mean_squared_error", cv=5)
  #use MSE as evaluation metric, 5-fold cross validation
  rmse_scores = np.sqrt(-scores) #negates negative MSE scores back to positive
  print (f"{model_name} RMSE: {rmse_scores} \n Mean RMSE: {rmse_scores.mean()} \n Standard deviation RMSE: {rmse_scores.std()}")

evaluate_model_custom(lr_pipeline)
evaluate_model_custom(dt_pipeline)
evaluate_model_custom(rf_pipeline)

from sklearn.model_selection import RandomizedSearchCV, learning_curve
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.dummy import DummyRegressor
import matplotlib.pyplot as plt

# Fine Tune the model
# baseline RMSE stats
baseline = DummyRegressor(strategy='mean')
baseline.fit(X_train, y_train)
baseline_preds = baseline.predict(X_test)

print("Baseline RMSE:", np.sqrt(mean_squared_error(y_test, baseline_preds)))


# fine tuning by testing the other models and other hyperparameters
models = {
    "Linear Regression": SGDRegressor(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}

# sample range of values for these parameters
min_samples_split_values = np.arange(2, 20)  # from 2 to 19
min_samples_leaf_values = np.arange(1, 20)   # from 1 to 19

# add new hyperparameters
param_grid_new = {
    "Linear Regression": {
        'model__penalty': [None, 'l2', 'l1', 'elasticnet'], # regularization type
        'model__alpha': [0.0001, 0.001, 0.01], # regularization strength
        'model__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'], # learning rate schedule
        'model__eta0': [0.01, 0.1, 0.5], # initial learning rate
        'model__tol': [1e-3, 1e-4, 1e-5],  # stopping tolerance
        'model__max_iter': [2000, 5000, 10000] # no. of iterations
    },
    "Random Forest": {
        'model__criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],   # impurity measures
        'model__n_estimators': [50, 100, 200, 300, 500],
        'model__max_depth': [None, 10, 20, 30, 50, 70], # control max depth of tree for DT and RF, to prevent overfitting (Pre-pruning)
        'model__min_samples_split': min_samples_split_values, # min samples to split an internal node
        'model__min_samples_leaf': min_samples_leaf_values, # min samples required to be at a leaf node
        'model__bootstrap': [True, False] # add regularisation toggle
    },
    "Decision Tree": {
        'model__criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': min_samples_split_values,
        'model__min_samples_leaf': min_samples_leaf_values
    }
}

# grid search which also returns best model
for name, model in models.items():
    print("\n==========================")
    print(f"Training Model: {name}")
    print("==========================")

pipeline = build_pipeline(model) # build pipeline for model
param_grid = param_grid_new[name] # get correct grid for model

grid_search_new = RandomizedSearchCV(pipeline, param_grid, cv=3,
        return_train_score=True,
        scoring='neg_root_mean_squared_error')

# fit to training data
grid_search_new.fit(X_train, y_train)


 # print best model and best parameters
print("Best params:", grid_search_new.best_params_)
    print("Best model:", grid_search_new.best_estimator_)
    print("Best RMSE:", -grid_search_new.best_score_)

 # extract and print all results as a dataframe
results = pd.DataFrame(grid_search_new.cv_results_)

 # show only relevant columns and print all RMSEs
 display_cols = [col for col in results.columns if 'param_' in col or 'mean_test_score' in col]
 results_display = results[display_cols].copy()
results_display['mean_test_RMSE'] = -results_display['mean_test_score']

print("\nAll RMSEs for each parameter combination:")
   print(results_display.sort_values('mean_test_RMSE'))

 # ---- plot overfitting by varying model complexity
 if isinstance(model, SGDRegressor):
  alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
  train_rmse_list, val_rmse_list = [], []

 for a in alphas:
          model_tuned = build_pipeline(SGDRegressor(alpha=a, max_iter=5000, tol=1e-4))
          model_tuned.fit(X_train, y_train)

 y_pred_train = model_tuned.predict(X_train)
  y_pred_val = model_tuned.predict(X_test)

  train_rmse_list.append(np.sqrt(mean_squared_error(y_train, y_pred_train)))
 val_rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred_val)))



   plt.figure(figsize=(7,5))
      plt.semilogx(alphas, train_rmse_list, 'o-', label='Training RMSE')
      plt.semilogx(alphas, val_rmse_list, 'o-', label='Validation RMSE')
      plt.xlabel("Regularization Strength (alpha)")
      plt.ylabel("RMSE")
      plt.title("Regularization Curve (SGDRegressor)")
      plt.legend()
      plt.grid(True)
      plt.show()

if isinstance(model, DecisionTreeRegressor):
      depths = range(1, 25)
      train_rmse_list, val_rmse_list = [], []

 for d in depths:
          model_tuned = build_pipeline(DecisionTreeRegressor(max_depth=d, random_state=42))
          model_tuned.fit(X_train, y_train)

  y_pred_train = model_tuned.predict(X_train)
y_pred_val = model_tuned.predict(X_test)

  train_rmse_list.append(np.sqrt(mean_squared_error(y_train, y_pred_train)))
 val_rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred_val)))

 plt.figure(figsize=(7,5))
      plt.plot(depths, train_rmse_list, 'o-', label='Training RMSE')
      plt.plot(depths, val_rmse_list, 'o-', label='Validation RMSE')
      plt.xlabel("Tree Depth")
      plt.ylabel("RMSE")
      plt.title("Depth vs RMSE (Decision Tree)")
      plt.legend()
      plt.grid(True)
      plt.show()

 if isinstance(model, RandomForestRegressor):
      depths = [2, 4, 6, 8, 10, 15, 20, 30, 40, None]
      train_rmse_list, val_rmse_list = [], []

  for d in depths:
          rf = build_pipeline(RandomForestRegressor(max_depth=d, n_estimators=100, random_state=42))
          rf.fit(X_train, y_train)

 y_pred_train = rf.predict(X_train)
 y_pred_val = rf.predict(X_test)

 train_rmse_list.append(np.sqrt(mean_squared_error(y_train, y_pred_train)))
 val_rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred_val)))

  plt.figure(figsize=(7,5))
      plt.plot([str(d) for d in depths], train_rmse_list, 'o-', label='Training RMSE')
      plt.plot([str(d) for d in depths], val_rmse_list, 'o-', label='Validation RMSE')
      plt.xlabel("Max Depth")
      plt.ylabel("RMSE")
      plt.title("Depth vs RMSE (Random Forest)")
      plt.legend()
      plt.grid(True)
      plt.show()


# ---- final predictions: predict age for X_test
final_predictions = grid_search_new.best_estimator_.predict(X_test)

 # evaluation scores
 final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    final_mae = mean_absolute_error(y_test, final_predictions)
    final_r2 = r2_score(y_test, final_predictions)

 print("Final RMSE on Test Set:", final_rmse)
    print("Final MAE on Test Set:", final_mae)
    print("Final R-squared on Test Set:", final_r2)

# --- for linear model only: print coefficients and plot predicted vs actual scatter plot
 model_in_pipeline = grid_search_new.best_estimator_.named_steps['model']
    if isinstance(model_in_pipeline, SGDRegressor):
        print("Model coefficients:", model_in_pipeline.coef_)
        print("Model intercept:", model_in_pipeline.intercept_)

  plt.figure(figsize=(6,6))
        plt.scatter(y_test, final_predictions, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("Actual Age")
        plt.ylabel("Predicted Age")
        plt.title(f"Predicted vs Actual ({name})")
        plt.show()



# feature importance
# shows the relative importance of each feature used by the model. higher value = more important feature which
# contributed more to model's decisions

# extract feature importances from randomforest model (best one from gridsearchCV above)
best_rf = grid_search_new.best_estimator_
importances = best_rf.named_steps['model'].feature_importances_
feature_names = best_rf.named_steps['preprocessor'].get_feature_names_out() #feature names

# get top 10 feature importances (ascending order)
top_indices = np.argsort(importances)[-10:]
top_features = feature_names[top_indices]
top_importances = importances[top_indices]

# plot bar graph
plt.figure(figsize=(10, 6))
plt.barh(top_features, top_importances)
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()




