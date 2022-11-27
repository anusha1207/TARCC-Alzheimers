import pandas as pd
import numpy as np
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
# import feature_selection as fs
import preprocessing as pb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression #linear_model.LogisticRegression (setting multi_class=”multinomial”)
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier

def get_data(non_genetic_df):

  df_combined = pb.preprocessing(non_genetic_df)

  #remove patient ID while doing feature selection
  df_features_comb = df_combined.drop(['PATID'], axis=1)
  ##### Split features and target variable #####
  X_comb = df_features_comb.drop(['P1_PT_TYPE'], axis=1, inplace = False)
  y_comb = df_features_comb['P1_PT_TYPE']

  return df_features_comb, X_comb, y_comb

def get_data(final_df):
      
  features = final_df.loc[:, final_df.columns != 'P1_PT_TYPE']
  y = final_df['P1_PT_TYPE']

  # standard scaling
  scaler = StandardScaler()
  X = scaler.fit_transform(features)

  print(features)

  # manually split: 80% train, 10% validation, 10% test sets
  X_train = features[:math.ceil(0.8*final_df.shape[0])+1]
  y_train = y[:math.ceil(0.8*final_df.shape[0])+1]
  X_val = features[math.ceil(0.8*final_df.shape[0])+1:-math.floor(0.1*final_df.shape[0])]
  y_val = y[math.ceil(0.8*final_df.shape[0])+1:-math.floor(0.1*final_df.shape[0])]
  X_test = features[-math.floor(0.1*final_df.shape[0]):]
  y_test = y[-math.floor(0.1*final_df.shape[0]):]

  return X_train, y_train, X_val, y_val, X_test, y_test


def ml_prep(final_df):
  
  features = final_df.loc[:, final_df.columns != 'P1_PT_TYPE']
  y = final_df['P1_PT_TYPE']

  # standard scaling
  scaler = StandardScaler()
  X = scaler.fit_transform(features)

  print(features)

  # manually split: 80% train, 10% validation, 10% test sets
  X_train = features[:math.ceil(0.8*final_df.shape[0])+1]
  y_train = y[:math.ceil(0.8*final_df.shape[0])+1]
  X_val = features[math.ceil(0.8*final_df.shape[0])+1:-math.floor(0.1*final_df.shape[0])]
  y_val = y[math.ceil(0.8*final_df.shape[0])+1:-math.floor(0.1*final_df.shape[0])]
  X_test = features[-math.floor(0.1*final_df.shape[0]):]
  y_test = y[-math.floor(0.1*final_df.shape[0]):]

  return X_train, y_train, X_val, y_val, X_test, y_test

non_genetic_df = pd.read_csv('20220916 updated TARCC Data for Dr Broussard.csv', low_memory=False)

# pre-process the raw data
df_features_comb, X_comb, y_comb = get_data(non_genetic_df)
    
# retrieve pickled combined features list
combined_features_list = pickle.load(open("pickled_combined_features_list.pkl", "rb" ))

# getting only top features after feature selection
final_features_df = df_features_comb[combined_features_list]
# merge the dataset for machine learning model
frames = [final_features_df, y_comb]
final_df = pd.concat(frames, axis=1)

# perform train_test_split
X_train, y_train, X_val, y_val, X_test, y_test = ml_prep(final_df)

def black_box_lgbm_classifier(n_estimator, max_depth, colsample_bytree, num_leaves):
  assert type(n_estimator) == int
  assert type(max_depth) == int
  assert type(num_leaves) == int
  clf_lgbm = lgbm.LGBMClassifier(n_estimators=n_estimator,
                                 max_depth=max_depth,
                                 colsample_bytree=colsample_bytree,
                                 num_leaves=num_leaves)
  clf_lgbm.fit(X_train, y_train)
  y_score = clf_lgbm.predict_proba(X_val)[:, 1]
  return roc_auc_score(y_val, y_score)

def lgbm_classifier_int_params(param_one, param_two, colsample_bytree, param_four):
  n_estimator=int(param_one)
  max_depth=int(param_two)
  num_leaves=int(param_four)
  return black_box_lgbm_classifier(n_estimator, max_depth, colsample_bytree, num_leaves)

def lgbm_optimize(iterations):
  params = {
      'param_one': [100, 1000],
      'param_two': [1, 14],
      'colsample_bytree': [0.5, 1.0],
      'param_four': [2, 200]
  }
  optimizer = BayesianOptimization(f=lgbm_classifier_int_params,
                                   pbounds=params, random_state=42)
  optimizer.maximize(n_iter=iterations)
  max_params = optimizer.max["params"]
  best_params = {
      "n_estimators": int(max_params['param_one']),
      "max_depth": int(max_params['param_two']),
      "colsample_bytree": max_params["colsample_bytree"],
      "num_leaves": int(max_params['param_four'])
  }
  return best_params

def black_box_random_forest(n_estimators, max_depth, min_samples_leaf, min_samples_split):
  assert type(n_estimators) == int
  assert type(max_depth) == int
  assert type(min_samples_leaf) == int
  assert type(min_samples_split) == int
  clf_rf = RandomForestClassifier(n_estimators=n_estimators,
                                  max_depth=max_depth,
                                  min_samples_leaf=min_samples_leaf,
                                  min_samples_split=min_samples_split)
  clf_rf.fit(X_train, y_train)
  y_score = clf_rf.predict_proba(X_val)[:, 1]
  return roc_auc_score(y_val, y_score)

def random_forest_int_params(param_one, param_two, param_three, param_four):
  n_estimators = int(param_one)
  max_depth = int(param_two)
  min_samples_leaf = int(param_three)
  min_samples_split = int(param_four)
  print(n_estimators)
  return black_box_random_forest(n_estimators, max_depth, min_samples_leaf, min_samples_split)

def random_forest_optimize(iterations):
  pbounds = {
    'param_one': (100, 500),
    'param_two': (5, 20),
    'param_three': (1, 5),
    'param_four': (2, 6)
  }
  optimizer = BayesianOptimization(f=random_forest_int_params,
                                   pbounds=pbounds, random_state=42)
  optimizer.maximize(n_iter=iterations)
  max_params = optimizer.max["params"]
  best_params = {
      "n_estimators": int(max_params["param_one"]),
      "max_depth": int(max_params["param_two"]),
      "min_samples_leaf": int(max_params["param_three"]),
      "min_samples_split": int(max_params["param_four"])
  }
  return best_params

random_forest_optimize(iterations=5)

def black_box_extra_trees_classifier(n_estimator, max_depth):
  assert type(n_estimator) == int
  assert type(max_depth) == int
  clf_et = ExtraTreesClassifier(n_estimators=n_estimator,
                                max_depth=max_depth)
  clf_et.fit(X_train, y_train)
  y_score = clf_et.predict_proba(X_val)[:, 1]
  return roc_auc_score(y_val, y_score)

def extra_trees_classifier_int_params(param_one, param_two):
  n_estimator=int(param_one)
  max_depth=int(param_two)
  return black_box_extra_trees_classifier(n_estimator, max_depth)

def extra_trees_optimize(iterations):
  params = {
    'param_one': [100, 1000],
    'param_two': [1, 14]
  }
  optimizer = BayesianOptimization(f=extra_trees_classifier_int_params,
                                   pbounds=params, random_state=42)
  optimizer.maximize(n_iter=iterations)
  max_params = optimizer.max["params"]
  best_params = {
      "n_estimators": int(max_params["param_one"]),
      "max_depth": int(max_params["param_two"])
  }
  return best_params

extra_trees_optimize(iterations=5)

def black_box_extra_trees_classifier(n_estimator, max_depth):
  assert type(n_estimator) == int
  assert type(max_depth) == int
  clf_et = ExtraTreesClassifier(n_estimators=n_estimator,
                                max_depth=max_depth)
  clf_et.fit(X_train, y_train)
  y_score = clf_et.predict_proba(X_val)[:, 1]
  return roc_auc_score(y_val, y_score)

def extra_trees_classifier_int_params(param_one, param_two):
  n_estimator=int(param_one)
  max_depth=int(param_two)
  return black_box_extra_trees_classifier(n_estimator, max_depth)

def extra_trees_optimize(iterations):
  params = {
    'param_one': [100, 1000],
    'param_two': [1, 14]
  }
  optimizer = BayesianOptimization(f=extra_trees_classifier_int_params,
                                   pbounds=params, random_state=42)
  optimizer.maximize(n_iter=iterations)
  max_params = optimizer.max["params"]
  best_params = {
      "n_estimators": int(max_params["param_one"]),
      "max_depth": int(max_params["param_two"])
  }
  return best_params

extra_trees_optimize(iterations=5)

def black_box_xgb_classifier(n_estimator, max_depth, colsample_bytree, gamma):
  assert type(n_estimator) == int
  assert type(max_depth) == int
  clf_et = XGBClassifier(n_estimators=n_estimator,
                         max_depth=max_depth,
                         colsample_bytree=colsample_bytree,
                         gamma=gamma)
  clf_et.fit(X_train, y_train)
  y_score = clf_et.predict_proba(X_val)[:, 1]
  return roc_auc_score(y_val, y_score)

def xgb_classifier_int_params(param_one, param_two, colsample_bytree, gamma):
  max_iter=int(param_one)
  max_depth=int(param_two)
  return black_box_xgb_classifier(max_iter, max_depth, colsample_bytree, gamma)

def xgb_optimize(iterations):
  params = {
    'param_one': [100, 1000],
    'param_two': [1, 14],
    'colsample_bytree': [0.5, 1.0],
    'gamma': [0.5, 1.0]
  }
  optimizer = BayesianOptimization(f=xgb_classifier_int_params,
                                   pbounds=params, random_state=42)
  optimizer.maximize(n_iter=iterations)
  max_params = optimizer.max["params"]
  best_params = {
      "n_estimators": int(max_params["param_one"]),
      "max_depth": int(max_params["param_two"]),
      "colsample_bytree": max_params["colsample_bytree"],
      "gamma": max_params["gamma"]
  }
  return best_params 

xgb_optimize(iterations=5)

def black_box_catboost_classifier(depth, border_count, learning_rate, l2_leaf_reg):
  assert type(depth) == int
  assert type(border_count) == int
  assert type(l2_leaf_reg) == int
  clf_cb = CatBoostClassifier(depth=depth,
                              border_count=border_count,
                              learning_rate=learning_rate,
                              l2_leaf_reg=l2_leaf_reg)
  clf_cb.fit(X_train, y_train)
  y_score = clf_cb.predict_proba(X_val)[:, 1]
  return roc_auc_score(y_val, y_score)

def catboost_classifier_int_params(param_one, param_two, learning_rate, param_four):
  depth=int(param_one)
  border_count=int(param_two)
  l2_leaf_reg=int(param_four)
  return black_box_catboost_classifier(depth, border_count, learning_rate, l2_leaf_reg)

def catboost_optimize(iterations):
  params = {
    'param_one': [1, 6],
    'param_two': [32, 255],
    'learning_rate': [-5.0, -2],
    'param_four': [3, 8]
  }
  optimizer = BayesianOptimization(f=catboost_classifier_int_params,
                                   pbounds=params, random_state=42)
  optimizer.maximize(n_iter=iterations)
  max_params = optimizer.max["params"]
  best_params = {
      "depth": int(max_params["param_one"]),
      "border_count": int(max_params["param_two"]),
      "learning_rate": max_params["learning_rate"],
      "l2_leaf_reg": int(max_params["param_four"])
  }
  return best_params

catboost_optimize(iterations=5)
