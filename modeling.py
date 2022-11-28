import preprocessing as pp
import feature_selection as fs
import pandas as pd
import math
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression #linear_model.LogisticRegression (setting multi_class=”multinomial”)
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import lightgbm as lgbm
import copy
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, classification_report, precision_score, recall_score, roc_auc_score, f1_score, fbeta_score
from sklearn import metrics
from bayes_opt import BayesianOptimization 

def get_data(non_genetic_df):

  df_combined = pp.preprocessing(non_genetic_df)

  #remove patient ID while doing feature selection
  df_features_comb = df_combined.drop(['PATID'], axis=1)
  ##### Split features and target variable #####
  X_comb = df_features_comb.drop(['P1_PT_TYPE'], axis=1, inplace = False)
  y_comb = df_features_comb['P1_PT_TYPE']

  return df_features_comb, X_comb, y_comb

def ml_prep(final_df):
  
  features = final_df.loc[:, final_df.columns != 'P1_PT_TYPE']
  y = final_df['P1_PT_TYPE']

  # standard scaling
  scaler = StandardScaler()
  X = scaler.fit_transform(features)

  # manually split: 80% train, 10% validation, 10% test sets
  X_train = features.iloc[:math.ceil(0.8*final_df.shape[0])+1]
  y_train = y.iloc[:math.ceil(0.8*final_df.shape[0])+1]
  X_val = features.iloc[math.ceil(0.8*final_df.shape[0])+1:-math.floor(0.1*final_df.shape[0])]
  y_val = y.iloc[math.ceil(0.8*final_df.shape[0])+1:-math.floor(0.1*final_df.shape[0])]
  X_test = features.iloc[-math.floor(0.1*final_df.shape[0]):]
  y_test = y.iloc[-math.floor(0.1*final_df.shape[0]):]
  
  return X_train, y_train, X_val, y_val, X_test, y_test

# Bayesian Tuning Methods
# Light Gradient Boosting (LGBM)
# LGBM doesn't work when running terminal
"""
def lgbm_optimize(iterations, X_train, y_train, X_val, y_val):
  def black_box_lgbm_classifier(n_estimator, max_depth, colsample_bytree, num_leaves):
    assert type(n_estimator) == int
    assert type(max_depth) == int
    assert type(num_leaves) == int
    clf_lgbm = lgbm.LGBMClassifier(n_estimators=n_estimator,
                                   max_depth=max_depth,
                                   colsample_bytree=colsample_bytree,
                                   num_leaves=num_leaves,
                                   random_state=42)
    clf_lgbm.fit(X_train, y_train)
    y_score = clf_lgbm.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_score)

  def lgbm_classifier_int_params(param_one, param_two, colsample_bytree, param_four):
    n_estimator=int(param_one)
    max_depth=int(param_two)
    num_leaves=int(param_four)
    return black_box_lgbm_classifier(n_estimator, max_depth, colsample_bytree, num_leaves)
  
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

def lgbm_optimize_classifier(params):
  return lgbm.LGBMClassifier(n_estimators=params["n_estimators"],
                             max_depth=params["max_depth"],
                             colsample_bytree=params["colsample_bytree"],
                             num_leaves=params["num_leaves"],
                             random_state=42)
"""

# Random Forests
def random_forest_optimize(iterations, X_train, y_train, X_val, y_val):
  def black_box_random_forest(n_estimators, max_depth, min_samples_leaf, min_samples_split):
    assert type(n_estimators) == int
    assert type(max_depth) == int
    assert type(min_samples_leaf) == int
    assert type(min_samples_split) == int
    clf_rf = RandomForestClassifier(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    min_samples_leaf=min_samples_leaf,
                                    min_samples_split=min_samples_split,
                                    random_state=42)
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

  pbounds = {
    'param_one': [100, 500],
    'param_two': [5, 20],
    'param_three': [1, 5],
    'param_four': [2, 6]
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

def random_forest_optimize_classifier(params):
  return RandomForestClassifier(n_estimators=params["n_estimators"],
                                max_depth=params["max_depth"],
                                min_samples_leaf=params["min_samples_leaf"],
                                min_samples_split=params["min_samples_split"],
                                random_state=42)

# Logistic Regression
def logistic_regression_optimize(iterations, X_train, y_train, X_val, y_val):
  def black_box_logistic_regression(class_weight, max_iter, solver):
    assert type(max_iter) == int
    clf_lr = LogisticRegression(class_weight=class_weight,
                                max_iter=max_iter,
                                solver=solver,
                                random_state=42)
    clf_lr.fit(X_train, y_train)
    y_score = clf_lr.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_score)

  def logistic_regression_int_params(param_one, param_two, param_three):
    class_weight = None
    class_weight_index = int(param_one)
    if class_weight_index == 0:
      class_weight = "balanced"
    else:
      class_weight = None
    max_iter=int(param_two)
    solver = None
    solver_index = int(param_three)
    if solver_index == 0:
      solver = "newton-cg"
    elif solver_index == 1:
      solver = "lbfgs"
    elif solver_index == 2:
      solver = "liblinear"
    elif solver_index == 3:
      solver = "sag"
    elif solver_index == 4:
      solver = "saga"
    return black_box_logistic_regression(class_weight, max_iter, solver)

  pbounds = {
    'param_one': [0, 1],
    'param_two': [1, 14],
    'param_three': [0, 4]
  }
  optimizer = BayesianOptimization(f=logistic_regression_int_params,
                                   pbounds=pbounds, random_state=42)
  optimizer.maximize(n_iter=iterations)
  max_params = optimizer.max["params"]
  class_weight = None
  class_weight_index = int(max_params["param_one"])
  if class_weight_index == 0:
    class_weight = "balanced"
  else:
    class_weight = None
  solver = None
  solver_index = int(max_params["param_three"])
  if solver_index == 0:
    solver = "newton-cg"
  elif solver_index == 1:
    solver = "lbfgs"
  elif solver_index == 2:
    solver = "liblinear"
  elif solver_index == 3:
    solver = "sag"
  elif solver_index == 4:
    solver = "saga"
  best_params = {
      "class_weight": class_weight,
      "max_iter": int(max_params["param_two"]),
      "solver": solver
  }
  return best_params

def logistic_regression_optimize_classifier(params):
  return LogisticRegression(class_weight=params["class_weight"],
                            max_iter=params["max_iter"],
                            solver=params["solver"],
                            random_state=42)

# Extra Trees
def extra_trees_optimize(iterations, X_train, y_train, X_val, y_val):
  def black_box_extra_trees_classifier(n_estimator, max_depth):
    assert type(n_estimator) == int
    assert type(max_depth) == int
    clf_et = ExtraTreesClassifier(n_estimators=n_estimator,
                                  max_depth=max_depth,
                                  random_state=42)
    clf_et.fit(X_train, y_train)
    y_score = clf_et.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_score)

  def extra_trees_classifier_int_params(param_one, param_two):
    n_estimator=int(param_one)
    max_depth=int(param_two)
    return black_box_extra_trees_classifier(n_estimator, max_depth)

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

def extra_trees_optimize_classifier(params):
  return ExtraTreesClassifier(n_estimators=params["n_estimators"],
                              max_depth=params["max_depth"],
                              random_state=42)

# eXtra Gradient Boosting (XGB)
# XGBoost classifier doesn't work when running with terminal.
"""
def xgb_optimize(iterations, X_train, y_train, X_val, y_val):
  def black_box_xgb_classifier(n_estimator, max_depth, colsample_bytree, gamma):
    assert type(n_estimator) == int
    assert type(max_depth) == int
    clf_et = XGBClassifier(n_estimators=n_estimator,
                           max_depth=max_depth,
                           colsample_bytree=colsample_bytree,
                           gamma=gamma,
                           random_state=42)
    clf_et.fit(X_train, y_train)
    y_score = clf_et.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_score)

  def xgb_classifier_int_params(param_one, param_two, colsample_bytree, gamma):
    max_iter=int(param_one)
    max_depth=int(param_two)
    return black_box_xgb_classifier(max_iter, max_depth, colsample_bytree, gamma)

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

def xgb_optimize_classifier(params):
  return XGBClassifier(n_estimators=params["n_estimators"],
                       max_depth=params["max_depth"],
                       colsample_bytree=params["colsample_bytree"],
                       gamma=params["gamma"],
                       random_state=42)
"""

# Categorical Boosting (Catboost)
def catboost_optimize(iterations, X_train, y_train, X_val, y_val):
  def black_box_catboost_classifier(depth, border_count, learning_rate, l2_leaf_reg):
    assert type(depth) == int
    assert type(border_count) == int
    assert type(l2_leaf_reg) == int
    clf_cb = CatBoostClassifier(depth=depth,
                                border_count=border_count,
                                learning_rate=learning_rate,
                                l2_leaf_reg=l2_leaf_reg,
                                random_state=42)
    clf_cb.fit(X_train, y_train)
    y_score = clf_cb.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_score)

  def catboost_classifier_int_params(param_one, param_two, learning_rate, param_four):
    depth=int(param_one)
    border_count=int(param_two)
    l2_leaf_reg=int(param_four)
    return black_box_catboost_classifier(depth, border_count, learning_rate, l2_leaf_reg)

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

def catboost_optimize_classifier(params):
  return CatBoostClassifier(depth=params["depth"],
                            border_count=params["border_count"],
                            learning_rate=params["learning_rate"],
                            l2_leaf_reg=params["l2_leaf_reg"],
                            random_state=42)
                            
def model_results(df, X_train, X_test, y_train, y_test, classifier_func, model_name):

  # perform evaluation on various models
  for model in range(len(classifier_func[:])):
    classifier_func[model].fit(X_train, y_train)

    print('-'*150)
    print(f'Evaluation for {model_name[model]}: ')
    y_pred = classifier_func[model].predict(X_test)
    evaluation(y_test, y_pred)
    
    # plot ROC curve; will have a separate function for this once we have all models with best parameters gathered
    metrics.plot_roc_curve(classifier_func[model], X_test, y_test, pos_label=1)
    plt.savefig(f'results/{model_name[model]}_ROC.pdf', format="pdf", bbox_inches="tight")
    plt.show()
    print() 

# Evaluation metrics
def evaluation(y_test, y_pred):

    # Accuracy classification score
    score = round(accuracy_score(y_test, y_pred), 4)
    print(f'Accuracy Score: {score*100}%')

    # precision score
    binary_averaged_precision = precision_score(y_test, y_pred, average = 'binary')
    print("Binary-Averaged Precision score: {:.4f}".format(binary_averaged_precision))

    # recall score
    binary_averaged_recall = recall_score(y_test, y_pred, average = 'binary')
    print("Binary-Averaged Recall score: {:.4f}".format(binary_averaged_recall))

    # f1 score
    binary_averaged_f1 = f1_score(y_test, y_pred, average = 'binary')
    print("Binary-Averaged F1 score: {:.4f}".format(binary_averaged_f1))

    # f beta score
    binary_f_beta = fbeta_score(y_test, y_pred, average='binary', beta=2)
    print("Binary-Averaged F-Beta score: {:.4f}".format(binary_f_beta))

    # Receiver Operating Characteristic Area Under Curve (ROC_AUC) Score
    # default average is 'macro'
    roc_auc_bi = roc_auc_score(y_test, y_pred, average = 'macro')
    print("ROC_AUC Score: {:.4f}".format(roc_auc_bi))

    # required to convert labels for auc_precision_recall
    # converted from 1 = AD and 2 = Control to 1 = AD and 0 = Control
    # convert y_test
    prc_y_test = copy.deepcopy(y_test)
    prc_y_test[prc_y_test==1] = 1
    prc_y_test[prc_y_test==2] = 0

    # convert y_pred
    prc_y_pred = copy.deepcopy(y_pred)
    prc_y_pred[prc_y_pred==1] = 1
    prc_y_pred[prc_y_pred==2] = 0

    # Compute Area Under the Precision Recall Curve; default average is 'macro'
    precision, recall, thresholds = precision_recall_curve(prc_y_test, prc_y_pred)
    auc_precision_recall = auc(recall, precision)
    print("AUPRC is: {:.4f}".format(auc_precision_recall))

    # Get specificity from classification report; include 'binary' results
    class_report = classification_report(y_test, y_pred, labels=[1,2])
    print("Classification Report: ")
    print(class_report)

    # plot the confusion matrix
    plt.figure(figsize = (18,8))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot = True, xticklabels = y_test.unique(), yticklabels = y_test.unique(), cmap = 'summer')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()


# Note: Use micro-average if classes are imbalance


def model_main(non_genetic_df):

  # pre-process the raw data
  df_features_comb, X_comb, y_comb = get_data(non_genetic_df)
    
  # retrieve pickled combined features list
  combined_features_list = pickle.load(open("data/pickled_combined_features_list.pkl", "rb" ))

  # getting only top features after feature selection
  final_features_df = df_features_comb[combined_features_list]
  # merge the dataset for machine learning model
  frames = [final_features_df, y_comb]
  final_df = pd.concat(frames, axis=1)

  # perform train_test_split
  X_train, y_train, X_val, y_val, X_test, y_test = ml_prep(final_df)


  # list of classifier functions
  classifier_func = [#lgbm.LGBMClassifier(colsample_bytree=0.46053366496668136,num_leaves= 122, random_state=42),
                    RandomForestClassifier(n_estimators=900, max_depth=8, random_state=42), 
                    #XGBClassifier(colsample_bytree= 0.840545160958208, gamma= 0.3433699189306628, max_depth= 2),                    
                    LogisticRegression(class_weight='balanced', max_iter=200, random_state=42, solver='sag'),
                    ExtraTreesClassifier(n_estimators=500, max_depth=3),
                    CatBoostClassifier(random_state=42)]  

  # list of classifier names
  model_name= [#'Light Gradient Boosting Method',
              'Random Forest', 
              #'eXtreme Gradient Boosting',
              'Logistic Regression', 
              'Extra Trees',
              'Categorical Boosting']
              
  # list of optimized classifier functions
  model_optimizers = [#lgbm_optimize_classifier(lgbm_optimize(500, X_train, y_train, X_val, y_val)),
                      random_forest_optimize_classifier(random_forest_optimize(500, X_train, y_train, X_val, y_val)),
                      #xgb_optimize_classifier(xgb_optimize(500, X_train, y_train, X_val, y_val)),
                      logistic_regression_optimize_classifier(logistic_regression_optimize(500, X_train, y_train, X_val, y_val)),
                      extra_trees_optimize_classifier(extra_trees_optimize(500, X_train, y_train, X_val, y_val)),
                      catboost_optimize_classifier(catboost_optimize(500, X_train, y_train, X_val, y_val))]

  # evaluate performance and feature importance for each algorithm
  # model_results(final_df, X_train, X_test, y_train, y_test, classifier_func, model_name)
  model_results(final_df, X_train, X_test, y_train, y_test, model_optimizers, model_name)

  

