from script import preprocessing as pp
from script import feature_selection as fs
import pandas as pd
import math
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pickle
import lightgbm as lgbm
import copy
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, classification_report, precision_score, recall_score, roc_auc_score, f1_score, fbeta_score
from sklearn import metrics
from bayes_opt import BayesianOptimization 

def get_data(non_genetic_df):
  """
  This function retrieves and splits data
  INPUTS:
    non_genetic_df -- <pd.DataFrame> raw biological dataset
  OUTPUTS:
      df_features -- <pd.DataFrame> features
      y -- <pd.Series> target variable
  """

  df= pp.preprocessing(non_genetic_df)

  #remove patient ID while doing feature selection
  df_features = df.drop(['PATID'], axis=1)
  ##### Split features and target variable #####
  y = df_features['P1_PT_TYPE']

  return df_features, y

def ml_prep(final_df):
  """
  This function splits data into test, train, and validation sets
  INPUTS:
    final_df -- <pd.DataFrame> dataset of selected features
  OUTPUTS:
      X_train -- <pd.DataFrame> training set features
      y_train -- <pd.Series> training set target variable
      X_val -- <pd.DataFrame> validation set features
      y_val -- <pd.Series> validation set target variable
      X_test -- <pd.DataFrame> testing set features
      y_test -- <pd.Series> testing set target variable
  """  
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
                            
def model_results(df, X_train, X_test, y_train, y_test, classifier_func, model_name):
  """
  This function evaluates each model's ROC curve
  INPUTS:
    df -- <pd.DataFrame> dataset
    X_train -- <pd.DataFrame> training set features
    y_train -- <pd.Series> training set target variable
    X_val -- <pd.DataFrame> validation set features
    y_val -- <pd.Series> validation set target variable
    classifier_func -- model 
    model_name -- <str> name of model
  OUTPUTS:
    ROC curve plot
  """
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
    """
    This function evaluates based on various scores
    INPUTS:
      y_test -- <pd.Series> target variable testing set
      y_pred -- <pd.Series> target variable predictions
    OUTPUTS:
      ROC curve plot
    """
   
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


def model_main(non_genetic_df):
  """
  This function evaluates each model with best parameters
  INPUTS:
    non_genetic_df -- <pd.DataFrame> raw biological dataset
  OUTPUT: model results
  """
  # pre-process the raw data
  df_features, y = get_data(non_genetic_df)
    
  # retrieve pickled combined features list
  combined_features_list = pickle.load(open("data/pickled_combined_features_list.pkl", "rb" ))

  # getting only top features after feature selection
  final_features_df = df_features[combined_features_list]
  # merge the dataset for machine learning model
  frames = [final_features_df, y]
  final_df = pd.concat(frames, axis=1)

  # perform train_test_split
  X_train, y_train, X_val, y_val, X_test, y_test = ml_prep(final_df)

  # load pickled models
  lgbm_model = pickle.load(open('script/lgbm_model_f_beta_7377%.pkl','rb'))
  catboost_model = pickle.load(open('script/catboost_model_7500fb%.pkl','rb'))
  rf_model = pickle.load(open('script/rf_model_f_beta_7937%.pkl','rb'))
  xgb_model = pickle.load(open('script/xgb_model_f_beta_7500%.pkl','rb'))
  et_model = pickle.load(open('script/extratrees_model_7091fb%.pkl','rb'))
  
  # We decided not to use logistic regression as a final model 
  # list of classifier functions
  classifier_func = [lgbm.LGBMClassifier(colsample_bytree=0.5544879541065879, max_depth=38, num_leaves= 28, random_state=42),
                     RandomForestClassifier(max_depth=20, n_estimators=100, min_samples_leaf=5, min_samples_split=5, random_state=42), 
                     XGBClassifier(colsample_bytree= 0.5614800142167471, gamma= 0.37470611364219275, max_depth= 6, random_state=42)  

  # list of classifier names
  model_name= ['Light Gradient Boosting Method',
              'Random Forest', 
              'eXtreme Gradient Boosting',
              ]

  # evaluate performance and feature importance for each algorithm
  model_results(final_df, X_train, X_test, y_train, y_test, classifier_func, model_name)

  

