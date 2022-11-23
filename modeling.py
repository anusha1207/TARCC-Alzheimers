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

def get_data(non_genetic_df):

  df_combined = preprocessing(non_genetic_df)

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
  X_train = features[:math.ceil(0.8*final_df.shape[0])+1]
  y_train = y[:math.ceil(0.8*final_df.shape[0])+1]
  X_val = features[math.ceil(0.8*final_df.shape[0])+1:-math.floor(0.1*final_df.shape[0])]
  y_val = y[math.ceil(0.8*final_df.shape[0])+1:-math.floor(0.1*final_df.shape[0])]
  X_test = features[-math.floor(0.1*final_df.shape[0])]
  y_test = y[-math.floor(0.1*final_df.shape[0])]

  return X_train, y_train, X_val, y_val, X_test, y_test



def model_results(df, X_train, X_test, y_train, y_test, classifier_func, model_name):

  # perform evaluation on various models

  for model in range(len(classifier_func[:])):
    classifier_func[model].fit(X_train, y_train)

    print('-'*150)
    print(f'Evaluation for {model_name[model]}: ')
    y_pred = classifier_func[model].predict(X_test)
    evaluation(y_test, y_pred)
    
    # plot ROC curve
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


def model_main(non_genetic_df, dataset_name='combined'):

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


  # list of classifier functions
  classifier_func = [lgbm.LGBMClassifier(colsample_bytree=0.46053366496668136,num_leaves= 122, random_state=42),
                    RandomForestClassifier(n_estimators=900, max_depth=8, random_state=42), 
                    XGBClassifier(colsample_bytree= 0.840545160958208, gamma= 0.3433699189306628, max_depth= 2),                    
                    LogisticRegression(class_weight='balanced', max_iter=200, random_state=42, solver='sag'),
                    ExtraTreesClassifier(n_estimators=500, max_depth=3),
                    CatBoostClassifier(random_state=42)]  

  # list of classifier names
  model_name= ['Light Gradient Boosting Method',
              'Random Forest', 
              'eXtreme Gradient Boosting',
              'Logistic Regression', 
              'Extra Trees',
              'Categorical Boosting']

  # evaluate performance and feature importance for each algorithm
  model_results(final_df, X_train, X_test, y_train, y_test, classifier_func, model_name, dataset_name)

  
