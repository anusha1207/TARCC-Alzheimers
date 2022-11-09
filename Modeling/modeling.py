from Preprocessing_Feature_Selection import preprocessing_blood as pb
from Preprocessing_Feature_Selection import preprocessing_other as po
from Preprocessing_Feature_Selection import feature_selection as fs
import pandas as pd
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
#import lightgbm
import copy
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, classification_report, precision_score, recall_score, roc_auc_score, f1_score, fbeta_score
from sklearn import metrics

def get_data(non_genetic_df):

  df_blood = pb.preprocessing(non_genetic_df)
  df_diagnosis = po.preprocessing(non_genetic_df)

  #remove patient ID while doing feature selection
  df_features_blood = df_blood.drop(['PATID'], axis=1)
  ##### Split features and target variable #####
  X_blood = df_features_blood.drop(['P1_PT_TYPE'], axis=1, inplace = False)
  y_blood = df_features_blood['P1_PT_TYPE']

  df_features_diag = df_diagnosis.drop(['PATID'], axis=1)
  ##### Split features and target variable #####
  X_diag = df_features_diag.drop(['P1_PT_TYPE'], axis=1, inplace = False)
  y_diag = df_features_diag['P1_PT_TYPE']

  return df_features_blood, df_features_diag, X_blood, y_blood, X_diag, y_diag


def ml_prep(final_df):
  
  features = final_df.loc[:, final_df.columns != 'P1_PT_TYPE']
  y = final_df['P1_PT_TYPE']

  # standard scaling
  scaler = StandardScaler()
  X = scaler.fit_transform(features)

  # train_test_split (80/20)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

  return X_train, X_test, y_train, y_test


def model_results(df, X_train, X_test, y_train, y_test, classifier_func, model_name, dataset='blood'):

  # perform evaluation on various models

  for model in range(len(classifier_func[:])):
    classifier_func[model].fit(X_train, y_train)

    print('-'*150)
    print(f'Evaluation for {model_name[model]}: ')
    y_pred = classifier_func[model].predict(X_test)
    evaluation(y_test, y_pred)
    
    # plot ROC curve
    metrics.plot_roc_curve(classifier_func[model], X_test, y_test, pos_label=1)
    plt.savefig(f'results/{model_name[model]}_ROC_{dataset}.pdf', format="pdf", bbox_inches="tight")
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


def model_main(non_genetic_df, dataset='blood'):

  # pre-process the raw data
  df_features_blood, df_features_diag, X_blood, y_blood, X_diag, y_diag = get_data(non_genetic_df)

  if dataset=='blood':
    
    # getting combined features after performing feature selection
    mi_dfb, mi_plotb, chi_dfb, chi_plotb, rf_dfb, rf_plotb, rfr_dfb, dtr_dfb, b_dfb, combined_featuresb = fs.results('blood', X_blood, y_blood, df_features_blood)

    # convert features to list
    combined_features_list_blood = combined_featuresb['Features'].to_list()
    # getting only top features after feature selection
    final_features_df_blood = df_features_blood[combined_features_list_blood]
    # merge the dataset for machine learning model
    frames_blood = [final_features_df_blood, y_blood]
    final_df_blood = pd.concat(frames_blood, axis=1)

    # perform train_test_split
    X_train, X_test, y_train, y_test = ml_prep(final_df_blood)

    # list of classifier functions
    classifier_func = [RandomForestClassifier(n_estimators=900, max_depth=8, random_state=42), 
                    XGBClassifier(colsample_bytree= 0.840545160958208, gamma= 0.3433699189306628, max_depth= 2),                    
                    GradientBoostingClassifier(n_estimators=300, max_depth=3), 
                    DecisionTreeClassifier(ccp_alpha=0.01, max_depth=6, max_features='log2', random_state=42),
                    LogisticRegression(class_weight='balanced', max_iter=200, random_state=42, solver='sag'),
                    ExtraTreesClassifier(n_estimators=500, max_depth=3),
                    CatBoostClassifier(random_state=42)]  

    # list of classifier names
    model_name= ['Random Forest', 
              'eXtreme Gradient Boosting',
              'Gradient Boosting', 
              'Decision Tree', 
              'Logistic Regression', 
              'Extra Trees',
              'Categorical Boosting']

    # evaluate performance and feature importance for each algorithm
    model_results(final_df_blood, X_train, X_test, y_train, y_test, classifier_func, model_name, dataset)

  elif dataset=='other':

    # getting combined features after performing feature selection
    mi_dfb, mi_plotb, chi_dfb, chi_plotb, rf_dfb, rf_plotb, rfr_dfb, dtr_dfb, b_dfb, combined_featuresd = fs.results('other', X_diag, y_diag, df_features_diag)

    # convert features to list
    combined_features_list_diagnosis = combined_featuresd['Features'].to_list()

    # getting only top features after feature selection
    final_features_df_diagnosis = df_features_diag[combined_features_list_diagnosis]

    # merge the dataset for machine learning model
    frames_diagnosis = [final_features_df_diagnosis, y_diag]
    final_df_diagnosis= pd.concat(frames_diagnosis, axis=1)
          
    # perform train_test_split
    X_train, X_test, y_train, y_test = ml_prep(final_df_diagnosis)

    # list of classifier functions; need to fine tune and re-train
    classifier_func = [RandomForestClassifier(n_estimators=900, max_depth=8, random_state=42), 
                    XGBClassifier(colsample_bytree= 0.5460418790379824, gamma= 0.3347828767144543, max_depth= 8),                    
                    GradientBoostingClassifier(n_estimators=300, max_depth=3), 
                    DecisionTreeClassifier(ccp_alpha=0.01, max_depth=6, max_features='log2', random_state=42),
                    LogisticRegression(class_weight='balanced', max_iter=200, random_state=42, solver='sag'),
                    ExtraTreesClassifier(n_estimators=500, max_depth=3),
                    CatBoostClassifier(random_state=42)] 

    # list of classifier names
    model_name= ['Random Forest', 
              'eXtreme Gradient Boosting',
              'Gradient Boosting', 
              'Decision Tree', 
              'Logistic Regression', 
              'Extra Trees',
              'Categorical Boosting']

    # evaluate performance and feature importance for each algorithm
    model_results(final_df_diagnosis, X_train, X_test, y_train, y_test, classifier_func, model_name, dataset)

    


