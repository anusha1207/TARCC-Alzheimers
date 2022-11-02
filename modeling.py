
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import make_scorer
import copy
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, classification_report, precision_score, recall_score, roc_auc_score, f1_score, fbeta_score

def ml_prep(final_df):
  
  features = final_df.loc[:, final_df.columns != 'P1_PT_TYPE']
  y = final_df['P1_PT_TYPE']

  # standard scaling
  scaler = StandardScaler()
  X = scaler.fit_transform(features)

  # train_test_split (80/20)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

  return X_train, X_test, y_train, y_test

def results(df, X_train, X_test, y_train, y_test, classifier_func, model_name):

    # perform evaluation on various models

    for model in range(len(classifier_func[:])):
      classifier_func[model].fit(X_train, y_train)

      print('-'*150)
      print(f'Evaluation for {model_name[model]}: ')
      y_pred = classifier_func[model].predict(X_test)
      evaluation(y_test, y_pred)
      print('-'*150)
      print()

def evaluation(y_test, y_pred):

    # Accuracy classification score
    score = round(accuracy_score(y_test, y_pred), 4)
    print(f'Accuracy Score: {score*100}%')
    print()
    
    # Macro averaged precision: calculate precision for all classes individually and then average them
    macro_averaged_precision = precision_score(y_test, y_pred, average = 'macro')
    print("Macro-Averaged Precision score: {:.4f}".format(macro_averaged_precision))

    # Micro averaged precision: calculate class wise true positive and false positive and then use that to calculate overall precision
    micro_averaged_precision = precision_score(y_test, y_pred, average = 'micro')
    print("Micro-Averaged Precision score: {:.4f}".format(micro_averaged_precision))

    # Macro averaged recall: calculate recall for all classes individually and then average them
    macro_averaged_recall = recall_score(y_test, y_pred, average = 'macro')
    print("Macro-averaged recall score: {:.4f}".format(macro_averaged_recall))
    
    # Micro averaged recall: calculate class wise true positive and false negative and then use that to calculate overall recall
    micro_averaged_recall = recall_score(y_test, y_pred, average = 'micro')
    print("Micro-Averaged recall score: {:.4f}".format(micro_averaged_recall))

    # Macro averaged F1 Score: calculate f1 score of every class and then average them
    macro_averaged_f1 = f1_score(y_test, y_pred, average = 'macro')
    print("Macro-Averaged F1 score: {:.4f}".format(macro_averaged_f1))

    # Micro averaged F1 Score: calculate macro-averaged precision score and macro-averaged recall score and then take there harmonic mean
    micro_averaged_f1 = f1_score(y_test, y_pred, average = 'micro')
    print("Micro-Averaged F1 score: {:.4f}".format(micro_averaged_f1))

    f_beta = fbeta_score(y_test, y_pred, average='macro', beta=2)
    print("F-Beta score: {:.4f}".format(f_beta))

    # Receiver Operating Characteristic Area Under Curve (ROC_AUC) Score
    roc_auc_bi = roc_auc_score(y_test, y_pred, average = 'macro')
    print("ROC_AUC Score: {:.4f}".format(roc_auc_bi))

    # required to convert labels for auc_precision_recall
    # converted from 1 = AD and 2 = Control to 0 = AD and 1 = Control
    # convert y_test
    prc_y_test = copy.deepcopy(y_test)
    prc_y_test[prc_y_test==1] = 0
    prc_y_test[prc_y_test==2] = 1

    # convert y_pred
    prc_y_pred = copy.deepcopy(y_pred)
    prc_y_pred[prc_y_pred==1] = 0
    prc_y_pred[prc_y_pred==2] = 1

    # Compute Area Under the Precision Recall Curve
    precision, recall, thresholds = precision_recall_curve(prc_y_test, prc_y_pred)
    auc_precision_recall = auc(recall, precision)
    print("AUPRC is: {:.4f}".format(auc_precision_recall))

    # Get specificity from classification report
    class_report = classification_report(y_test, y_pred, labels=[1,2])
    print("Classification Report: ")
    print(class_report)

    # plot the confusion matrix
    plt.figure(figsize = (18,8))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot = True, xticklabels = y_test.unique(), yticklabels = y_test.unique(), cmap = 'summer')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    
def main(non_genetic_df):

  # pre-process the raw data

  X_train, X_test, y_train, y_test = ml_prep(final_df)

  # list of classifier functions
  classifier_func = [lgbm.LGBMClassifier(colsample_bytree=0.46053366496668136,num_leaves= 122, random_state=42),
                    RandomForestClassifier(n_estimators=900, max_depth=8, random_state=42),       
                    XGBClassifier(colsample_bytree= 0.5460418790379824, gamma= 0.3347828767144543, max_depth= 8),                     
                    GradientBoostingClassifier(n_estimators=300, max_depth=3), 
                    DecisionTreeClassifier(ccp_alpha=0.01, max_depth=6, max_features='log2', random_state=42),
                    LogisticRegression(class_weight='balanced', max_iter=200, random_state=42, solver='sag'),
                    ExtraTreesClassifier(n_estimators=500, max_depth=3)] 

  # list of classifier names
  model_name= ['Light Gradient Boosting Method', 
               'Random Forest', 
               'eXtreme Gradient Boosting', 
               'Gradient Boosting', 
               'Decision Tree', 
               'Logistic Regression', 
               'Extra Trees',
               'Categorical Boosting']

  # evaluate performance and feature importance for each algorithm
  results(final_df, X_train, X_test, y_train, y_test, classifier_func, model_name)

