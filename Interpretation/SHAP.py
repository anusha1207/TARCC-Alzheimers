import shap
import pandas as pd
import modeling as m
import feature_selection as fs


def shap_tree(model, model_name, X_train, y_train, input_df):
    model.fit(X_train, y_train)
    return shap.TreeExplainer(model).shap_values(input_df)
    
def shap_explainer(model, model_name, X_train, y_train, input_df):
    model.fit(X_train, y_train)
    return shap.Explainer(model, input_df, feature_names=input_df.columns)

def shap_plot(model):
    return


def shap_main(non_genetic_df, dataset='blood'):
    df_features_blood, df_features_diag, X_blood, y_blood, X_diag, y_diag = m.get_data(non_genetic_df)
    if dataset == 'blood':
        # getting combined features after performing feature selection
        mi_dfb, mi_plotb, chi_dfb, chi_plotb, rf_dfb, rf_plotb, rfr_dfb, dtr_dfb, b_dfb, combined_featuresb = fs.results(
            'blood', X_blood, y_blood, df_features_blood)

        # convert features to list
        combined_features_list_blood = combined_featuresb['Features'].to_list()
        # getting only top features after feature selection
        final_features_df_blood = df_features_blood[combined_features_list_blood]
        # merge the dataset for machine learning model
        frames_blood = [final_features_df_blood, y_blood]
        final_df_blood = pd.concat(frames_blood, axis=1)

        # perform train_test_split
        X_train, X_test, y_train, y_test = m.ml_prep(final_df_blood)

        # list of classifier functions
        classifier_func = [lgbm.LGBMClassifier(colsample_bytree=0.46053366496668136, num_leaves=122, random_state=42),
                           RandomForestClassifier(n_estimators=900, max_depth=8, random_state=42),
                           XGBClassifier(colsample_bytree=0.840545160958208, gamma=0.3433699189306628, max_depth=2),
                           GradientBoostingClassifier(n_estimators=300, max_depth=3),
                           DecisionTreeClassifier(ccp_alpha=0.01, max_depth=6, max_features='log2', random_state=42),
                           # LogisticRegression(class_weight='balanced', max_iter=200, random_state=42, solver='sag'),
                           ExtraTreesClassifier(n_estimators=500, max_depth=3),
                           CatBoostClassifier(random_state=42)]

        # list of classifier names
        model_name = ['Light Gradient Boosting Method',
                      'Random Forest',
                      'eXtreme Gradient Boosting',
                      'Gradient Boosting',
                      'Decision Tree',
                      # 'Logistic Regression',
                      'Extra Trees',
                      'Categorical Boosting']

        for model_index in range(len(classifier_func[:])):
            model = classifier_func[model_index]
            name = model_name[model_index]
            model_shap_values = shap_tree(model, name, X_train, y_train, final_df_blood)

    elif dataset == 'other':

        # getting combined features after performing feature selection
        mi_dfb, mi_plotb, chi_dfb, chi_plotb, rf_dfb, rf_plotb, rfr_dfb, dtr_dfb, b_dfb, combined_featuresd = fs.results(
            'other', X_diag, y_diag, df_features_diag)

        # convert features to list
        combined_features_list_diagnosis = combined_featuresd['Features'].to_list()

        # getting only top features after feature selection
        final_features_df_diagnosis = df_features_diag[combined_features_list_diagnosis]

        # merge the dataset for machine learning model
        frames_diagnosis = [final_features_df_diagnosis, y_diag]
        final_df_diagnosis = pd.concat(frames_diagnosis, axis=1)

        # perform train_test_split
        X_train, X_test, y_train, y_test = m.ml_prep(final_df_diagnosis)

        # list of classifier functions; need to fine tune and re-train
        classifier_func = [lgbm.LGBMClassifier(colsample_bytree=0.46053366496668136, num_leaves=122, random_state=42),
                           RandomForestClassifier(n_estimators=900, max_depth=8, random_state=42),
                           XGBClassifier(colsample_bytree=0.5460418790379824, gamma=0.3347828767144543, max_depth=8),
                           GradientBoostingClassifier(n_estimators=300, max_depth=3),
                           DecisionTreeClassifier(ccp_alpha=0.01, max_depth=6, max_features='log2', random_state=42),
                           LogisticRegression(class_weight='balanced', max_iter=200, random_state=42, solver='sag'),
                           ExtraTreesClassifier(n_estimators=500, max_depth=3),
                           CatBoostClassifier(random_state=42)]

        # list of classifier names
        model_name = ['Light Gradient Boosting Method',
                      'Random Forest',
                      'eXtreme Gradient Boosting',
                      'Gradient Boosting',
                      'Decision Tree',
                      'Logistic Regression',
                      'Extra Trees',
                      'Categorical Boosting']
