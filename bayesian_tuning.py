import pandas as pd
import numpy as np
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from Preprocessing_Feature_Selection import feature_selection as fs
from Preprocessing_Feature_Selection import preprocessing_blood as pb
from Preprocessing_Feature_Selection import preprocessing_other as po
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression #linear_model.LogisticRegression (setting multi_class=”multinomial”)
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier


def get_data(non_genetic_df):
    df_blood = pb.preprocessing(non_genetic_df)
    df_diagnosis = po.preprocessing(non_genetic_df)

    # remove patient ID while doing feature selection
    df_features_blood = df_blood.drop(['PATID'], axis=1)
    ##### Split features and target variable #####
    X_blood = df_features_blood.drop(['P1_PT_TYPE'], axis=1, inplace=False)
    y_blood = df_features_blood['P1_PT_TYPE']

    df_features_diag = df_diagnosis.drop(['PATID'], axis=1)
    ##### Split features and target variable #####
    X_diag = df_features_diag.drop(['P1_PT_TYPE'], axis=1, inplace=False)
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

def bayesian_optimization(X_train, y_train, model, params, n_iter, random_state):
    opt = BayesSearchCV(
        model,
        params,
        n_iter=n_iter,
        random_state=random_state, 
        scoring='roc_auc'
    )
    opt = opt.fit(X_train, y_train)
    return opt.best_params_

def bayesian_results(X_train, y_train, classifier_func, classifier_params, model_names, n_iter, random_state):
    model_best_params = {}
    for model_index in range(0, len(classifier_func)):
        model = classifier_func[model_index]
        params = classifier_params[model_index]
        name = model_names[model_index]
        best_params = bayesian_optimization(X_train, y_train, model, params, n_iter, random_state)
        model_best_params[name] = best_params
    return model_best_params

def bayesian_tuning_main(df):
    df_features_blood, df_features_diag, X_blood, y_blood, X_diag, y_diag = get_data(df)
    X_train, X_test, y_train, y_test = ml_prep(df_features_blood)
    classifier_func = [
        RandomForestClassifier(),
        LogisticRegression(),
        ExtraTreesClassifier(),
        XGBClassifier(),
        CatBoostClassifier()
    ]
    classifier_params = [
        {
            'n_estimators': Integer(100, 500),
            'max_depth': Integer(5, 20),
            'min_samples_leaf': Integer(1, 5),
            'min_samples_split': Integer(2, 6)
        },
        {
            'class_weight': Categorical(['balanced', None]),
            'max_iter': Integer(1, 14),
            'solver': Categorical(["newton-cg", "lbfgs", "liblinear", "sag", "saga"])
        },
        {
            'n_estimators': Integer(100, 1000),
            'max_depth': Integer(1, 14)
        },
        {
            'n_estimators': Integer(100, 1000),
            'max_depth': Integer(1, 14),
            'colsample_bytree': Real(0.5, 1.0),
            'gamma': Real(0.5, 1.0)
        },
        {
            'depth': Integer(1, 6),
            'border_count': Integer(32, 255),
            'learning_rate': Real(-5.0, -2),
            'l2_leaf_reg': Integer(3, 8)
        }
    ]
    model_names = [
        'Random Forest',
        'Logistic Regression',
        'Extra Trees',
        'eXtreme Gradient Boost',
        'Categorical Boosting'
    ]
    bayesian_results(X_train, y_train, classifier_func, classifier_params, model_names, n_iter=500, random_state=2)

non_genetic_df = pd.read_csv('20220916 updated TARCC Data for Dr Broussard.csv')
bayesian_tuning_main(non_genetic_df)

