import pandas as pd
import numpy as np
from hyperopt import tpe,hp,Trials
from hyperopt.fmin import fmin
import feature_selection as fs
import preprocessing as pre
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression #linear_model.LogisticRegression (setting multi_class=”multinomial”)
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
import pickle



def get_data(non_genetic_df):

  df_blood = pre.preprocessing(non_genetic_df)

  #remove patient ID while doing feature selection
  df_features_blood = df_blood.drop(['PATID'], axis=1)
  ##### Split features and target variable #####
  X_blood = df_features_blood.drop(['P1_PT_TYPE'], axis=1, inplace = False)
  y_blood = df_features_blood['P1_PT_TYPE']

  return df_features_blood, X_blood, y_blood

non_genetic_df = pd.read_csv('20220916 updated TARCC Data for Dr Broussard.csv')
df_features_blood, X_blood, y_blood = get_data(non_genetic_df)

combined_featuresb = pickle.load(open("final_features", "rb" ))
    # convert features to list
combined_features_list_blood = combined_featuresb['Features'][:15].to_list()
    # getting only top features after feature selection
final_features_df_blood = df_features_blood[combined_features_list_blood]
    # merge the dataset for machine learning model
frames_blood = [final_features_df_blood, y_blood]
final_df_blood = pd.concat(frames_blood, axis=1)

def ml_prep(final_df):
  
  features = final_df.loc[:, final_df.columns != 'P1_PT_TYPE']
  y = final_df['P1_PT_TYPE']

  # standard scaling
  scaler = StandardScaler()
  X = scaler.fit_transform(features)

  # train_test_split (80/20)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

  return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = ml_prep(final_df_blood)

### Random Forest
seed=2
def objective(params):
    est=int(params['n_estimators'])
    md=int(params['max_depth'])
    msl=int(params['min_samples_leaf'])
    mss=int(params['min_samples_split'])
    model=RandomForestClassifier(n_estimators=est, max_depth=md, min_samples_leaf=msl,min_samples_split=mss)
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    score=fbeta_score(y_test, pred, beta = 2)
    return score

def optimize(trial):
    params={'n_estimators':hp.uniform('n_estimators',100,500),
           'max_depth':hp.uniform('max_depth',5,20),
           'min_samples_leaf':hp.uniform('min_samples_leaf',1,5),
           'min_samples_split':hp.uniform('min_samples_split',2,6)}
    best=fmin(fn=objective,space=params,algo=tpe.suggest,trials=trial,max_evals=500,rstate=np.random.default_rng(seed))
    return best



X_train, X_test, y_train, y_test = ml_prep(df_features_blood)

trial=Trials()
best=optimize(trial)

print('Random Forest parameters: ', best)


### eXtreme Gradient Boosting

seed=2
def objective(params):
    est=int(params['n_estimators'])
    md=int(params['max_depth'])
    cst=int(params['colsample_bytree'])
    g=int(params['gamma'])
    model=XGBClassifier(n_estimators=est, max_depth=md, colsample_bytree=cst, gamma = g)
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    score=fbeta_score(y_test, pred, beta = 2)
    return score

def optimize(trial):
    params={'n_estimators':hp.uniform('n_estimators',100,1000, 1),
           'max_depth':hp.uniform('max_depth', 1, 14),
           'colsample_bytree':hp.uniform('colsample_bytree', 0.5, 1, 0.05),
           'gamma':hp.uniform('gamma', 0.5, 1, 0.05)}
    best=fmin(fn=objective,space=params,algo=tpe.suggest,trials=trial,max_evals=500,rstate=np.random.default_rng(seed))
    return best


trial=Trials()
best=optimize(trial)

print('XGboost parameters: ', best)

### Gradient Boosting

def objective(params):
    est=int(params['n_estimators'])
    md=int(params['max_depth'])
    model=GradientBoostingClassifier(n_estimators=est, max_depth=md)
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    score=fbeta_score(y_test, pred, beta = 1)
    return score

def optimize(trial):
    params={'n_estimators':hp.uniform('n_estimators',100,1000, 1),
           'max_depth':hp.uniform('max_depth', 1, 14)}
    best=fmin(fn=objective,space=params,algo=tpe.suggest,trials=trial,max_evals=500,rstate=np.random.default_rng(seed))
    return best


trial=Trials()
best=optimize(trial)

print('Gradient Boosting parameters: ', best)

### Decision Tree

def objective(params):
    ccp=int(params['ccp_alpha'])
    md=int(params['max_depth'])
    mf = params['max_features']
    model=DecisionTreeClassifier(ccp_alpha=ccp, max_depth= md, max_features= mf, random_state=42)
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    score=fbeta_score(y_test, pred, beta = 1)
    return score

def optimize(trial):
    params={'ccp_alpha':hp.uniform('n_estimators',100,1000, 1),
           'max_depth':hp.uniform('max_depth', 1, 14),
           'max_features': hp.choice('max_features', ["auto","log2","sqrt", None])}
    best=fmin(fn=objective,space=params,algo=tpe.suggest,trials=trial,max_evals=500,rstate=np.random.default_rng(seed))
    return best


trial=Trials()
best=optimize(trial)

print('Decision Tree parameters: ', best)

### Logistic Regression
seed = 2
def objective(params):
    cw=params['class_weight']
    mi=int(params['max_iter'])
    solv = params['solver']
    pen = params['penalty']
    c = int(params['C'])
    model=LogisticRegression(class_weight=cw, max_iter=mi, random_state=42, solver=solv, penalty = pen, C = c)
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    score=fbeta_score(y_test, pred, beta = 1)
    return -1 * score

def optimize(trial):
    params={'class_weight':hp.choice('class_weight',['balanced', None]),
           'max_iter':hp.uniform('max_iter', 1, 14),
           'solver': hp.choice('solver', ["lbfgs"]),
           'penalty' : hp.choice('penalty', ["l2", "none"]),
           'C' : hp.uniform('C', 0.01, 100)}
    best=fmin(fn=objective,space=params,algo=tpe.suggest,trials=trial,max_evals=500,rstate=np.random.default_rng(seed))
    return best


trial=Trials()
best=optimize(trial)

print('Logistic Regression parameters: ', best)

### Extra Trees

def objective(params):
    est=int(params['n_estimators'])
    md=int(params['max_depth'])
    model=ExtraTreesClassifier(n_estimators=est, max_depth=md)
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    score=fbeta_score(y_test, pred, beta = 1)
    return score

def optimize(trial):
    params={'n_estimators':hp.uniform('n_estimators',100,1000, 1),
           'max_depth':hp.uniform('max_depth', 1, 14)}
    best=fmin(fn=objective,space=params,algo=tpe.suggest,trials=trial,max_evals=500,rstate=np.random.default_rng(seed))
    return best


trial=Trials()
best=optimize(trial)

print('Extra Trees parameters: ', best)

### CatBoost
seed = 2
def objective(params):
    lr=params['learning_rate']
    depth=int(params['depth'])
    l2lr=params['l2_leaf_reg']
    bc=int(params['border_count'])
    model=CatBoostClassifier(learning_rate=lr, depth= depth, l2_leaf_reg = l2lr, border_count= bc, verbose= False)
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    score=fbeta_score(y_test, pred, beta = 1)
    return -1 * score

def optimize(trial):
    params={
        'depth': hp.quniform("depth", 1, 6, 1),
        'border_count': hp.uniform ('border_count', 32, 255),
        'learning_rate': hp.loguniform('learning_rate', -5.0, -2),
        'l2_leaf_reg': hp.uniform('l2_leaf_reg', 3, 8),
       }
    best=fmin(fn=objective,space=params,algo=tpe.suggest,trials=trial,max_evals=500,rstate=np.random.default_rng(seed))
    return best


trial=Trials()
best=optimize(trial)

print('Cat Boost parameters: ', best)