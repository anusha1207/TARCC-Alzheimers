import pandas as pd
import numpy as np
from hyperopt import tpe,hp,Trials
from hyperopt.fmin import fmin
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

### Tune Random Forest model
def objective(params):

    est=int(params['n_estimators'])
    #feat= params['max_features']
    md=int(params['max_depth'])
    msl=int(params['min_samples_leaf'])
    mss=int(params['min_samples_split'])

    model=RandomForestClassifier(n_estimators=est, max_depth=md, min_samples_leaf=msl,min_samples_split=mss, random_state=42)
    model.fit(X_train,y_train)
    pred=model.predict(X_val)
    score=fbeta_score(y_val, pred, beta = 1.5)
    print("F_beta {:.3f} params {}".format(score, params))
    
    return score

def optimize(trial):
    params={'n_estimators':hp.quniform('n_estimators',100,1100, 100),
            #'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2', None]),
           'max_depth':hp.quniform('max_depth',5,50, 5),
           'min_samples_leaf':hp.quniform('min_samples_leaf',1,10, 1),
           'min_samples_split':hp.quniform('min_samples_split',2,10,1)}
    best=fmin(fn=objective,space=params,algo=tpe.suggest,trials=trial,max_evals=500)
    return best

trial=Trials()
best=optimize(trial)

print('Random Forest parameters: ', best)

############################################################################################################

### tune XGboost model
def objective(params):
    params={
        # 'n_estimators':int(params['n_estimators']),
            'max_depth':int(params['max_depth']),
            'colsample_bytree':float(params['colsample_bytree']),
            'gamma':float(params['gamma']),}
            # 'min_child_weight':int(params['min_child_weight']),
            # 'eta':float(params['eta']),}
            # 'subsample':int(params['subsample'])}

    model=XGBClassifier(**params, random_state=42)
    model.fit(X_train,y_train)
    pred=model.predict(X_val)
    score=fbeta_score(y_val, pred, beta = 1.5)
    print("F_beta {:.3f} params {}".format(score, params))

    return score

def optimize(trial):
    space={
        # 'n_estimators':hp.quniform('n_estimators',100,1000, 100),
           'max_depth':hp.quniform('max_depth', 1, 12, 2),
           'colsample_bytree':hp.uniform('colsample_bytree',0.3,1),
           'gamma':hp.uniform('gamma', 0, 1),}
          #  'min_child_weight':hp.quniform('min_child_weight', 1, 12, 2),
          #  'eta':hp.uniform('eta', 0.1, 0.5),
          #  'subsample':hp.quniform('subsample', 0.5, 1, 0.05) }
          
    best=fmin(fn=objective,space=space,algo=tpe.suggest,trials=trial,max_evals=500)

    return best

trial=Trials()
best=optimize(trial)

print('XGboost parameters: ', best)

############################################################################################################

### tune LGBM model
def objective(params):

  params={
  'colsample_bytree':float(params['colsample_bytree']),

  # 'learning_rate':float(params['learning_rate']),
  'max_depth': int(params['max_depth']),
  # 'max_bin': int(params['max_bin']),
  # 'min_data_in_leaf': int(params['min_data_in_leaf']),
  # 'subsample': float(params['subsample']),
  'num_leaves': int(params['num_leaves'])}

  lgbm_model = lgbm.LGBMClassifier(n_jobs=-1,early_stopping_rounds=None,**params, random_state=42)
  lgbm_model.fit(X_train,y_train)
  pred=lgbm_model.predict(X_val)
  score=fbeta_score(y_val, pred, beta = 1.5)
  print("F_beta {:.3f} params {}".format(score, params))

  return score

def optimize(trial):

    space = {
    'colsample_bytree': hp.uniform('colsample_bytree',0.3,1),
    # 'learning_rate': hp.uniform('learning_rate',0.01,1),
    'max_depth': hp.choice('max_depth', np.arange(2, 100, 1, dtype=int)),
    # 'max_bin': hp.uniform('max_bin',20,90),
    # 'min_data_in_leaf': hp.uniform('min_data_in_leaf', 20,80),
    # 'subsample': hp.uniform('subsample', 0.01, 1),
    'num_leaves': hp.choice('num_leaves', np.arange(8, 200, 1, dtype=int))}
   
    best=fmin(fn=objective,space=space,algo=tpe.suggest,trials=trial,max_evals=500)
    return best

trial=Trials()
best=optimize(trial)

print("LGBM estimated optimum {}".format(best))
