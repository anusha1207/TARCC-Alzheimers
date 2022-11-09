import pandas as pd
import numpy as np
from hyperopt import tpe,hp,Trials
from hyperopt.fmin import fmin
from Preprocessing_Feature_Selection import feature_selection as fs
from Preprocessing_Feature_Selection import preprocessing_blood as pb
from Preprocessing_Feature_Selection import preprocessing_other as po
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score



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

seed=2
def objective(params):
    est=int(params['n_estimators'])
    md=int(params['max_depth'])
    msl=int(params['min_samples_leaf'])
    mss=int(params['min_samples_split'])
    model=RandomForestClassifier(n_estimators=est, max_depth=md, min_samples_leaf=msl,min_samples_split=mss)
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    score=f1_score(y_test, pred, average = 'binary')
    return score

def optimize(trial):
    params={'n_estimators':hp.uniform('n_estimators',100,500),
           'max_depth':hp.uniform('max_depth',5,20),
           'min_samples_leaf':hp.uniform('min_samples_leaf',1,5),
           'min_samples_split':hp.uniform('min_samples_split',2,6)}
    best=fmin(fn=objective,space=params,algo=tpe.suggest,trials=trial,max_evals=500,rstate=np.random.default_rng(seed))
    return best


non_genetic_df = pd.read_csv('20220916 updated TARCC Data for Dr Broussard.csv')
df_features_blood, df_features_diag, X_blood, y_blood, X_diag, y_diag = get_data(non_genetic_df)
X_train, X_test, y_train, y_test = ml_prep(df_features_blood)

trial=Trials()
best=optimize(trial)

print(best)


