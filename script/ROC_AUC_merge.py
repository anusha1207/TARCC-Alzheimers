### Plotting ROC_AUC curve from all models into one plot 

# load pickled models
lgbm_model = pickle.load(open('script/lgbm_model_f_beta_7377%.pkl','rb'))
catboost_model = pickle.load(open('script/catboost_model_7500fb%.pkl','rb'))
rf_model = pickle.load(open('script/rf_model_f_beta_7937%.pkl','rb'))
xgb_model = pickle.load(open('script/xgb_model_f_beta_7500%.pkl','rb'))
et_model = pickle.load(open('script/extratrees_model_7091fb%.pkl','rb'))

# ROC Curve
# LGBM
from sklearn.metrics import roc_curve
y_pred_prob1 = lgbm_model.predict_proba(X_test)[:,1]
fpr1 , tpr1, thresholds1 = roc_curve(y_test, y_pred_prob1, pos_label=1)

# CATBOOST
y_pred_prob2 = catboost_model.predict_proba(X_test)[:,1]
fpr2 , tpr2, thresholds2 = roc_curve(y_test, y_pred_prob2, pos_label=1)

# RF
y_pred_prob3 = xgb_model.predict_proba(X_test)[:,1]
fpr3 , tpr3, thresholds3 = roc_curve(y_test, y_pred_prob3, pos_label=1)

# XGBOOST
y_pred_prob4 = xgb_model.predict_proba(X_test)[:,1]
fpr4 , tpr4, thresholds4 = roc_curve(y_test, y_pred_prob4, pos_label=1)

# EXTRATREES
y_pred_prob5 = et_model.predict_proba(X_test)[:,1]
fpr5 , tpr5, thresholds5 = roc_curve(y_test, y_pred_prob5, pos_label=1)

plt.plot([0,1],[0,1], 'k--')
plt.plot(tpr1, fpr1, label= "LGBM")
plt.plot(tpr2, fpr2, label= "CATBOOST")
plt.plot(tpr3, fpr3, label= "RANDOM_FOREST")
plt.plot(tpr4, fpr4, label= "XGBOOST")
plt.plot(tpr5, fpr5, label= "EXTRATREES")

plt.legend()
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title('Receiver Operating Characteristic')
# plt.savefig(f"Combined_ROC_Plot.pdf", format="pdf", bbox_inches="tight")
plt.show()
