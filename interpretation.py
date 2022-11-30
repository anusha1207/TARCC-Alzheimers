import shap
import modeling as m
import lightgbm as lgbm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from catboost import CatBoostClassifier
import pickle

def shap_function(classifier_func, model_name, final_features_df):
    for model in range(len(classifier_func[:])):
        shap_values = shap.TreeExplainer(classifier_func[model]).shap_values(final_features_df)
        #print(model_name[model])
        #print(shap_values)
        shap.summary_plot(shap_values, final_features_df, plot_type='bar', show=False)
        title = "SHAP Graph of " + model_name[model] + " Model"
        plt.title(title)
        plt_title = model_name[model] + "_SHAP.pdf"
        plt.savefig(plt_title, format="pdf", bbox_inches="tight")
        plt.show()

def interpretation_main(non_genetic_df):
    # pre-process the raw data
    df_features_comb, X_comb, y_comb = get_data(non_genetic_df)
    
    # retrieve pickled combined features list
    combined_features_list = pickle.load(open("pickled_combined_features_list.pkl", "rb" ))

    # getting only top features after feature selection
    final_features_df = df_features_comb[combined_features_list]
    # merge the dataset for machine learning model
    frames = [final_features_df, y_comb]
    final_df = pd.concat(frames, axis=1)

    lgbm_model = pickle.load(open("lgbm_model_8929%.pkl", "rb"))
    rf_model = pickle.load(open("rf_model_8750%.pkl", "rb"))
    xgb_model = pickle.load(open("xgb_model_8571%.pkl", "rb"))
    extratrees_model = pickle.load(open("extratrees_model_8636%.pkl", "rb"))
    catboost_model = pickle.load(open("catboost_model_8712%.pkl", "rb"))

    classifier_func = [lgbm_model, rf_model, xgb_model, extratrees_model, catboost_model]
    model_name = ['Light Gradient Boosting Method',
                'Random Forest',
                'eXtreme Gradient Boosting',
                'Extra Trees',
                'Categorical Boosting']
    shap_function(classifier_func, model_name, final_features_df)
