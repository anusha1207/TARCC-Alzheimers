import shap
from sklearn.ensemble import RandomForestClassifier
import pickle
import modeling as m

def shap_function(classifier_func, model_name, final_features_df):
    # get shap values based on the model and input dataframe passed in
    shap_values = shap.TreeExplainer(classifier_func).shap_values(final_features_df)
    
    # graph a SHAP summary plot showing top features in order of influence to output
    shap.summary_plot(shap_values, final_features_df, show=False)
    title = "SHAP Graph of " + model_name + " Model"
    plt.title(title)
    plt.legend(['', 'Class \'1\'', 'Class \'2\''])
    
    # save SHAP summary plot as a PDF file
    plt_title = model_name + "_SHAP.pdf"
    plt.savefig(plt_title, format="pdf", bbox_inches="tight")
    
    # show the SHAP summary plot graph
    plt.show()

def interpretation_main(non_genetic_df):
    """
    This function interprets the best performing model in determining
    in whether each of the  top features chosen during feature selection 
    resist or progress the progression of Alzheimer's for each patient 
    and even overall
    INPUTS:
        non_genetic_df -- <pd.DataFrame> raw biological dataset
    OUTPUT: SHAP summary plot
    """
    # pre-process the raw data
    df_features_comb, X_comb, y_comb = m.get_data(non_genetic_df)
    
    # retrieve pickled combined features list
    combined_features_list = pickle.load(open("/content/drive/MyDrive/Capstone COMP 549/TARCC_F22-main/data/pickled_combined_features_list.pkl", "rb" ))

    # getting only top features after feature selection
    final_features_df = df_features_comb[combined_features_list]
    # merge the dataset for machine learning model and interprettation
    frames = [final_features_df, y_comb]
    final_df = pd.concat(frames, axis=1)

    # load pickled best model
    best_model = pickle.load(open("/content/drive/MyDrive/Capstone COMP 549/TARCC_F22-main/data/rf_model_best.pkl", "rb"))
    
    # get interpretation of best model according to F-score
    shap_function(best_model, "Random Forests", final_features_df)
