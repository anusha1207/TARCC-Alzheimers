import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif as MIC
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
import warnings
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression #linear_model.LogisticRegression (setting multi_class=”multinomial”)
from sklearn.feature_selection import RFE
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
# from sklearn import externals
# import joblib

from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from pyparsing import printables

def find_features(model, features, score):
    """
    This function lists and plots the top features
    INPUTS:
         model -- <str> name of model
         features -- <list> list of features
         score -- <list> score of features
    OUTPUTS:
        df -- <pd.DataFrame> features and their scores
        plot -- <pd.DataFrame.plot.bar> plot of top 20 features and scores
    """
    dict = {'Features':features,'Score':score}
    df=pd.DataFrame(dict)
    df=df.sort_values(by='Score', ascending=False)
    df=df[:20]
    df.plot.bar(x='Features',y='Score')
    plt.title(f'Feature Importance for {model}')
    # plt.savefig(f'results/{model}_features.pdf', format="pdf", bbox_inches="tight")
    plot = plt.show()
    return df, plot


def random_forest_select(X,y):
    """
    This function lists and plots random forest feature selection
    INPUTS:
        X -- <pd.DataFrame> features
        y -- <pd.Series> target variable
    OUTPUTS:
        rf_df -- <pd.DataFrame> features and their scores
        rf_plot -- <pd.DataFrame.plot.bar> plot of top 30 features and scores
    """
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=42)
    
    #Standard Scaling
    sc=StandardScaler()
    sc.fit(X_train)
    X_train_std=sc.transform(X_train)
    X_test_std=sc.transform(X_test)
    
    Xcols=list(X.columns)
    X_train_std=pd.DataFrame(X_train_std, columns=Xcols)
    X_test_std=pd.DataFrame(X_test_std, columns=Xcols)
    
    #Fitting the model
    forest = RandomForestClassifier(n_estimators=100)
    forest.fit(X_train_std, y_train.values.ravel())
    
    #RF Score
    rf_score=forest.feature_importances_.tolist()

    #RF Features
    rf_features=list(X.columns)

    rf_dict = {'Features':rf_features,'Score':rf_score}
    rf_df=pd.DataFrame(rf_dict)
    rf_df=rf_df.sort_values(by='Score', ascending=False)
    rf_df=rf_df.head(30)
    rf_df.plot.bar(x='Features',y='Score')
    plt.title(f'Feature Importance for Random Forest')
    # plt.savefig(f"results/RandomForest_features_{name}.pdf", format="pdf", bbox_inches="tight")
    rf_plot = plt.show()
    return rf_df, rf_plot


def recursive_selection(rfe, X, y):
    """
    This function lists recursive feature selection
    INPUTS:
        rfe -- <sklearn.feature_selection._rfe.RFE> RFE model
        X -- <pd.DataFrame> features
        y -- <pd.Series> target variables
    OUTPUTS:
        rfs_df -- <pd.DataFrame> features and their scores
    """
    rfs=rfe.fit(X, y)
    #RFS Features
    rfs_features= list(X.columns)
    #RFS Scores
    rfs_score= rfs.support_.tolist()
    rfs_dict = {'Features':rfs_features,'Score':rfs_score}
    rfs_df= pd.DataFrame.from_dict(rfs_dict)
    rfs_df=rfs_df.loc[rfs_df['Score'] == True]
    return rfs_df


##### Boruta #####
def boruta_select(X,y):
    """
    This function finds top features by the boruta package
    INPUTS:
        X -- <pd.DataFrame> features
        y -- <pd.Series> target variables
    OUTPUTS:
        boruta_features -- <list> features and their scores
    """
    # let's initialize Boruta
    feat_selector = BorutaPy(
        verbose=2,
        estimator=model,
        n_estimators='auto',
        max_iter=10  # number of iterations to perform
    )

    # train Boruta
    # N.B.: X and y must be numpy arrays
    feat_selector.fit(np.array(X), np.array(y))

    # print support and ranking for each feature
    print("\n------Support and Ranking for each feature------")

    for i in range(len(feat_selector.support_)):
        if feat_selector.support_[i]:
            print(X.columns[i],
                " - Ranking: ", feat_selector.ranking_[i])

    boruta_features=[]
    for i in range(len(feat_selector.support_)):
        if feat_selector.support_[i]:
            boruta_features.append(X.columns[i])
    return boruta_features


def results(df_features):
    """
    This function returns the top features of each selection method and the overall top features
    INPUTS:
        df_features -- <pd.DataFrame> preprocessed dataset
    OUTPUTS:
        mi_df -- <pd.DataFrame> mutual info features and their scores
        mi_plot -- <pd.DataFrame.plot.bar> plot of top 30 mututal info features and scores
        chi_df -- <pd.DataFrame> chi square features and their scores
        chi_plot -- <pd.DataFrame.plot.bar> plot of top 30 chi square features and scores
        rf_df -- <pd.DataFrame> random forest features and their scores
        rf_plot -- <pd.DataFrame.plot.bar> plot of top 30 random forest features and scores
        rfr_df -- <pd.DataFrame> random forest recursive selection features and their scores
        dtr_df -- <pd.DataFrame> decision tree recursive selection features and their scores
        b_df -- <pd.DataFrame> boruta features and their scores
        combined_features -- <pd.DataFrame> combined features and their scores
    """

    X = df_features.drop(['P1_PT_TYPE'], axis=1, inplace = False)
    y = df_features['P1_PT_TYPE']

    ###### Mutual Info ######
    mi= MIC(X,y)
    #Mutual Info Features
    mi_cols=list(X.columns)
    #Mutual Info Scores
    mi_score=mi.tolist()
    mi_df, mi_plot = find_features(f'mutual_info_', mi_cols, mi_score)

    ###### Chi-Square ######

    df2=df_features
    df2[df2<0]=99 #Chi square doesn't recognize negative values- missing values aka -9 are resubstituted as 99
    X1 = df2.drop(['P1_PT_TYPE'], axis=1, inplace = False)
    y1 = df2['P1_PT_TYPE']
    chi = chi2(X1,y1)
    #Chi Test Score
    chi_score=chi[1].tolist()
    # #Chi Test Features
    chi_features=list(X1)
    chi_df, chi_plot = find_features(f'chi-square_', chi_features, chi_score)

    ###### Random Forest ######
    rf_df, rf_plot = random_forest_select(X,y)

    ###### Recursive Selections ######
    rfr = RFE(estimator=RandomForestClassifier(), n_features_to_select=30)
    rfr_df = recursive_selection(rfr, X, y)

    dtr = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=30)
    dtr_df = recursive_selection(dtr, X, y)

    ########### Boruta ############
    b_df = boruta_select(X,y)

    ##### Combining all methods #####
    features=[]
    def combine_features():
        """
        This function combines top features from each model and selects the top 28 most frequent features
        OUPUT:
        combined_features -- <pd.DataFrame> top 28 most frequent top features from every model
        """
        features=list(mi_df['Features'])+list(rf_df['Features'])+list(rfr_df['Features'])+list(dtr_df['Features'])+b_df
        features=pd.DataFrame(features).reset_index(drop=True)
        features.columns = ['Features']
        counts = features['Features'].value_counts().to_frame().reset_index()
        counts.columns=['Features','Frequency']
        combined_features=counts.head(28)
        combined_feature_list=list(counts['Features'])
        return combined_features
    combined_features = combine_features()
    return mi_df, mi_plot, chi_df, chi_plot, rf_df, rf_plot, rfr_df, dtr_df, b_df, combined_features

# run to pickle features
# getting combined features after performing feature selection
"""
mi_dfb, mi_plotb, chi_dfb, chi_plotb, rf_dfb, rf_plotb, rfr_dfb, dtr_dfb, b_dfb, combined_features = results(df_features)
# convert features to list
combined_features_list= combined_features['Features'][:15].to_list()
pickle.dump(combined_features_list, open('pickled_combined_features_list.pkl','wb'))
"""