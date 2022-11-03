import preprocessing_blood as pb
import preprocessing_other as p
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
import config
# from sklearn import externals
# import joblib
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from pyparsing import printables

### Load data
data = pd.read_csv(config.DATA, low_memory=False)

df_blood = pb.preprocessing(data)
df_diagnosis = p.preprocessing(data)

#remove patient ID while doing feature selection
df_features_blood = df_blood.drop(['PATID'], axis=1)
##### Split features and target variable #####
X_blood = df_features_blood.drop(['P1_PT_TYPE'], axis=1, inplace = False)
y_blood = df_features_blood['P1_PT_TYPE']

df_features_diag = df_diagnosis.drop(['PATID'], axis=1)
##### Split features and target variable #####
X_diag = df_features_diag.drop(['P1_PT_TYPE'], axis=1, inplace = False)
y_diag = df_features_diag['P1_PT_TYPE']


def find_features(model, features, score):
    """
    This function lists and plots the top features
    """
    dict = {'Features':features,'Score':score}
    df=pd.DataFrame(dict)
    df=df.sort_values(by='Score', ascending=False)
    df=df[:20]
    df.plot.bar(x='Features',y='Score')
    plt.title(f'Feature Importance for {model}')
    plt.savefig(f'results/{model}_features.pdf', format="pdf", bbox_inches="tight")
    plot = plt.show()
    return df, plot


def random_forest_select(name, X,y):
    """
    This function lists and plots random forest feature selection
    """
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
    
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
    plt.savefig(f"results/RandomForest_features_{name}.pdf", format="pdf", bbox_inches="tight")
    rf_plot = plt.show()
    return rf_df, rf_plot


def recursive_selection(rfe, X, y):
    """
    This function lists recursive feature selection
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


def fb_selection(name, model, direction_name, direction):
    """
    This function lists and plots forward or backward feature selection 
    """
    ff1 = sfs(model, k_features=30, forward=direction, verbose=2, scoring='accuracy')
    ff1 = ff1.fit(X,y)
    ff1_dict=ff1.get_metric_dict(confidence_interval=0.8)
    ff1_df = pd.DataFrame.from_dict(ff1_dict).T
    fig1=plot_sfs(ff1_dict, kind='ci')
    plt.title(f'{direction_name} Feature Selection using {model} (With confidence interval)')
    plt.savefig(f'results/{model}_features_{name}.pdf', format="pdf", bbox_inches="tight")
    plt.grid()
    plot = plt.show()
    ff1_features=list(ff1_df['feature_names'][30])
    return ff1_features, fig1, plot


# def kruskal_select(name,X, y):
#     kruskal_features = []
#     kruskal_scores = []
#     for col in X.columns:
#         feature = X[col]

#         result = stats.kruskal(list(feature), list(y))
        
#         # reject null hypothesis if p <= p_value, else fail to reject null hypothesis and accept the column
#         if result.pvalue > 0:
#             kruskal_features.append(col)
#             kruskal_scores.append(result.pvalue)
    
#     # print(kruskal_features, kruskal_scores)
#     kruskal_features= pd.DataFrame(kruskal_features)
#     kruskal_scores= pd.DataFrame(kruskal_scores)
#     kruskal_df= pd.concat([kruskal_features, kruskal_scores], axis=1)
#     kruskal_df.columns = ['Features', 'Score']
#     kruskal_df = kruskal_df.sort_values(by='Score', ascending=False)
#     kruskal_df.iloc[:30, :]
#     kruskal_df.iloc[:30,:].plot.bar(x='Features',y='Score')
#     plt.title('Feature Selection using Kruskall_Wallace')
#     plt.savefig(f"results/Kruskall_features_{name}.pdf", format="pdf", bbox_inches="tight")
#     kw_plot = plt.show()
#     return kruskal_df, kw_plot

##### Boruta #####
def boruta_select(X,y):
    model = RandomForestRegressor(n_estimators=100, max_depth=5)
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

###### Mutual Info ######
def results(name, X,y, df_features):
    mi= MIC(X,y)
    #Mutual Info Features
    mi_cols=list(X.columns)
    #Mutual Info Scores
    mi_score=mi.tolist()
    mi_df, mi_plot = find_features(f'mutual_info_{name}', mi_cols, mi_score)

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
    chi_df, chi_plot = find_features(f'chi-square_{name}', chi_features, chi_score)

    ###### Random Forest ######

    rf_df, rf_plot = random_forest_select(name, X,y)

    ###### Recursive Selections ######
    rfr = RFE(estimator=RandomForestClassifier(), n_features_to_select=30)
    rfr_df = recursive_selection(rfr, X, y)

    dtr = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=30)
    dtr_df = recursive_selection(dtr, X, y)

    ###### Forward and Backward selection ######
    # these take so long to run
    #dtf_df, dtf_fig, dtf_plot = fb_selection(model = DecisionTreeClassifier(), direction_name = 'Forward', direction = True)
    #rff_df, rff_fig, rff_plot =fb_selection(model = RandomForestClassifier(), direction_name = 'Forward', direction = True)

    #lrb_df, lrb_fig, lrb_plot = fb_selection(model = LogisticRegression(), direction_name = 'Backward', direction = False)
    #dtb_df, dtb_fig, dtb_plot = fb_selection(model = DecisionTreeClassifier(), direction_name = 'Backward', direction = False)

    ############ Kruskal-Wallis #############

#     kruskal_df, kw_plot = kruskal_select(name, X,y)

    ########### Boruta ############
    b_df = boruta_select(X,y)

    ##### Combining all methods #####
    features=[]
    def combine_features():
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


mi_dfb, mi_plotb, chi_dfb, chi_plotb, rf_dfb, rf_plotb, rfr_dfb, dtr_dfb, kruskal_dfb, combined_featuresb = results('blood', X_blood, y_blood, df_features_blood)
mi_dfd, mi_plotd, chi_dfd, chi_plotd, rf_dfd, rf_plotd, rfr_dfd, dtr_dfd, kruskal_dfd, combined_featuresd = results('other', X_diag, y_diag, df_features_diag)
