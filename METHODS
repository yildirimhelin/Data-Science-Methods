import pandas as pd
import numpy as np

#Gives the percentage of missing values for all variables of 'df'. 
def missing_percent(df):
    missing = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([missing, percent], axis=1, keys=['Missing', 'Percent'])
    return missing_data

#Variables that have higher missing percentage than 'percent' are droped from the 'df'.
def filter_missing_columns(df,percent):
    missing_data=missing_percent(df)
    remaining_colums=missing_data[missing_data['Percent']<percent]
    #print('{} are removed'.format(missing_data[missing_data['Percent']>=percent]))
    a = remaining_colums.index
    return df[a]

#Variables that have most of its values as same are droped from the 'df'.
#Eg: single_value_elimination(df,0.95) drops features of 'df' if more than 95% of the values are the same.
def single_value_elimination(df,percent):
    modes = df.mode().iloc[0,:]
    remaining_columns=[]
    for i in range(len(df.columns)):
        count = 0
        for j in range(len(df)):
            if df.iloc[j, i] == modes[i]:
                count = count + 1
        p = (count / len(df))
        if p<=percent: remaining_columns.append(df.mode().iloc[0,:].index[i])
    return df[remaining_columns]
    
#Counts number of outliers
#Is outlier if falls outside of 1.5 times of an interquartile range above the 3rd quartile and below the 1st quartile.
def print_num_outliers_IQR(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    b=np.where((df > (Q3 + 1.5 * IQR)) | (df < (Q1 - 1.5 * IQR)))
    print('Total # of outliers: ', len(b[0]))
    print('# of rows containing outliers: ',len(np.unique(b[0])),'(%.2f' %(len(np.unique(b[0]))/len(df)),' of data)') 

#Is outlier if falls outside of n standard deviations of the mean. 
def print_num_outliers_Z(df,n=3):             
    from scipy import stats
    z=np.abs(stats.zscore(df))
    print('Total # of outliers: ',len(np.where(z > n)[0]))
    print('# of rows containing outliers: ',len(np.unique(np.where(z > n)[0])), '(%.2f' %((len(np.unique(np.where(z > n)[0])))/len(df)),' of data)')

#Drops the rows that contain outliers in at least one of the features based on Z-score
#(falling more than 'm' stdev far from mean is outlier).
def reject_outliers_Z(df, m=3):
    from scipy import stats
    z = np.abs(stats.zscore(df))
    df2=df[(z < m).all(axis=1)]
    print ("Shape without Z-score outliers: ",df2.shape)
    return df2
    
# This method can be used to understand the discrimination power of variables independently for feature elimination. 
# It generates univariate logit models with all of the variables seperately and calculates the gini values of models. 
# Variables which have gini< 0.07 is considered not to distinguish the target variable and therefore not explanatory.
def print_univariate_gini(x,y):
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    LogisticRegression(solver='lbfgs')
    LogReg = LogisticRegression()
    gini_univariate=[] 
    for i in range(len(x.columns)):
        a = np.array(x.iloc[:,i]).reshape(-1, 1)
        LogReg.fit(a,y)
        y_pred=LogReg.predict(a)
        y_score=LogReg.predict_proba(a)[:,1]
        fpr, tpr, thresholds = metrics.roc_curve(y,y_score,pos_label=1)
        auc = metrics.roc_auc_score(y, y_score)
        gini=(2 * auc) - 1
        gini_univariate.append(gini)
        print(x.columns[i],' gini: %.4f' %gini)
    gini_univariate=pd.DataFrame(gini_univariate,index=x.columns,columns=['GINI'])
    gini_univariate.sort_values(by='GINI', ascending=False,inplace=True)
    return gini_univariate
