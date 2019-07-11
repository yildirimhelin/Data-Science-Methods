import pandas as pd
import numpy as np

#Gives the percentage of missing values for all variables of 'df'. 
def missing_percent(df):
    missing = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([missing, percent], axis=1, keys=['Missing', 'Percent'])
    return missing_data

#Variables that have higher missing percentage than 'percent' are dropped from the 'df'.
def filter_missing_columns(df,percent):
    missing_data=missing_percent(df)
    remaining_colums=missing_data[missing_data['Percent']<percent]
    #print('{} are removed'.format(missing_data[missing_data['Percent']>=percent]))
    a = remaining_colums.index
    return df[a]

#Variables that have most of its values as same are dropped from the 'df'.
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

def coerce_data_percentile (df,coerce_var,p=5): #p is the percent that you want to coerce (over 100)
    df_coerced=df.copy()    
    for i in range(len(coerce_var)):
        p_min=np.percentile(df[coerce_var[i]],p)
        p_max=np.percentile(df[coerce_var[i]],(100-p))
        df_coerced[coerce_var[i]]=np.clip(df[coerce_var[i]],p_min,p_max) 
    return df_coerced
    
def coerce_data_Z (df,coerce_var,m=3): #m is the standard deviation that you want to coerce
    df_coerced=df[coerce_var].copy()    
    for i in range(len(coerce_var)):
        z=stats.zscore(df[coerce_var[i]])
        p_max=df[coerce_var[i]][(z > m)].min()
        p_min=df[coerce_var[i]][(z < (-1*m))].max()
        if np.isnan(p_max): p_max=df[coerce_var[i]].max()
        if np.isnan(p_min): p_min=df[coerce_var[i]].min()
        df_coerced[coerce_var[i]]=np.clip(df[coerce_var[i]],p_min,p_max)
    #df_coerced.add_prefix('CO_',inplace=True)
    return df_coerced

# This method can be used to understand the discrimination power of variables independently for feature elimination. 
# It generates univariate logit models with all of the variables separately and calculates the Gini values of models. 
# Variables which have Gini< 0.07 is considered not to distinguish the target variable and therefore not explanatory.
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

#This method does correlation elimination based on the discrimination power of variables (used Gini in this case).
#When a full correlation matrix and a Gini list given (above method gives it) it returns a new data frame where only one of the highly correlated variables remain.
#Here high correlation is representing greater than 50%. For example, if there are 2 variables with 70% correlation, 
#it checks the Gini list and keep the one with higher Gini, which is considered to distinguish the target variable better. 
#Using Gini is not necessary, you can use another value that represents the importance of variables too.
def corr_elimination(corr_matrix, gini_list): #Gini list must be ordered descending.
    columns = np.full((corr_matrix.shape[0],), True, dtype=bool)
    for i in range(corr_matrix.shape[0]):    
        if columns[i]:
            for j in range(i+1, corr_matrix.shape[0]):        
                if abs(corr_matrix.iloc[i,j]) >= 0.5:   #50% can be changes here.           
                    if (gini_list.iloc[i].values>=gini_list.iloc[j].values and columns[j]):
                        columns[j] = False
                    elif (gini_list.iloc[j].values>gini_list.iloc[i].values and columns[j]):
                        columns[i] = False
    selected_columns=corr_matrix.columns[columns]
    return selected_columns 

#The following 2 methods can be used to categorize your continuous data by using decision trees. 
#Univariate decision trees are generated separately with each variable and the buckets that trees generated are being used to 
#create new variables that are the categorical version of the original one. 

#This algorithm is just about finding the optimal depth for each tree. The output of this method will be input to the following one which generates the final trees.
# The method should be called in a for loop which scans all the variables in the 'x_train' data frame. 'var' parameter represents this loop value.
def OptDepth_DecTree(x_train,y_train,var):
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier
    score_ls = []     # here I will store the roc auc
    score_std_ls = [] # here I will store the standard deviation of the roc_auc
    for tree_depth in [1,2]:
        tree_model = DecisionTreeClassifier(max_depth=tree_depth)
        scores = cross_val_score(tree_model,x_train.iloc[:,int(var)].to_frame(),       
        y_train, cv=3, scoring='roc_auc')   
        score_ls.append(np.mean(scores))
        score_std_ls.append(np.std(scores))  
    temp = pd.concat([pd.Series([1,2]), pd.Series(score_ls), pd.Series(score_std_ls)], axis=1)
    temp.columns = ['depth', 'roc_auc_mean', 'roc_auc_std']   
    a=np.where(temp.iloc[:,1]==temp.iloc[:,1].max()) #gives the index of maximum auc
    s = int(a[0])
    #print(x_train.iloc[:,int(var)].name)
    #print(temp)
    return s+1 #gives the value that have maximum auc

#This is the main algorithm that categorizes continuous variables with the help of decision tree.
#If you want to categorize the null values separately, just leave them as null. This algorithm puts them to bucket '1'. If there is no null, bucket numbers start from '2'.
def Binnig_DecTree(x_train,y_train,opt_depth): 
    from sklearn.tree import DecisionTreeClassifier
    c=len(x_train.columns)
    for j in range(c): #column  
        x=x_train.iloc[:,j]
        #tree_model=DecisionTreeClassifier(min_samples_leaf=int(len(x)*0.1)) #if you don't want to restrict the max_depth.
        tree_model=DecisionTreeClassifier(max_depth=opt_depth[j], min_samples_leaf=int(len(x_train)*0.1)) #There should be at least 10% of the total data in a leaf
        null_index     =x[pd.isnull(x)].index
        not_null_index =x[pd.notnull(x)].index        
        x_null=x[null_index] 
        y_null=y_train[null_index]         
        x_not_null=x[not_null_index] 
        y_not_null=y_train[not_null_index]
        tree_model.fit(x_not_null.to_frame(), y_not_null)
        #Generates 'TREE_' columns which give the decision tree predictions.
        x_train.loc[not_null_index,'TREE_'+str(x.name)]=tree_model.predict_proba(x_not_null.to_frame())[:,1] 
        x_train.loc[null_index,    'TREE_'+str(x.name)]=np.mean(y_null)     
        #Shows the max value in each bucket (break point)
        bins=(pd.concat( [x_train.groupby (['TREE_'+str(x.name)]) [x.name].max()], axis=1))
        bins=np.sort(bins,0)          
        bins=[float(i) for i in bins] #float to list
        if(pd.isnull(bins[len(bins)-1])):
            del bins[len(bins)-1]
        del bins[len(bins)-1]     
        bins.insert(0,-100000000000)  #if you have greater or smaller values in your data, just change the limits manually. 
        bins.append(1000000000000)    
        labels=[i+2 for i in range(len(bins)-1)] #Labels start from '2'.               
        
        category = pd.cut(x,bins=bins,labels=labels) #Groups the variable based on bin separations.
        category = category.to_frame()
        category[x.name] = category[x.name].cat.add_categories([1])
        category[x.name].fillna(1, inplace=True) #All the nulls are being labeled as '1'.
        category.columns = ['BIN_'+x.name]       
        x_train = pd.concat([x_train,category],axis = 1)
        print(j, ' finished')
    return x_train

#This algorithm fits the train data to logistic regression in 'statsmodels' library and generates the summary table and AUC/GINI values of the model in test data.
#It also returns the p-values of each variable so that the ones greater than 0.05 can be checked easily and eliminated.
def logit_statmodels(x_train,y_train,x_test,y_test):
    import statsmodels.api as sm
    from sklearn.metrics import roc_auc_score
    #x_sm=df_final[gini_list.tail(2).index]
    x_cons=[]
    x_cons_test=[]
    x_cons=sm.add_constant(x_train)
    x_cons_test=sm.add_constant(x_test)
    y_train = list(y_train)
    y_test = list(y_test)
    logit_model=sm.Logit(y_train,x_cons)
    result=logit_model.fit()
    #print(result.summary())
    print(result.summary2())  
    #print(result.pvalues)
    #x_sm_summary_table = b_scaled.describe()
    y_pred= result.predict(x_cons_test)
    print('AUC: ' ,roc_auc_score(y_test, y_pred))
    print('GINI: ',((roc_auc_score(y_test, y_pred)-0.5)*2))
    return result.pvalues
       
