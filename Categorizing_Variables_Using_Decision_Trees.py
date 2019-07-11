from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd

cancer= load_breast_cancer()
data = cancer.data
column_names = cancer.feature_names
x = pd.DataFrame(data, columns=column_names)
y = cancer.target

#The following 2 methods can be used to categorize continuous variables by using decision trees. 
#Univariate decision trees are generated separately with each variable and the buckets that trees generated are being used to 
#create new variables that are the categorical versions of the original ones. 

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

#Determines the optimum depth for each variable and keeps them in a list.
opt_depth=[]
for i in range(len(x.columns)): 
    print("started",i)
    opt_depth.append(OptDepth_DecTree(x,y,i))
    print("finished",i)

#Generates a dataframe that consists of all categorized versions of variables. 
#'1' is the category that keeps null's. If the variable has no null, categories start from '2'.
x_categorized=Binnig_DecTree(x,y,opt_depth)

#After categorizing, you can use the categories as continuous values where the values are the predictions of trees 
#(in this case, mean of target in the corresponding bucket).This data frame is x_TREE.
# On the other hand, you can continue with categorical values which is x_BIN.  
x_TREE=x_categorized.iloc[:,x_categorized.columns.str.contains('TREE_')]
x_BIN=x_categorized.iloc[:,x_categorized.columns.str.contains('BIN_')]

