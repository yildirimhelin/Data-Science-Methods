
import pandas as pd
import numpy as np

#Gives the percentage of missing values for all variables of 'df'. 
def missing_percent(df):
    missing = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([missing, percent], axis=1, keys=['Missing', 'Percent'])
    return missing_data
