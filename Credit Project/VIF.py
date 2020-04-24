import os
os.chdir('/home/nmduy/CA683/Credit Project')
#from fancyimpute import IterativeImputer as MICE
import pandas as pd
import numpy as np
import random
from statsmodels.stats.outliers_influence import variance_inflation_factor 
import json

data = pd.read_csv('data/encoded_train_rmcor.csv')
#data_remove_ft = data.drop(['SK_ID_CURR', 'TARGET'], axis=1)
data = data.dropna()
data_remove_ft = data.drop(['TARGET'], axis=1)

print(f"Original rows / Dropna rows: {len(data)} / {len(data_remove_ft)}")

categorical_count = data_remove_ft.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
categorical_name = list(categorical_count.keys())
continuous_name = [x for x in list(data_remove_ft.columns) if x not in categorical_name]
print(f"Number of Continuous / Category: {len(continuous_name)} / {len(categorical_name)}")

def calculate_vif_(X, thresh=100):
    result = {}
    cols = X.columns
    variables = np.arange(X.shape[1])
    dropped=True
    while dropped:
        dropped=False
        c = X[cols[variables]].values
        c = np.array(c, dtype=float)
        vif = [variance_inflation_factor(c, ix) for ix in np.arange(c.shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X[cols[variables]].columns[maxloc] + '\' at index: ' + str(maxloc) + '\' VIF: ' + str(max(vif)))
            result[X[cols[variables]].columns[maxloc]] = str(max(vif))
            variables = np.delete(variables, maxloc)
            dropped=True
            
    return X[cols[variables]], result

data_checked_vif, dict_result = calculate_vif_(data_remove_ft, thresh=5)
with open('vif_rmcor_5.json', 'w') as f:
    json.dump(dict_result, f)

#data_checked_vif['SK_ID_CURR'] = list(data['SK_ID_CURR'])
data_checked_vif['TARGET'] = list(data['TARGET'])

data_checked_vif.to_csv('encoded_train_rmcor_vif5.csv', index=False)

print(data_checked_vif.shape)
