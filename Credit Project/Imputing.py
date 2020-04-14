import os
os.chdir('/home/nmduy/CA683/Credit Project')
from fancyimpute import IterativeImputer as MICE
import pandas as pd
import numpy as np

Data_Folder = 'encoded_data/'

app_train = pd.read_csv(Data_Folder+'encoded_train.csv')

app_train_remove_target = app_train.drop(app_train.columns[-1], axis=1) # remove target
app_train_remove_target = app_train_remove_target.drop(app_train_remove_target.columns[0], axis=1) # remove SKID

imputed_app_train = MICE().fit_transform(app_train_remove_target)

imputed_pd = pd.DataFrame(data=imputed_app_train)
imputed_pd.columns = list(app_train_remove_target.columns)
imputed_pd['TARGET'] = list(app_train['TARGET'])
imputed_pd['SK_ID_CURR'] = list(app_train['SK_ID_CURR'])
imputed_pd.to_csv('imputed_train.csv', index=False)