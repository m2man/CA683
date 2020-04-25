import os
os.chdir('/home/nmduy/CA683/Credit Project')
from fancyimpute import IterativeImputer as MICE
import pandas as pd
import numpy as np

data = pd.read_csv('data/merged_application_train.csv')
data_remove_id = data.drop(['SK_ID_CURR'], axis=1) if 'SK_ID_CURR' in list(data.columns) else data.copy()
data_remove_id = data_remove_id.drop(['TARGET'], axis=1) if 'TARGET' in list(data_remove_id.columns) else data_remove_id.copy()
data_remove_id = data_remove_id.drop(['DAYS_EMPLOYED_ANOM'], axis=1) if 'DAYS_EMPLOYED_ANOM' in list(data_remove_id.columns) else data_remove_id.copy()
data_remove_id = data_remove_id.drop(['Unnamed: 0'], axis=1) if 'Unnamed: 0' in list(data_remove_id.columns) else data_remove_id.copy()

encoded_data = pd.get_dummies(data_remove_id)

imputed_app_train = MICE().fit_transform(encoded_data)

imputed_pd = pd.DataFrame(data=imputed_app_train)
imputed_pd.columns = list(encoded_data.columns)
imputed_pd['TARGET'] = list(data['TARGET'])
imputed_pd['SK_ID_CURR'] = list(data['SK_ID_CURR'])
imputed_pd.to_csv('data/imputed_merged_train.csv', index=False)
