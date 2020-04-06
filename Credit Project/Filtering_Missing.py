import os
os.chdir('/Users/duynguyen/DuyNguyen/Gitkraken/CA683/Credit Project')

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

Data_Folder = 'data/'

########### LIBRARY ############
def count_missing_row(df):
    return df.isnull().any(axis=1).sum()

def count_missing_cell(df):
    return np.count_nonzero(df.isnull().values.ravel())

def basic_info(df):
    info = {}
    info['numb_sample'] = df.shape[0]
    info['numb_ft'] = df.shape[1]
    info['numb_row_missing'] = count_missing_row(df)
    info['port_row_missing'] = round(info['numb_row_missing'] / info['numb_sample'], 4)
    info['numb_cell_missing'] = count_missing_cell(df)
    info['port_cell_missing'] = round(info['numb_cell_missing'] / (info['numb_sample'] * info['numb_ft']), 4)
    
    temp = df.isnull().sum()
    temp_keys = list(temp.keys())
    temp_val = [temp[x] for x in temp_keys]
    temp_pc = [round(temp[x]/info['numb_sample'],4) for x in temp_keys]
    temp_nb_dict = {}
    temp_pc_dict = {}
    for idx, x in enumerate(temp_keys):
        temp_nb_dict[x] = temp_val[idx]
        temp_pc_dict[x] = temp_pc[idx]

    info['column_numb_missing'] = temp_nb_dict
    info['column_port_missing'] = temp_pc_dict

    # total cell = numb_sample * numb_ft
    return info


############# PROCESSING ############
##### Read data base
entire_data = {}

entire_data['application_train'] = pd.read_csv(Data_Folder+'application_train.csv')
entire_data['application_test'] = pd.read_csv(Data_Folder+'application_test.csv')
entire_data['bureau_balance'] = pd.read_csv(Data_Folder+'bureau_balance.csv')
entire_data['bureau'] = pd.read_csv(Data_Folder+'bureau.csv')
entire_data['credit_card_balance'] = pd.read_csv(Data_Folder+'credit_card_balance.csv')
entire_data['installments_payments'] = pd.read_csv(Data_Folder+'installments_payments.csv')
entire_data['pos_cash_balance'] = pd.read_csv(Data_Folder+'pos_cash_balance.csv')
entire_data['previous_application'] = pd.read_csv(Data_Folder+'previous_application.csv')

all_data_name = list(entire_data.keys())

##### Information of each dataset
info_data = {}
for data_name in all_data_name:
    info_data[data_name] = basic_info(entire_data[data_name])
# Save
with open('basic_info.pickle', 'wb') as f:
    pickle.dump(info_data,f)
# # Load basic info
# with open('basic_info.pickle', 'rb') as f:
#     info_data = pickle.load(f)

##### Missing Portion in entire dataset
missing_portion = []
for name in all_data_name:
    if name != 'application_test':
        temp = info_data[name]['column_port_missing']
        value = [temp[x] for x in list(temp.keys())]
        missing_portion += value

# # Plot histogram to check the missing portion between features
# missing_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# missing_hist = plt.hist(missing_portion, bins = missing_bins)
# for i in range(len(missing_bins)-1):
#     plt.text((missing_hist[1][i]+missing_hist[1][i+1])/2 - 0.025,missing_hist[0][i]+2,str(int(missing_hist[0][i])))
# plt.title('Histogram of Number of feature with its Missing Portion')
# plt.xlabel('Missing Portion')
# plt.ylabel('Number of Features (Columns)')

'''
Main Points:
- Total 218 features (columns) in the entire dataset (all csv file)
- 62 / 218 features miss over 40%
- 47 / 218 features miss over 50%
'''

##### Keep data contains sk id features
# data_contain_sk_id = []
# column_name_data_contain_sk_id = []
# for data_name in all_data_name:
#     column_name = list(entire_data[data_name].columns)
#     if 'SK_ID_CURR' in column_name:
#         data_contain_sk_id.append(data_name)
#         column_name_data_contain_sk_id.append(column_name)


##### NUMERICALIZE (if load the basic_info.pickle)
'''
Main Points:
- OWN_CAR_AGE: Missing value is based on FLAG_OWN_CAR --> Not really missing entirely
'''
# OWN_CAR_AGE missing depend on FLAG_OWN_CAR --> fix it
entire_data['application_train']['FLAG_OWN_CAR'] = entire_data['application_train']['FLAG_OWN_CAR'].map({'Y': 1, 'N': 0})
entire_data['application_train']['FLAG_OWN_REALTY'] = entire_data['application_train']['FLAG_OWN_REALTY'].map({'Y': 1, 'N': 0})
entire_data['application_train']['OWN_CAR_AGE'] = entire_data['application_train'].apply(
    lambda row: -1 if np.isnan(row['OWN_CAR_AGE']) and row['FLAG_OWN_CAR'] == 0 else row['OWN_CAR_AGE'],
    axis=1
)

entire_data['application_test']['FLAG_OWN_CAR'] = entire_data['application_test']['FLAG_OWN_CAR'].map({'Y': 1, 'N': 0})
entire_data['application_test']['FLAG_OWN_REALTY'] = entire_data['application_test']['FLAG_OWN_REALTY'].map({'Y': 1, 'N': 0})
entire_data['application_test']['OWN_CAR_AGE'] = entire_data['application_test'].apply(
    lambda row: -1 if np.isnan(row['OWN_CAR_AGE']) and row['FLAG_OWN_CAR'] == 0 else row['OWN_CAR_AGE'],
    axis=1
)

entire_data['application_train']['CODE_GENDER'] = entire_data['application_train']['CODE_GENDER'].map({'M': 1, 'F': 0})
entire_data['application_test']['CODE_GENDER'] = entire_data['application_test']['CODE_GENDER'].map({'M': 1, 'F': 0})

entire_data['application_train']['NAME_CONTRACT_TYPE'] = entire_data['application_train']['NAME_CONTRACT_TYPE'].map({'Cash loans': 1, 'Revolving loans': 0})
entire_data['application_test']['NAME_CONTRACT_TYPE'] = entire_data['application_test']['NAME_CONTRACT_TYPE'].map({'Cash loans': 1, 'Revolving loans': 0})

# Adjust the OWN_CAR_AGE again
info_data['application_train'] = basic_info(entire_data['application_train'])
info_data['application_test'] = basic_info(entire_data['application_test'])
# Save
with open('basic_info.pickle', 'wb') as f:
    pickle.dump(info_data,f)

##### REMOVE FEATURE containing missing portion over the threshold (0.4)
missing_threshold = 0.4
filtered_entire_data = {}
drop_column = {}
for name in all_data_name:
    temp = entire_data[name]
    info = info_data[name]
    column_port_missing = info['column_port_missing']
    list_drop_column = []
    for key, value in column_port_missing.items():
        if value >= missing_threshold:
            list_drop_column.append(key)
    temp = temp.drop(list_drop_column, axis=1)
    filtered_entire_data[name] = temp
    drop_column[name] = list_drop_column
print(drop_column)

# Delete Entire dataset --> only use filtered data
del entire_data
del info_data

##### Information of filtered dataset
filtered_info_data = {}
for data_name in all_data_name:
    filtered_info_data[data_name] = basic_info(filtered_entire_data[data_name])
# Save
with open('filtered_basic_info.pickle', 'wb') as f:
    pickle.dump(filtered_info_data,f)
# # Load basic info
# with open('filtered_basic_info.pickle', 'rb') as f:
#     filtered_info_data = pickle.load(f)

##### Write the filtered dataset #####
for name in all_data_name:
    filtered_entire_data[name].to_csv(f"filtered_{name}.csv")

