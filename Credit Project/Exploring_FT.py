import os
os.chdir('/Users/duynguyen/DuyNguyen/Gitkraken/CA683/Credit Project')

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
import seaborn as sns

# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

Data_Folder = 'filtered_data/'

app_train = pd.read_csv(Data_Folder+'filtered_application_train.csv')
app_test = pd.read_csv(Data_Folder+'filtered_application_test.csv')
app_train.drop(app_train.columns[0], axis=1, inplace=True) # remove the index column created by writing file
app_test.drop(app_test.columns[0], axis=1, inplace=True)

##### OUTPUT DISTRIBUTION #####
output_count = app_train['TARGET'].value_counts()

app_train['TARGET'].astype(int).plot.hist()

##### FINDING CATEGORICAL AND CONTINUOUS FT #####
# We already converted binary categorical classes from Analyzing_Process.py
categorical_count = app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
categorical_name = list(categorical_count.keys())
continuous_name = [x for x in list(app_train.columns) if x not in categorical_name]

##### FINDING ANOMALIES #####
# PLOT BOXPLOT OF EACH FT
for ft in continuous_name[1:]:
    ms = 365 if ft[0:4] == 'DAYS' else 1
    ft_title = ft+"_YEARS" if ft[0:4] == 'DAYS' else ft
    print(f'Ploting Boxplot of {ft}')
    fig = (app_train[ft]/ms).plot.box(title=f"{ft}")
    fig.figure.savefig(f"figures/box/{ft}.png")
    fig.figure.clf()

# PLOT HISTOGRAM OF EACH FT
for ft in continuous_name[1:]:
    ms = 365 if ft[0:4] == 'DAYS' else 1
    ft_title = ft+"_YEARS" if ft[0:4] == 'DAYS' else ft
    print(f'Ploting histogram of {ft}')
    fig = (app_train[ft]/ms).plot.hist(title=f"{ft}")
    fig.figure.savefig(f"figures/hist/{ft}.png")
    fig.figure.clf()

##### CHANGING SOME FT #####
# convert to positive
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
app_test['DAYS_BIRTH'] = abs(app_test['DAYS_BIRTH'])

app_train['DAYS_EMPLOYED'] = abs(app_train['DAYS_EMPLOYED'])
app_test['DAYS_EMPLOYED'] = abs(app_test['DAYS_EMPLOYED'])

app_train['DAYS_ID_PUBLISH'] = abs(app_train['DAYS_ID_PUBLISH'])
app_test['DAYS_ID_PUBLISH'] = abs(app_test['DAYS_ID_PUBLISH'])

app_train['DAYS_LAST_PHONE_CHANGE'] = abs(app_train['DAYS_LAST_PHONE_CHANGE'])
app_test['DAYS_LAST_PHONE_CHANGE'] = abs(app_test['DAYS_LAST_PHONE_CHANGE'])

app_train['DAYS_REGISTRATION'] = abs(app_train['DAYS_REGISTRATION'])
app_test['DAYS_REGISTRATION'] = abs(app_test['DAYS_REGISTRATION'])

# Assign NaN for wrong number
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243
app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

##### FINDING CORRELATION #####
## Only work for continous ft
# Correlation with TARGET
correlations = app_train[[x for x in continuous_name if x not in ['SK_ID_CURR']]].corr()['TARGET'].sort_values()
# Correlation between variables
correlations_internal = app_train[[x for x in continuous_name if x not in ['TARGET', 'SK_ID_CURR']]].corr()

##### EXPLORATION WITHIN FEATURE #####
for ft in continuous_name[1:]:
    try:
        ms = 365 if ft[0:4] == 'DAYS' else 1
        sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, ft] / ms, label = 'target == 0')
        sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, ft] / ms, label = 'target == 1')
        plt.ylabel('Density')
        plt.title(f'Distribution of {ft}')
        plt.savefig(f'figures/ft_dist_target/{ft}.png')
        plt.clf()
    except:
        print(f"Error in {ft}")

##### ENCODE CATEGORICAL VARIABLES #####
# Create one-hot encoding --> also solve the missing problem in categorical variables
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

# Align train and test (since maybe some value of categorical in train not in test)
train_labels = app_train['TARGET']
app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1) # keep column appear both dataset
app_train['TARGET'] = train_labels