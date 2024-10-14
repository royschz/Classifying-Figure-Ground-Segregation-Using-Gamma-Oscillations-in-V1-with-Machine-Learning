#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:25:40 2024

@author: rodrigosanchez
"""

#%% Data cleaning 
import pandas as pd

data = pd.read_csv('/Users/rodrigosanchez/Documents/ScientificProgramming/Experiment.csv')
#%% Missing values 
missing = data.isnull().sum()
print("Missing values in each column:\n", missing)
dataCleaned1 = data.dropna()

#%% Duplicates 
dupli = dataCleaned1.duplicated().sum()
print(f"Number of duplicate rows: {dupli}")
dataCleaned1 = dataCleaned1.drop_duplicates()

#%% Outliers 
def remove_outliers(data, columns):
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    return data   

dataCleaned1 = remove_outliers(dataCleaned1, ['ContrastHeterogeneity', 'GridCoarseness'])

#%% check 
print(dataCleaned1.dtypes)

dataCleaned1['SubjectID'] = dataCleaned1['SubjectID'].astype(int)
dataCleaned1['SessionID'] = dataCleaned1['SessionID'].astype(int)
dataCleaned1['BlockID'] = dataCleaned1['BlockID'].astype(int)
    
#%% Save the data clenaed 
dataCleaned1.to_csv('/Users/rodrigosanchez/Documents/ScientificProgramming/cleaned_data.csv', index=False)
