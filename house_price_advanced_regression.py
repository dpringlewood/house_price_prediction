"""
House price regression competition from Kaggle.

Submissions are evaluated on Root-Mean-Squared-Error(RMSE)
between the logarithm of the predicted value and the
logarithm of the observed sales price.
"""

# Handle imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 100)

house_data = pd.read_csv(r'./train.csv')
print(house_data.head())
print(house_data.info())

# Do some visual-EDA
sns.scatterplot('PoolArea', 'SalePrice', data=house_data[house_data['PoolArea'] != 0])

"""
Looking at the data initially we can see some fields have 
missing data. Some of these we can drop as they have many 
missing values, and intuitively they may not have a huge
factor on the final price of a house. These are:
Alley, MiscFeature.
"""

house_data.drop(['Alley', 'MiscFeature', 'PoolArea'], inplace=True, axis=1)

"""
Next we need to impute missing values. This would be for the
following fields: LotFrontage, MasVnrType, MasVnrArea, BsmtQual,
BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, Electrical,
FireplaceQu, GarageType, GarageYrBlt, GarageFinish, GarageQual,
GarageCond, PoolQC, Fence. 

Note, some of these may not have a value as they don't exist eg pool,
but we should make sure the data is clean anyway.

>> Replaced the single missing electrical value with the median value
>> Missing Masonry areas were assumed to not exist, replaced with None.
>> Basement values that did not exist were replaced with none, assumed they
did not exist. (checked in a scratchpad)
>> Garage values were replaced with none where there was no garage.
>> Exploring the pool data shows only 7 houses with pools, and the 
scatterplot shows no real correlation to house price. I will convert the
pool data to 'HasPool' as a label.
>> FireplaceQu has null values where there are no fireplaces in the house.
>> Fence - where there are null values assume no fence
"""

# Start with the easy stuff

fill_values = {'Electrical': house_data['Electrical'].value_counts().idxmax(),
               'MasVnrType': 'None', 'MasVnrArea': 0.0, 'BsmtQual':'NA',
               'BsmtCond':'NA', 'BsmtExposure':'NA', 'BsmtFinType1':'NA',
               'BsmtFinSF1':'NA', 'BsmtFinType2':'NA', 'BsmtFinSF2':'NA',
               'BsmtUnfSF':'NA', 'GarageType':'NA', 'GarageYrBlt':'NA',
               'GarageFinish':'NA', 'GarageQual':'NA', 'GarageCond':'NA',
               'PoolQC': 0, 'FireplaceQu': 'NA', 'Fence': 'NA'
               }
house_data.fillna(value=fill_values, inplace=True)
# Replace the PoolQC with 1
house_data[house_data['PoolQC'] != 0] = 1
print(house_data.info())

