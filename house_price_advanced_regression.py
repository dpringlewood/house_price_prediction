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
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


pd.set_option('display.max_columns', 100)

house_data = pd.read_csv(r'./train.csv')
# print(house_data.head())
# print(house_data.info())

# Do some visual-EDA
# sns.scatterplot('PoolArea', 'SalePrice', data=house_data[house_data['PoolArea'] != 0])

"""
Looking at the data initially we can see some fields have 
missing data. Some of these we can drop as they have many 
missing values, and intuitively they may not have a huge
factor on the final price of a house. These are:
Alley, MiscFeature.
* I leave MiscFeature Value in there, as I assume that the
value of the MiscFeature is the same to the buyer and seller.
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

"""
The last bit of data to impute is the LotFrontage parameter. My guess is that
this can be estimated by the LotArea, but I want to explore the data visually
first to support my hypothesis, and see if there is any outliers etc
"""

# sns.scatterplot(data=house_data, x='LotArea', y='LotFrontage')
# plt.show()
print(house_data['LotFrontage'].sort_values(ascending=False).head(25))

"""
We can see that there seems to be a linear relationship between the size of the lot
and the Lot Frontage. However this is skewed by some outliers. Potentially we can
look at getting rid of the outliers to make our imputation more accurate.
"""

print(house_data[house_data['LotFrontage'].isnull()]['LotArea'].describe())
print(house_data[house_data['LotFrontage'].isnull()]['LotArea'].sort_values(ascending=False))

lot_frontage_mask = (house_data['LotFrontage'].isnull()) & (house_data['LotArea'] < 40000)
sns.swarmplot(data=house_data[lot_frontage_mask], y='LotArea', x='MSZoning')
plt.show()

sns.scatterplot(data=house_data[house_data['LotArea'] < 40000],
                x='LotArea', y='LotFrontage')
plt.show()


"""
I'm going to try something new here to test my skills and hopefully come up with a more
accurate estimation of the LotFrontage variable. I will actually fit it to a simple linear
regression against the LotArea, and use that to calculate the missing values. I will exclude
outliers with a LotArea over 40,000. This is only 6 properties but skews the data.
"""

# Initialize the LinearRegression model

linear = LinearRegression()

# Fit the regression model on my available data & filter out outliers
train_mask = (house_data['LotArea']<40000) & (house_data['LotFrontage'].notnull())
X_train = np.array(house_data[train_mask]['LotArea'])
y_train = np.array(house_data[train_mask]['LotFrontage'])

# Reshape my data to a 2D array for sklearn
X_train = X_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

# Fit my model
linear.fit(X_train, y_train)
print('Intercept is: ', linear.intercept_)
print('Coefficient is: ',linear.coef_)
print(linear.score(X_train, y_train))

"""
The R-squared value is only about 0.4, I've seen better. This will do for now.
Now that I can estimate the LotFrontage given the LotArea I can fill in the blanks. This
is a LinearRegression with a single coefficent, so I might just do this manually rather than
use the .predict() method
"""
# Fill in the missing values
for i in house_data[house_data['LotFrontage'].isnull()].index:
     house_data.loc[i, 'LotFrontage'] = (house_data.loc[i, 'LotArea']*linear.coef_[0][0])+linear.intercept_[0]

"""
Data wrangling is far from over. Even though I now have a complete dataset, the quality
of the data has to be inspected, and cleaned up.
"""

# Take a quick look that all the qualitative data matches the description.
for i in house_data.select_dtypes('object').columns:
    print(i)
    print(house_data[i].value_counts())

"""
It looks OK, there are a few 1 values, and I assume that this is where there is no
data. Not really and issue as these will be converted into dummy variables for a 
linear regression in the future.

Lets look at the numerical values to see if there are any obvious correlations in 
the data, or any massive ourliers. It will also be good to see how our sales price 
is distributed.
"""

sns.distplot(house_data['SalePrice'])

"""
Says something about the price
"""
# Get the dummies variables and assign to the house_data df.
house_data = pd.get_dummies(house_data, drop_first=True)

#Get our target variable 'SalePrice'
X = np.array(house_data.drop('SalePrice', axis=1))
y = np.array(house_data['SalePrice'])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
y_train = y_train.reshape(-1, 1)
linear = LinearRegression()
lasso = Lasso(alpha=1, normalize=True, max_iter=100000)

# steps = [('scaler', StandardScaler()),
#           ('linear', LinearRegression())]
# pipeline = Pipeline(steps)
# linear_scaled = pipeline.fit(X_train, y_train)
# print('You are only so good: ', linear_scaled.score(X_test, y_test))
linear.fit(X_train, y_train)
lasso.fit(X_train, y_train)
print('Linear Score is: ',linear.score(X_test, y_test))
print('Lasso Score is: ', lasso.score(X_test, y_test))

y_pred_linear = linear.predict(X_test)
y_pred_lasso = lasso.predict(X_test)

print('MSE Linear: ', (mean_squared_error(y_test, y_pred_linear)**(1/2)))
print('MSE Linear: ', (mean_squared_error(y_test, y_pred_lasso)**(1/2)))




