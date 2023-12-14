#Importing the necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Reading the dataset
df = pd.read_csv('car_price.csv',index_col=0)
df.head(10)

#dimensions of the shape
df.shape

#datatypes of all columns in the dataset
df.dtypes

#Info on the dataset
df.info()

#Check the number of null values in the dataset
df.isna().sum()

#Enables us to display all rowns and columns of the data
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#Handling duplicate values
dup = df.duplicated()
df.drop_duplicates(inplace=True)

#Here we are deleting the rows that contain 0th owner, as those cars are new, and we are dealing with prices of old cars
#Converting the rows that contain 0th owner of ownership into null values
df.loc[df['ownership']=='0th Owner',df.columns]= np.nan
df = df.dropna(axis=0)

#Functions valid,brand and space_only are created to split the data under the column and also convert those with object datatype into integer/float datatypes
def valid(data):
    data = data.split(' ')
    data = data[0]
    return float(data)

def brand(data):
    data = data.split(' ')
    data = data[0]
    return data

def space_only(data):
    data = data.split(' ')
    data = data[0]
    data = data.replace(',','')
    return float(data)

#Applying the above functions to the respective columns and changing their datatypes
#valid
df['engine'] = df['engine'].apply(valid)
df.engine = df.engine.astype(int)

#space_only
df['car_prices_in_rupee'] = df['car_prices_in_rupee'].apply(space_only)
df['kms_driven'] = df['kms_driven'].apply(space_only)
df['car_prices_in_rupee'] = df['car_prices_in_rupee'].astype(float)
df['kms_driven'] = df['kms_driven'].astype('int')
df['manufacture'] = df['manufacture'].astype('int')
df.Seats = df.Seats.astype('category')

#Here in car_prices_in rupee some values are not in terms of lakhs, so we divide the values greater than 100 by a lakh
df.loc[df.car_prices_in_rupee>100,'car_prices_in_rupee'] = df.car_prices_in_rupee/100000

#On observing the engine column we find the rows that have 0 cc, which is not possible in cars, so those rows are also to be deleted
df.loc[df['engine']==0,'engine']=np.nan
df= df.dropna(axis=0)

#Creating new columns like car_brand and Age_of_car, to make the data more informative
df['car_brand'] = df['car_name'].apply(brand)
df['Age_of_car'] = 2023 - df['manufacture']

#In car_brand, some of the values like Land, Mini and Maruti are incomplete
#Hence we are renaming the respective values
df.loc[df['car_brand']=='Land','car_brand'] = 'Land Rover'
df.loc[df['car_brand']=='Mini','car_brand'] = 'Mini Cooper'
df.loc[df['car_brand']=='Maruti','car_brand'] = 'Maruti Suzuki'

#Also renaming a few columns 
col = {'car_prices_in_rupee':'car_price_in_lakhs','engine':'engine_in_cc'}
df.rename(columns=col,inplace=True)

#Statistical Description of the dataset
df.describe()
#Analysis so far
# 1. Total number of car details is 5412
# 2. Values like mean, std and iqr vary from one column to another

#UNIVARIATE ANALYSIS
#Categorical Analysis
# Define the list of categorical columns to analyze
cat_cols = ['fuel_type', 'transmission','ownership','Seats']

# Create subplots
fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(12, 32))
axs = axs.ravel()  

# Loop through each categorical column
for i, col in enumerate(cat_cols):
    sns.countplot(x=df[col], data=df, palette='bright', ax=axs[i], saturation=0.95)
    for cont in axs[i].containers:
        axs[i].bar_label(cont, color='black', size=10)
    axs[i].set_title(f'Count Plot of {col.capitalize()}')
    axs[i].set_xlabel(col.capitalize())
    axs[i].set_ylabel('Count')
    
# Adjust layout and show plots
plt.tight_layout()
plt.show()

#Due to excessive amount of brands in car_brand, it is visualized seperately
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
sns.countplot(x=df['car_brand'], data=df, palette='bright',ax=axes)
for container in axes.containers:
        axes.bar_label(container, color='black', size=10)
X=plt.gca().xaxis
for item in X.get_ticklabels():
  item.set_rotation(90)
plt.show()

#Numerical analysis
num_cols = ['kms_driven','engine_in_cc','Age_of_car','car_price_in_lakhs']

plt.figure(figsize=(12, 8))
for col in num_cols:
    plt.subplot(3, 5, num_cols.index(col) + 1)
    sns.histplot(data=df[col], bins=20, kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()

#Observation:
# 1. People prefered to buy 5-seater cars compared to other cars
# 2. Manual cars were bought more than the Automatic ones
# 3. Prices varied from 0.5 lakhs to 95 lakhs
# 4. High demand for petrol cars, after diesel cars 
# 5. Maurti Suzuki cars were bought more than the other brand cars

#BIVARIATE ANALYSIS
# 1. Numerical vs Numerical
#Creating a correlation heatmap with numerical columns
plt.figure(figsize=(10,8))
sns.heatmap(df[num_cols].corr(), annot=True,fmt='.2f')


# 2. Categorical vs Numercial
#categorical columns vs price
plt.figure(figsize=(12, 32))
for feature in cat_cols:
    plt.subplot(4,1, cat_cols.index(feature) + 1)
    sns.boxplot(x=df[feature], y=df['car_price_in_lakhs'])
    plt.title(f'{feature} vs. Car Price in lakhs')
plt.tight_layout()
plt.show()

#car_brand vs price
plt.figure(figsize=(12, 9))
sns.boxplot(x=df['car_brand'], y=df['car_price_in_lakhs'])
plt.title('Car Brand vs. Car Price in lakhs')
X=plt.gca().xaxis
for item in X.get_ticklabels():
  item.set_rotation(90)
plt.show()

# 3. Categorical vs Categorical
#fuel type vs transmission
df.groupby(['fuel_type','transmission']).agg({'car_price_in_lakhs':np.mean}).plot.bar(color = 'red')
X=plt.gca().xaxis

for item in X.get_ticklabels():
  item.set_rotation(90)

#car_brand vs transmission
df.groupby(['car_brand','transmission']).agg({'car_price_in_lakhs':np.mean}).plot.bar(color = 'yellow')
X=plt.gca().xaxis

for item in X.get_ticklabels():
  item.set_rotation(90)

#Final Analysis:
#Automatic Cars cost a lot more than Manual Cars
#Petrol and Diesel cars cost more than othe rfuel cars
#From the heatmap, we have found that their is a slightly positive correlation between the engine_in_cc and the car_price_in_lakhs

#After further normalization and feature scaling, this data will be available for modelling
