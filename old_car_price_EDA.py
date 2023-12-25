##Step 1:Importing the necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Step 2:Reading the dataset
df = pd.read_csv('car_price.csv',index_col=0)
df.head(10)

#Step 3:Analyzing the data 
#dimensions of the shape
df.shape

#datatypes of all columns in the dataset
df.dtypes

#Info on the dataset
df.info()

#Check the number of null values in the dataset
df.isna().sum()

#Handling duplicate values
dup = df.duplicated()
df.drop_duplicates(inplace=True)

#Check for null values
df.isna().sum()

#Step 4: Data Reduction
df.loc[df['ownership']=='0th Owner',df.columns]= np.nan
df.loc[df['engine']==0,'engine']=np.nan
df = df.dropna(axis=0)

#Step 5: Data Manipulation
#Functions to manipulate the data
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


df['engine'] = df['engine'].apply(valid)
df.engine = df.engine.astype(int)

df['car_prices_in_rupee'] = df['car_prices_in_rupee'].apply(space_only)
df['kms_driven'] = df['kms_driven'].apply(space_only)
df['car_prices_in_rupee'] = df['car_prices_in_rupee'].astype(float)
df['kms_driven'] = df['kms_driven'].astype('int')
df['manufacture'] = df['manufacture'].astype('int')
df.Seats = df.Seats.astype('category')

#Renaming few columns
col = {'car_prices_in_rupee':'car_price_in_lakhs','engine':'engine_in_cc'}
df.rename(columns=col,inplace=True)

df.loc[df.car_prices_in_rupee>100,'car_prices_in_rupee'] = df.car_prices_in_rupee/100000

#Step 6: Creating new features
df['car_brand'] = df['car_name'].apply(brand)
df['Age_of_car'] = 2023 - df['manufacture']

df.loc[df['car_brand']=='Land','car_brand'] = 'Land Rover'
df.loc[df['car_brand']=='Mini','car_brand'] = 'Mini Cooper'
df.loc[df['car_brand']=='Maruti','car_brand'] = 'Maruti Suzuki'

#Statistical Description of the dataset
df.describe()

#Step 7:UNIVARIATE ANALYSIS
#Categorical Analysis
cat_cols = ['fuel_type', 'transmission','ownership','Seats']

fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(12, 32))
axs = axs.ravel()  

for i, col in enumerate(cat_cols):
    sns.countplot(x=df[col], data=df, palette='bright', ax=axs[i], saturation=0.95)
    for cont in axs[i].containers:
        axs[i].bar_label(cont, color='black', size=10)
    axs[i].set_title(f'Count Plot of {col.capitalize()}')
    axs[i].set_xlabel(col.capitalize())
    axs[i].set_ylabel('Count')

plt.tight_layout()
plt.show()

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

#Step 8: BIVARIATE ANALYSIS
# 1. Numerical vs Numerical
plt.figure(figsize=(10,8))
sns.heatmap(df[num_cols].corr(), annot=True,fmt='.2f')


# 2. Categorical vs Numercial
#Categorical vs Price
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
