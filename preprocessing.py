import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

plt.style.use('seaborn')
import seaborn as sns

import re
import xgboost as xgb

import warnings

warnings.filterwarnings("ignore")

df_initial = pd.read_csv('berlin-airbnb-data/listings_summary.csv')

# checking shape
print("The dataset has {} rows and {} columns.".format(*df_initial.shape))

# ... and duplicates
print("It contains {} duplicates.".format(df_initial.duplicated().sum()))

print(df_initial.head(1))

# check the columns we currently have
print(df_initial.columns)

# define the columns we want to keep
columns_to_keep = ['id', 'space', 'description', 'host_has_profile_pic', 'neighbourhood_group_cleansed',
                   'latitude', 'longitude', 'property_type', 'room_type', 'accommodates', 'bathrooms',
                   'bedrooms', 'bed_type', 'amenities', 'square_feet', 'price', 'cleaning_fee',
                   'security_deposit', 'extra_people', 'guests_included', 'minimum_nights',
                   'instant_bookable', 'is_business_travel_ready', 'cancellation_policy']

df_raw = df_initial[columns_to_keep].set_index('id')
print("The dataset has {} rows and {} columns - after dropping irrelevant columns.".format(*df_raw.shape))

# By the way, how many different **room types** do we have?
print("Details about room types")
print(df_raw.room_type.value_counts(normalize=True))

# And how many different **property types** are we up against?
print("Details about property types")
print(df_raw.property_type.value_counts(normalize=True))

# Cleaning price columns
print("Displaying cleaning price columns")
print(df_raw[['price', 'cleaning_fee', 'extra_people', 'security_deposit']].head(3))

# checking Nan's in "price" column
print("Number of NaNs in price column")
print(df_raw.price.isna().sum())

# Nan's in "cleaning_fee" column
print("Number of NaNs in cleaning fee column")
print(df_raw.cleaning_fee.isna().sum())
df_raw.cleaning_fee.fillna('$0.00', inplace=True)
print("Number of NaNs in cleaning fee column after processing")
print(df_raw.cleaning_fee.isna().sum())

# Nan's in "security_deposit" column
print("Number of NaNs in security deposit column")
print(df_raw.security_deposit.isna().sum())
df_raw.security_deposit.fillna('$0.00', inplace=True)
print("Number of NaNs in security deposit column after processing")
print(df_raw.security_deposit.isna().sum())

print("Number of NaNs in extra column")
print(df_raw.extra_people.isna().sum())

# Let's remove the dollar signs in all four columns and convert the string values into numerical ones:
# clean up the columns (by method chaining)
df_raw.price = df_raw.price.str.replace('$', '').str.replace(',', '').astype(float)
df_raw.cleaning_fee = df_raw.cleaning_fee.str.replace('$', '').str.replace(',', '').astype(float)
df_raw.security_deposit = df_raw.security_deposit.str.replace('$', '').str.replace(',', '').astype(float)
df_raw.extra_people = df_raw.extra_people.str.replace('$', '').str.replace(',', '').astype(float)

# We shouldn't miss investigating the `price` - it might need some cleaning to be of use to us:
print("Price description")
print(df_raw['price'].describe())
red_square = dict(markerfacecolor='r', markeredgecolor='r', marker='.')
df_raw['price'].plot(kind='box', xlim=(0, 1000), vert=False, flierprops=red_square, figsize=(16, 2));

df_raw.drop(df_raw[(df_raw.price > 400) | (df_raw.price == 0)].index, axis=0, inplace=True)

print("Price description after processing")
print(df_raw['price'].describe())
print("The dataset has {} rows and {} columns - after being price-wise preprocessed.".format(*df_raw.shape))
plt.show()
