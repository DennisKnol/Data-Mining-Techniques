import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from functions import missing_values, convert_date_time, combine_competitors

"""
This file is to explore a small fraction of the data to reduce the run time.

Methods from this file are to be applied on the complete data set later.

"""


df = pd.read_csv("df_sample.csv")
df = df.drop(df.columns[0], axis=1)

print(df.info())
print(df.columns)
print("Shape of the dataframe: ", df.shape, "\n")  # (1000, 54)
print("Count and percentage of is null values: \n", missing_values(df))

print("Number of unique search IDs:\n ", df["search_id"].unique())
print("Number of unique hotel IDs:\n ", df["prop_id"].unique())

booking_percentage = 100 * (df["booking_bool"].sum()/df.shape[0])
clicking_percentage = 100 * (df["click_bool"].sum()/df.shape[0])

df = convert_date_time(df)
sns.countplot('month', data=df)
plt.show()

df["position"][df["booking_bool"] == 1].value_counts(normalize=True).sort_index().plot(kind='bar')
plt.show()

df["position"][df["click_bool"] == 1].value_counts(normalize=True).sort_index().plot(kind='bar')
plt.show()

# create and plot new column with the total number of individuals in family / group
df["family_count"] = df["srch_children_count"] + df["srch_adults_count"]
sns.countplot("family_count", data=df)
plt.show()

# plot of absolute value of the difference between prop price and the mean price of previous stays
df["delta_price"] = np.abs(df["price_usd"] - df["visitor_hist_adr_usd"])
sns.distplot(df["delta_price"][df["booking_bool"] == 1].dropna())
plt.show()


df = combine_competitors(df)

corr = df.corr()
plt.figure(figsize=(15, 10))
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
plt.show()

# Count plots
sns.countplot('visitor_location_country_id', data=df)
plt.show()

sns.countplot('prop_country_id', data=df)
plt.show()

sns.countplot('prop_starrating', data=df)
plt.show()

sns.countplot('prop_brand_bool', data=df)
plt.show()


# Density plots
sns.distplot(df['price_usd'][df["booking_bool"] == 1], label="booked")
sns.distplot(df['price_usd'][df["booking_bool"] == 0], label="not booked")
plt.legend()
plt.show()

sns.distplot(df["orig_destination_distance"][df["booking_bool"] == 1].dropna())
plt.show()

# box plots
sns.boxplot(x="booking_bool", y="position", data=df)
plt.show()

sns.boxplot(x="booking_bool", y="click_bool", data=df)
plt.show()
