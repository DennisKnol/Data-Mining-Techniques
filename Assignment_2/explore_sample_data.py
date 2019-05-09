import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from functions import missing_values, convert_date_time, combine_competitors, find_outlier, remove_outlier


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

print("Number of unique search IDs:\n ", len(df["srch_id"].unique()))
print("Number of unique hotel IDs:\n ", len(df["prop_id"].unique()))

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
# TODO: relatief verschil plotten


# df = combine_competitors(df)

# Correlation matrices and corresponding heat maps
columns = ['srch_id', 'date_time', 'site_id', 'visitor_location_country_id',
           'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id',
           'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool',
           'prop_location_score1', 'prop_location_score2',
           'prop_log_historical_price', 'position', 'price_usd', 'promotion_flag',
           'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window',
           'srch_adults_count', 'srch_children_count', 'srch_room_count',
           'srch_saturday_night_bool', 'srch_query_affinity_score',
           'orig_destination_distance', 'random_bool', 'click_bool',
           'gross_bookings_usd', 'booking_bool', 'date', 'time',
           'year', 'month', 'day', 'delta_price', 'family_count'
           ]

corr = df[columns].corr()
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
)
plt.show()

# correlation for all with booking_bool = 1
corr_booked = df[columns][df["booking_bool"] == 1].corr()
plt.figure(figsize=(15, 10))
ax = sns.heatmap(
    corr_booked,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
plt.show()

# correlation for all with booking_bool = 0
corr_not_booked = df[columns][df["booking_bool"] == 0].corr()
plt.figure(figsize=(15, 10))
ax = sns.heatmap(
    corr_not_booked,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
plt.show()

# plot absolute value of difference in correlations when booked or not booked
delta_corr = (corr_booked - corr_not_booked)

plt.figure(figsize=(15, 10))
ax = sns.heatmap(
    delta_corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
plt.show()

# Count plots
attributes = ["visitor_location_country_id", "prop_country_id", "prop_starrating", "prop_brand_bool"]
for i in attributes:
    sns.countplot(i, data=df, alpha=0.6)
    plt.show()


# Density plots
attributes = [
    "price_usd", "prop_review_score", "prop_location_score1",
    "prop_location_score2", "position", "orig_destination_distance"
]

for i in attributes:
    sns.distplot(df[i][df["booking_bool"] == 1].dropna(), label="booked", kde=False, norm_hist=True)
    sns.distplot(df[i][df["booking_bool"] == 0].dropna(), label="not booked", kde=False, norm_hist=True)
    plt.legend()
    plt.show()


# box plots
y_list = ["position", "click_bool"]
for y in y_list:
    sns.boxplot(x="booking_bool", y=y, data=df)
    plt.show()


# Find outliers
outlier_rows = find_outlier(df)
number_of_outlier_row = len(outlier_rows)

