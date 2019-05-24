import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from functions import *


df = pd.read_csv("training_set_VU_DM.csv")
df_prepped = pd.read_csv("prepped_training_set_VU_DM.csv")

gross_bookings_usd = 97.208949
visitor_hist_starrating = 94.920364
visitor_hist_adr_usd = 94.897735
srch_query_affinity_score = 93.598552
orig_destination_distance = 32.425766
prop_location_score2 = 21.990151
prop_review_score = 0.148517

x = [
    gross_bookings_usd, visitor_hist_starrating, visitor_hist_adr_usd,
    srch_query_affinity_score, orig_destination_distance, prop_location_score2, prop_review_score
]


x_names = np.array(['gross_bookings_usd', 'visitor_hist_starrating', 'visitor_hist_adr_usd',
                    'srch_query_affinity_score', 'orig_destination_distance', 'prop_location_score2',
                    'prop_review_score'
                    ]
                   )

perc = np.linspace(0,100,len(x))

plt.figure(figsize=(13, 5))
ax = sns.barplot(perc, x, alpha=0.8, palette=["lightblue", "teal", "wheat", "beige", "grey", "lavender"])
ax.set_xticklabels(
    x_names,
    rotation=20,
    horizontalalignment='right'
)
ax.set_ylabel('percentage')
ax.set_title('Percentage of Missing Values per Feature')
plt.show()

mean_price_booked = df[df["booking_bool"] == 1]["price_usd"].mean()
mean_prop_location_score1_booked = df[df["booking_bool"] == 1]["prop_location_score1"].mean()
mean_prop_location_score2_booked = df[df["booking_bool"] == 1]["prop_location_score2"].mean()

mean_price_not_booked = df[df["booking_bool"] == 0]["price_usd"].mean()
mean_prop_location_score1_not_booked = df[df["booking_bool"] == 0]["prop_location_score1"].mean()
mean_prop_location_score2_not_booked = df[df["booking_bool"] == 0]["prop_location_score2"].mean()

df_prepped["log_prop_location_score2"] = - np.log(df_prepped["prop_location_score2"]+0.0001)


plt.figure(figsize=(14, 8))

plt.subplot2grid((2, 2), (0, 0))
sns.distplot(
    df_prepped["price_usd"][df_prepped["booking_bool"] == 1].dropna(),
    label="booked", kde=False, norm_hist=True, color="teal")
sns.distplot(
    df_prepped["price_usd"][df_prepped["booking_bool"] == 0].dropna(),
    label="not booked", kde=False, norm_hist=True, color="grey")
plt.legend()

plt.subplot2grid((2, 2), (0, 1))
sns.distplot(
    df_prepped["prop_location_score2"][df_prepped["booking_bool"] == 1].dropna(),
    label="booked", kde=False, norm_hist=True, color="teal")
sns.distplot(
    df_prepped["prop_location_score2"][df_prepped["booking_bool"] == 0].dropna(),
    label="not booked", kde=False, norm_hist=True, color="grey")
plt.legend()

plt.subplot2grid((2, 2), (1, 0))
sns.distplot(
    df_prepped["prop_starrating"][df_prepped["booking_bool"] == 1].dropna(),
    label="booked", kde=False, norm_hist=True, color="teal")
sns.distplot(
    df_prepped["prop_starrating"][df_prepped["booking_bool"] == 0].dropna(),
    label="not booked", kde=False, norm_hist=True, color="grey")
plt.legend()

plt.subplot2grid((2, 2), (1, 1))
sns.distplot(
    df_prepped["prop_review_score"][df_prepped["booking_bool"] == 1].dropna(),
    label="booked", kde=False, norm_hist=True, color="teal")
sns.distplot(
    df_prepped["prop_review_score"][df_prepped["booking_bool"] == 0].dropna(),
    label="not booked", kde=False, norm_hist=True, color="grey")
plt.legend()

plt.show()

df["promotion_flag"][df["click_bool"] == 1].value_counts(normalize=True).sort_index().plot(kind='bar', color='blue')
df["promotion_flag"][df["click_bool"] == 0].value_counts(normalize=True).sort_index().plot(kind='bar', color='red')
plt.show()

sns.distplot(
    df_prepped["promotion_flag"][df_prepped["booking_bool"] == 1].dropna(),
    label="booked", kde=False, norm_hist=True, color="teal")
sns.distplot(
    df_prepped["promotion_flag"][df_prepped["booking_bool"] == 0].dropna(),
    label="not booked", kde=False, norm_hist=True, color="grey")
plt.legend()

plt.show()

mean_booked_with_flag = df_prepped[df_prepped["promotion_flag"] == 1]["booking_bool"].mean()
mean_booked_without_flag = df_prepped[df_prepped["promotion_flag"] == 0]["booking_bool"].mean()
mean_booked_with_brand = df_prepped[df_prepped["prop_brand_bool"] == 1]["booking_bool"].mean()
mean_booked_without_brand = df_prepped[df_prepped["prop_brand_bool"] == 0]["booking_bool"].mean()

x = [
    mean_booked_with_flag,
    mean_booked_without_flag,
    mean_booked_with_brand,
    mean_booked_without_brand
]


x_names = np.array(['promotion_flag = 1',
                    'promotion_flag = 0',
                    'prop_brand_bool = 1',
                    'prop_brand_bool = 0'
                    ]
                   )

perc = np.linspace(0,100,len(x))

plt.figure(figsize=(16, 5))
ax = sns.barplot(perc, x, alpha=0.8, palette=["lightblue", "teal", "wheat", "beige"])
ax.set_xticklabels(
    x_names,
    rotation=20,
    horizontalalignment='right'
)
ax.set_ylabel('probability')
ax.set_title('The Average Probability of Being Booked')
plt.show()

df["diff_starrating"] = np.abs(df["visitor_hist_starrating"] - df["prop_starrating"])
df["diff_price_usd"] = np.abs(df["visitor_hist_adr_usd"] - df["price_usd"])

def find_outlier(data):
    """
    Function finding outliers with the IQR method.
    Returns a list of all rows that contain 2 or more outliers

    """
    columns = [
        "diff_price_usd"
    ]
    outliers = []
    for col_name in columns:
        q1 = data[col_name].quantile(0.05)
        q3 = data[col_name].quantile(0.95)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        outliers.extend(
            data[(data[col_name] < lower_bound) | (data[col_name] > upper_bound)].index
        )
    rows = list(k for k, v in Counter(outliers).items() if v > 0)
    return rows

def remove_outlier(data, outliers):
    """
    Function removing the outliers found by the function find_outlier()

    """
    data = data.drop(outliers, axis=0).reset_index(drop=True)
    print(len(outliers), " rows have been deleted")
    return data


outliers = find_outlier(df)
df = remove_outlier(df, outliers)


plt.figure(figsize=(14, 4))

plt.subplot2grid((1, 2), (0, 0))
sns.distplot(
    df["diff_price_usd"][df["booking_bool"] == 1].dropna(),
    label="booked", kde=False, norm_hist=True, color="teal")
sns.distplot(
    df["diff_price_usd"][df["booking_bool"] == 0].dropna(),
    label="not booked", kde=False, norm_hist=True, color="grey")
plt.legend()
plt.subplot2grid((1, 2), (0, 1))
sns.distplot(
    df["diff_starrating"][df["booking_bool"] == 1].dropna(),
    label="booked", kde=False, norm_hist=True, color="teal", bins=15)
sns.distplot(
    df["diff_starrating"][df["booking_bool"] == 0].dropna(),
    label="not booked", kde=False, norm_hist=True, color="grey", bins=15)
plt.legend()

plt.show()

# Correlation matrices and corresponding heat maps
columns = ['site_id', 'visitor_location_country_id',
           'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id',
           'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool',
           'prop_location_score1', 'prop_location_score2',
           'prop_log_historical_price', 'price_usd', 'promotion_flag',
           'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window',
           'srch_adults_count', 'srch_children_count', 'srch_room_count',
           'srch_saturday_night_bool', 'srch_query_affinity_score',
           'orig_destination_distance', 'random_bool',
           ]

corr = df[columns].corr()
plt.figure(figsize=(19, 10))
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=30,
    horizontalalignment='right'
)
plt.show()
