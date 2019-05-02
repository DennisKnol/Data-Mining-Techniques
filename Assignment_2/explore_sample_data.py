import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from functions import missing_values, convert_date_time, combine_competitors

df = pd.read_csv("df_sample.csv")
df = df.drop(df.columns[0], axis=1)

print(df.info())
print(df.columns)
print("Shape of the dataframe: ", df.shape, "\n")  # (1000, 54)

print("Count and percentage of is null values: \n", missing_values(df))


df = convert_date_time(df)
sns.countplot('month', data=df)
plt.show()

df["position"][df["booking_bool"] == 1].value_counts(normalize=True).sort_index().plot(kind='bar')
plt.show()

df["position"][df["click_bool"] == 1].value_counts(normalize=True).sort_index().plot(kind='bar')
plt.show()



# train_data.Sex.value_counts().sort_index().plot(
#     kind='pie',
#     colors=["pink", "lightblue"],
#     autopct="%.1f%%",
#     title="Gender"
# )
#

# df = combine_competitors(df)

# corr = df.corr()
# plt.figure(figsize=(15, 10))
# ax = sns.heatmap(
#     corr,
#     vmin=-1, vmax=1, center=0,
#     cmap=sns.diverging_palette(20, 220, n=200),
#     square=True
# )
# ax.set_xticklabels(
#     ax.get_xticklabels(),
#     rotation=45,
#     horizontalalignment='right'
# );
# plt.show()

# Count plots
# sns.countplot('visitor_location_country_id', data=df)
# plt.show()
#
# sns.countplot('prop_country_id', data=df)
# plt.show()

# sns.countplot('prop_starrating', data=df)
# plt.show()
#
# sns.countplot('prop_brand_bool', data=df)
# plt.show()


# Density plots
# sns.distplot(df['price_usd'])
# plt.show()

# sns.distplot(df['orig_destination_distance'].notnull())
# plt.show()


