import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from functions import *

# read in data
df = pd.read_csv("training_set_VU_DM.csv")

# detect and remove outliers
outlier_rows = find_outlier(df)
number_of_outlier_row = len(outlier_rows)
df = remove_outlier(df, outlier_rows)

# handle missing values
df = fill_missing_values(df)
df = fill_prop_location_score_2(df)

df = prep_prop_log_historical_price(df)
df = fill_orig_destination_distance(df)

df = drop_data(df)

df = create_srch_columns(df)

df = bin_price_data(df)

print(len(df["visitor_location_country_id"].unique()))

print(missing_value_count(df))
