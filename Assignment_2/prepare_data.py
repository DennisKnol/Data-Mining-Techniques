import pandas as pd
import numpy as np
import math

import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter

from functions import *
from prep_data_function import *

# read in data
df = pd.read_csv("training_set_VU_DM.csv")
print("checkpoint 1")
df_test = pd.read_csv("test_set_VU_DM.csv")
print("checkpoint 2")

# prep data with custom function
df = prep_data(df)                  # 21898  rows have been deleted
print("checkpoint 3")
df_test = prep_data(df_test)        # 22026  rows have been deleted
print("checkpoint 4")

normalize_features = ["price_usd", "prop_location_score1", "prop_location_score2", "prop_review_score"]
wrt_features = ["srch_id", "prop_id", "srch_booking_window", "srch_destination_id", "site_id"]

for feature in normalize_features:
    for wrt_feature in wrt_features:
        df = normalize_feature(df, feature=feature, wrt_feature=wrt_feature)
        df_test = normalize_feature(df_test, feature=feature, wrt_feature=wrt_feature)

print("checkpoint 5")

# check missing values
print(missing_value_count(df))
print(missing_value_count(df_test))

df.to_csv("prepped_training_set_VU_DM.csv", index=False)
print("checkpoint 6")
df_test.to_csv("prepped_test_set_VU_DM.csv", index=False)
print("checkpoint 7")


def new_prep(data):
    mean_destination_distance = data.groupby("srch_id")["orig_destination_distance"].mean()
    data["mean_destination_distance"] = data["srch_id"].apply(lambda x: mean_destination_distance[x])

    data = data.drop(columns="norm_prop_review_score_wrt_prop_id")
    data = data.drop(columns="norm_prop_location_score1_wrt_prop_id")
    return data
