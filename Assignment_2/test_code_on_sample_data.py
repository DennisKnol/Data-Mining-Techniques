import pandas as pd
import numpy as np
import math

import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter

from functions import *
from prep_data_function import *


data = pd.read_csv("df_sample.csv")
data = prep_data(data)

print(missing_value_count(data))

normalize_features = ["price_usd", "prop_location_score1", "prop_location_score2", "prop_review_score"]
wrt_features = ["srch_id", "prop_id", "srch_booking_window", "srch_destination_id"]

for feature in normalize_features:
    for wrt_feature in wrt_features:
        normalize_feature(data, feature=feature, wrt_feature=wrt_feature)


# # Density plots
# attributes = [
#     "prop_location_score1",
#     "prop_location_score2",
#     "log_prop_location_score2"
# ]
#
# for i in attributes:
#     sns.distplot(data[i][data["booking_bool"] == 1].dropna(), label="booked", kde=False, norm_hist=True)
#     sns.distplot(data[i][data["booking_bool"] == 0].dropna(), label="not booked", kde=False, norm_hist=True)
#     plt.legend()
#     plt.show()
