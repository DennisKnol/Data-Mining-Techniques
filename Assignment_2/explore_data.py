import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("training_set_VU_DM.csv")

sample = df.sample(1000)
sample.to_csv("df_sample.csv")

print(df.info())
print("Shape of the dataframe: ", df.shape, "\n")  # (4958347, 54)
print("Sum of is null values: \n", df.isnull().sum(), "\n")
print("Sum of NaN values:\n ", df.isna().sum())

# df[["date", "time"]] = df["date_time"].str.split(" ", expand=True)
# df[["year", "month", "day"]] = df["date"].str.split("-", expand=True)
# df["date"] = pd.to_date(df['date'], format='%d%b%Y:%H:%M:%S.%f')
# df = df.drop(["date"], axis=1)

# seaborn correlation heatmap
