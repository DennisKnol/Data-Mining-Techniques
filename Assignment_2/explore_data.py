import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("training_set_VU_DM.csv")

print(df.info())
print("Sum of is null values: \n", df.isnull().sum(), "\n")
print("Sum of NaN values:\n ", df.isna().sum())

df[["date", "time"]] = df["date_time"].str.split(" ", expand=True)
df[["year", "month", "day"]] = df["date"].str.split("-", expand=True)
# df["date"] = pd.to_date(df['date'], format='%d%b%Y:%H:%M:%S.%f')
# df = df.drop(["date"], axis=1)
