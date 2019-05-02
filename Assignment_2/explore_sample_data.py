import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("df_sample.csv")
df = df.drop(df.columns[0], axis=1)

print(df.info())
print("Shape of the dataframe: ", df.shape, "\n")  # (4958347, 54)
print("Sum of is null values: \n", df.isnull().sum(), "\n")
print("Sum of NaN values:\n ", df.isna().sum())

df[["date", "time"]] = df["date_time"].str.split(" ", expand=True)
df[["year", "month", "day"]] = df["date"].str.split("-", expand=True)