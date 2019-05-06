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
print("Number of unique search IDs:\n ", df["search_id"].unique())
