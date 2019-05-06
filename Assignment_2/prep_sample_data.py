import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from functions import missing_values, convert_date_time, combine_competitors


df = pd.read_csv("df_sample.csv")
df = df.drop(df.columns[0], axis=1)

comp_rate = [df["comp{}_rate".format(i)] for i in range(1, 9)]

