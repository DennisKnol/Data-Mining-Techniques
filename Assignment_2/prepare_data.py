import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from functions import *
from prep_data import *

# read in data
df = pd.read_csv("training_set_VU_DM.csv")
df_test = pd.read_csv("test_set_VU_DM.csv")

# prep train
df = prep_data(df)

# prep test
df_test = prep_data(df_test)

# check missing values
print(missing_value_count(df))
print(missing_value_count(df_test))

df.to_csv("prepped_training_set_VU_DM.csv", index=False)
df_test.to_csv("prepped_test_set_VU_DM.csv", index=False)
