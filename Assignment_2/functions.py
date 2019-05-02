import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


def missing_values(data):
    """
    function returning a column with a count of missing values
    and a column with the percentage of missing values
    """
    count = data.isnull().sum()
    percentage = (data.isnull().sum()/df.shape[0])
    missing_data = pd.concat([count, percentage], axis=1, keys=["Count", "Percentage"])
    missing_data.sort_values(["Count", "Percentage"], ascending=False, inplace=True)
    return missing_data

