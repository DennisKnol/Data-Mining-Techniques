import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter


def missing_values(data):
    """
    Function returning a column with a count of missing values
    and a column with the percentage of missing values

    """
    count = data.isnull().sum()
    percentage = (data.isnull().sum()/data.shape[0])
    missing_data = pd.concat([count, percentage], axis=1, keys=["Count", "Percentage"])
    missing_data.sort_values(["Count", "Percentage"], ascending=False, inplace=True)
    return missing_data


def convert_date_time(data):
    """
    Function splitting the 'date_time' column in 'date' and 'time'
    Subsequently splitting 'date' in 'year', 'month' and 'day'

    """
    data[["date", "time"]] = data["date_time"].str.split(" ", expand=True)
    data[["year", "month", "day"]] = data["date"].str.split("-", expand=True)
    return data


def combine_competitors(data):
    data["comp_rate"] = [data["comp{}_rate".format(i)] for i in range(1, 9)]
    data["comp_inv"] = [data["comp{}_inv".format(i)] for i in range(1, 9)]
    data["comp_rate_percent_diff"] = [data["comp{}_rate_percent_diff".format(i)] for i in range(1, 9)]
    return data


def find_outlier(data):
    """
    Function finding outliers with the IQR method.

    Returns a list of all rows that contain 3 or more outliers

    """

    outliers = []
    for col_name in ["Age", "SibSp", "Parch", "Fare"]:
        q1 = data[col_name].quantile(0.25)
        q3 = data[col_name].quantile(0.75)
        iqr = q3-q1
        lower_bound = q1 - (1.5*iqr)
        upper_bound = q3 + (1.5*iqr)
        outliers.extend(data[(data[col_name] < lower_bound) | (data[col_name] > upper_bound)].index)
    rows = list(k for k, v in Counter(outliers).items() if v > 2)
    return rows


def remove_outlier(data, outliers):
    """
    Function removing the outliers found by the function find_outlier()

    """
    data = data.drop(outliers, axis=0).reset_index(drop=True)
    print(len(outliers), " rows have been deleted")
    return data
