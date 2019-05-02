import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


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
