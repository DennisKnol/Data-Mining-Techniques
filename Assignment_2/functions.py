import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter


def missing_value_count(data):
    """
    Function returning a column with a count of missing values
    and a column with the percentage of missing values

    """
    count = data.isnull().sum()
    percentage = (data.isnull().sum()/data.shape[0]) * 100
    missing_data = pd.concat([count, percentage], axis=1, keys=["Count", "Percentage"])
    missing_data.sort_values(["Count", "Percentage"], ascending=False, inplace=True)
    return missing_data


def missing_values(data):
    """
    Function dealing  missing values in the data set

    """

    # fill missing prop_review_score values with 0
    data["prop_review_score"] = data["prop_review_score"].fillna(0, inplace=True)

    # fill property location score with mean of corresponding search id
    data["prop_location_score2"] = data.groupby("search_id")

    # drop gross_bookings_usd, too many missing values
    data = data.drop(data.columns["gross_bookings_usd"], axis=1)
    return data


def convert_date_time(data):
    """
    Function splitting the 'date_time' column in 'date' and 'time'
    Subsequently splitting 'date' in 'year', 'month' and 'day'

    """
    data[["date", "time"]] = data["date_time"].str.split(" ", expand=True)
    data[["year", "month", "day"]] = data["date"].str.split("-", expand=True)
    return data


def combine_competitors(data):
    """
    WIP
    :param data:
    :return:
    """
    data["comp_rate"] = [data["comp{}_rate".format(i)] for i in range(1, 9)]
    data["comp_inv"] = [data["comp{}_inv".format(i)] for i in range(1, 9)]
    data["comp_rate_percent_diff"] = [data["comp{}_rate_percent_diff".format(i)] for i in range(1, 9)]
    # sum(relative_difference)/len(relative_difference)
    return data


def competitor_count(data):
    """
    Function creating a column with the number of competitors

    """
    data["competitor_count"] = data[["comp{}_rate".format(i) for i in range(1, 9)]].abs.sum(axis=1)
    return data


def find_outlier(data):
    """
    Function finding outliers with the IQR method.
    Returns a list of all rows that contain 2 or more outliers

    """
    columns = ["visitor_hist_adr_usd", "price_usd", "srch_length_of_stay", "srch_booking_window"]
    outliers = []
    for col_name in columns:
        q1 = data[col_name].quantile(0.1)
        q3 = data[col_name].quantile(0.9)
        iqr = q3-q1
        lower_bound = q1 - (1.5*iqr)
        upper_bound = q3 + (1.5*iqr)
        outliers.extend(data[(data[col_name] < lower_bound) | (data[col_name] > upper_bound)].index)
    rows = list(k for k, v in Counter(outliers).items() if v > 0)
    return rows


def remove_outlier(data, outliers):
    """
    Function removing the outliers found by the function find_outlier()

    """
    data = data.drop(outliers, axis=0).reset_index(drop=True)
    print(len(outliers), " rows have been deleted")
    return data
