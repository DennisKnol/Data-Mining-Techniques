import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import linear_model

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


def fill_missing_values(data):
    """
    Function dealing  missing values in the data set

    """
    # fill missing prop_review_score values with 0
    data["prop_review_score"] = data["prop_review_score"].fillna(0)

    # transform and fill srch_query_affinity_score
    data['srch_query_affinity_score'] = np.exp(data['srch_query_affinity_score']).fillna(0)

    return data


def fill_prop_location_score_2(data):
    """
    Function filling the missing values for the property location score 2. Missing values are predicted
    using a linear regression model with the following explanatory variables:

    'prop_review_score', 'prop_starrating', 'prop_brand_bool' & 'prop_location_score1'

    """
    # consider all the rows where 'prop_location_score2' is not null training data
    train_data = data.loc[pd.notnull(data["prop_location_score2"])]
    X_train = train_data[["prop_review_score", "prop_starrating", "prop_brand_bool", "prop_location_score1"]].values
    y_train = train_data["prop_location_score1"].values

    # fit model
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)

    X = data.loc[pd.isnull(data["prop_location_score2"])][
        ["prop_review_score", "prop_starrating", "prop_brand_bool", "prop_location_score1"]
    ].values

    # fill empty scores with predicted values
    data.loc[pd.isnull(data["prop_location_score2"]), "prop_location_score2"] = model.predict(X)
    return data


def fill_orig_destination_distance(data):

    # initially, fill with average with mean age in the corresponding property country id and visitors location
    mean_distance_1 = data.groupby(["prop_country_id", "visitor_location_country_id"])["orig_destination_distance"].mean()
    mean_distance_1 = mean_distance_1.dropna()
    print(mean_distance_1)

    data["orig_destination_distance"] = data[["orig_destination_distance", "prop_country_id", "visitor_location_country_id"]].apply(
        lambda x: mean_distance_1[x[["prop_country_id", "visitor_location_country_id"]]] if pd.isnull(x["orig_destination_distance"]) else x[
            "orig_destination_distance"], axis=1
    )


    mean_distance_2 = data.groupby(["srch_destination_id", "visitor_location_country_id"])["orig_destination_distance"].mean()
    mean_distance_2 = mean_distance_2.dropna()
    print(mean_distance_2)

    mean_distance_3 = data.groupby("visitor_location_country_id")["orig_destination_distance"].mean()
    mean_distance_3 = mean_distance_3.dropna()
    print(mean_distance_3)

    data["orig_destination_distance"] = data[["orig_destination_distance", "visitor_location_country_id"]].apply(
        lambda x: mean_distance_3[x["visitor_location_country_id"]] if pd.isnull(x["orig_destination_distance"]) else x["orig_destination_distance"], axis=1
    )

    # mean_age_per_title = data.groupby("Title")["Age"].mean()
    # data["Age"] = data[["Age", "Title"]].apply(
    #     lambda x: mean_age_per_title[x["Title"]] if pd.isnull(x["Age"]) else x["Age"], axis=1
    # )

    return data


def drop_missing_values(data):
    """
    Function dropping columns

    """
    # drop gross_bookings_usd, too many missing values
    data = data.drop(columns=["gross_bookings_usd"], axis=1)
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
