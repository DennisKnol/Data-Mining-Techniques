import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import linear_model

from collections import Counter


# TODO: price_usd: drop outlier


def missing_value_count(data):
    """
    Function returning a column with a count of missing values
    and a column with the percentage of missing values

    """
    count = data.isnull().sum()
    percentage = (data.isnull().sum() / data.shape[0]) * 100
    missing_data = pd.concat([count, percentage], axis=1, keys=["Count", "Percentage"])
    missing_data.sort_values(["Count", "Percentage"], ascending=False, inplace=True)
    return missing_data


def fill_missing_values(data):
    """
    Function dealing  missing values in the data set

    """
    # fill missing prop_review_score values with 0
    data["prop_review_score"] = data["prop_review_score"].fillna(0)

    # new column, 0 is not reviewed, 1 if reviewed
    data["prop_review_score_bool"] = np.where(data["prop_review_score"] == 0, 0, 1)

    # transform and fill srch_query_affinity_score
    data["srch_query_affinity_score"] = np.exp(data["srch_query_affinity_score"]).fillna(0)
    # or drop?

    return data


def create_bools(data):
    data["sold_prev_period_bool"] = np.where(data["prop_log_historical_price"] == 0, 0, 1)
    data["prop_review_score_bool"] = np.where(data["prop_review_score"] == 0, 0, 1)
    data["prop_starrating_bool"] = np.where(data["prop_starrating"] == 0, 0, 1)
    return data


def fill_prop_location_score_2(data):
    """
    Function filling the missing values for the property location score 2. Missing values are predicted
    using a linear regression model with the following explanatory variables:

    'prop_review_score', 'prop_starrating', 'prop_brand_bool' & 'prop_location_score1'

    """
    # consider all the rows where 'prop_location_score2' is not null training data
    train_data = data.loc[pd.notnull(data["prop_location_score2"])]
    X_train = train_data[
        [
            "prop_review_score",
            "prop_starrating",
            "prop_brand_bool",
            "prop_location_score1",
        ]
    ].values
    y_train = train_data["prop_location_score1"].values

    # fit model
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)

    X = data.loc[pd.isnull(data["prop_location_score2"])][
        [
            "prop_review_score",
            "prop_starrating",
            "prop_brand_bool",
            "prop_location_score1",
        ]
    ].values

    # fill empty scores with predicted values
    data.loc[
        pd.isnull(data["prop_location_score2"]), "prop_location_score2"
    ] = model.predict(X)

    data.loc[data["prop_location_score2"] < 0, "prop_location_score2"] = 0
    return data


def prep_prop_log_historical_price(data):
    """

    A 0 will occur if the hotel was not sold in that period: create boolean, sold / not sold

    """
    data["sold_prev_period_bool"] = np.where(data["prop_log_historical_price"] == 0, 0, 1)

    mean_prop_price = data.groupby("prop_id")["prop_log_historical_price"].mean()
    data.loc[
        data["prop_log_historical_price"] == 0, "prop_log_historical_price"
    ] = data.loc[
        data["prop_log_historical_price"] == 0, ["prop_id", "prop_log_historical_price"]
    ].apply(
        lambda x: mean_prop_price[x["prop_id"]], axis=1
    )

    # data["prop_log_historical_price"] = np.exp(data["prop_log_historical_price"])
    return data


def fill_orig_destination_distance(data):
    mean_distance_01 = data.groupby("srch_id")["orig_destination_distance"].mean()
    data.loc[data["orig_destination_distance"].isna(), "orig_destination_distance"] = data.loc[data["orig_destination_distance"].isna(),
        ["orig_destination_distance", "srch_id"],
    ].apply(
        lambda x: mean_distance_01[x["srch_id"]], axis=1
    )

    mean_distance_02 = data.groupby("visitor_location_country_id")["orig_destination_distance"].mean()
    data.loc[data["orig_destination_distance"].isna(), "orig_destination_distance"] = data.loc[data["orig_destination_distance"].isna(),
        ["orig_destination_distance", "visitor_location_country_id"],
    ].apply(
        lambda x: mean_distance_02[x["visitor_location_country_id"]], axis=1
    )

    mean_distance_03 = data.groupby("srch_destination_id")["orig_destination_distance"].mean()
    data.loc[data["orig_destination_distance"].isna(), "orig_destination_distance"] = data.loc[
        data["orig_destination_distance"].isna(),
        ["orig_destination_distance", "srch_destination_id"],
    ].apply(
        lambda x: mean_distance_03[x["srch_destination_id"]], axis=1
    )

    data["orig_destination_distance"].fillna((data["orig_destination_distance"].mean()), inplace=True)

    # # initially, fill with average in the corresponding property country id and visitors location
    # mean_distance_1 = data.groupby(["visitor_location_country_id", "prop_country_id"])["orig_destination_distance"].mean().reset_index()
    #
    #
    # data.loc[data["orig_destination_distance"].isna(), "orig_destination_distance"] = data.loc[
    #     data["orig_destination_distance"].isna(),
    #     ["orig_destination_distance", "visitor_location_country_id", "prop_country_id"],
    # ].apply(
    #     lambda x: mean_distance_1[x[["visitor_location_country_id", "prop_country_id"]]],
    #     axis=1,
    # )
    #
    #

    # # second, fill with average in the corresponding property srch id and visitors location
    # mean_distance_2 = data.groupby(
    #     ["srch_destination_id", "visitor_location_country_id"]
    # )["orig_destination_distance"].mean().reset_index()
    # data.loc[
    #     data["orig_destination_distance"].isna(), "orig_destination_distance"
    # ] = data.loc[
    #     data["orig_destination_distance"].isna(),
    #     [
    #         "orig_destination_distance",
    #         "srch_destination_id",
    #         "visitor_location_country_id",
    #     ],
    # ].apply(
    #     lambda x: mean_distance_2[
    #         x[["srch_destination_id", "visitor_location_country_id"]]
    #     ],
    #     axis=1,
    # )

    # mean_distance_3 = data.groupby("visitor_location_country_id")["orig_destination_distance"].mean()
    # data.loc[
    #     data["orig_destination_distance"].isna(), "orig_destination_distance"
    # ] = data.loc[
    #     data["orig_destination_distance"].isna(),
    #     ["orig_destination_distance", "visitor_location_country_id"],
    # ].apply(
    #     lambda x: mean_distance_3[x["visitor_location_country_id"]], axis=1
    # )
    # 
    # print(mean_distance_3)
    # print(mean_distance_1)
    # print(mean_distance_1.shape)
    return data


def bin_price_data(data):
    """Create bins for the price data based on predefined bin size

    """
    bin_size = 20
    bins = list(range(0, math.ceil(max(data['price_usd'])), bin_size))
    print(bins)
    print(pd.cut(data, bins))
    return data



def convert_date_time(data):
    """
    Function splitting the 'date_time' column in 'date' and 'time'
    Subsequently splitting 'date' in 'year', 'month' and 'day'

    """
    data[["date", "time"]] = data["date_time"].str.split(" ", expand=True)
    data[["year", "month", "day"]] = data["date"].str.split("-", expand=True)
    return data


def competitor_count(data):
    """
    Function creating a column with the number of competitors
    Competitors that do not have availability will be removed from statistics

    """
    for i in range(1, 9):
        data.loc[data[f"comp{i}_inv"] != 0, f"comp{i}_rate"] = np.nan

    columns_rate = ["comp{}_rate".format(i) for i in range(1, 9)]
    data["competitor_count"] = data[columns_rate].count(axis=1)
    data["competitor_lower_percent"] = (data[columns_rate] < 0).sum(axis=1)
    data["competitor_fraction_lower"] = data.competitor_lower_percent.div(data.competitor_count)
    data.loc[~np.isfinite(data["competitor_fraction_lower"]), 'competitor_fraction_lower'] = 0

    return data


def create_bins(data):
    pass


def create_srch_columns(data):
    """
    Function creating a column with the total count of travelers and a column with the ratio guests per room

    """
    data["srch_travelers_count"] = data["srch_adults_count"] + data["srch_children_count"]
    data["guests_per_room"] = data["srch_travelers_count"] / data["srch_room_count"]
    return data


def find_outlier(data):
    """
    Function finding outliers with the IQR method.
    Returns a list of all rows that contain 2 or more outliers

    """
    columns = [
        # "visitor_hist_adr_usd",
        "price_usd",
        "srch_length_of_stay",
        "srch_booking_window",
    ]
    outliers = []
    for col_name in columns:
        q1 = data[col_name].quantile(0.05)
        q3 = data[col_name].quantile(0.95)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        outliers.extend(
            data[(data[col_name] < lower_bound) | (data[col_name] > upper_bound)].index
        )
    rows = list(k for k, v in Counter(outliers).items() if v > 0)
    return rows


def remove_outlier(data, outliers):
    """
    Function removing the outliers found by the function find_outlier()

    """
    data = data.drop(outliers, axis=0).reset_index(drop=True)
    print(len(outliers), " rows have been deleted")
    return data


def drop_data(data):
    """
    Function dropping columns

    """
    # drop gross_bookings_usd, too many missing values
    data = data.drop(columns=["gross_bookings_usd"], axis=1)

    # drop all comp data for competitors 1 - 8
    # data = data.drop([col for col in data.columns if 'comp' in col])
    return data
