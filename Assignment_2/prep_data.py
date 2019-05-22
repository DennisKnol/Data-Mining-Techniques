import pandas as pd
import numpy as np
import math
from collections import Counter


def prep_data(data):
    """
    Function preparing the data

    Every change to the data is mentioned before execution

    """

    # split "date_time" column into "date" and "time"
    data[["date", "time"]] = data["date_time"].str.split(" ", expand=True)

    # split "date" into "year", "month" and "day"
    data[["year", "month", "day"]] = data["date"].str.split("-", expand=True)

    # TODO: visitor_hist_starrating & visitor_hist_adr_usd

    # create boolean: 0 indicates the property has no stars, 1 indicates it has stars
    data["prop_starrating_bool"] = np.where(data["prop_starrating"] == 0, 0, 1)

    # fill missing prop_review_score values with 0
    data["prop_review_score"] = data["prop_review_score"].fillna(0)

    # create boolean 0 if not reviewed, 1 if reviewed
    data["prop_review_score_bool"] = np.where(data["prop_review_score"] == 0, 0, 1)

    # fill empty prop_location_score2 with mean per search_id
    mean_location_score = data.groupby("srch_id")["prop_location_score2"].mean()
    mean_location_score.loc[mean_location_score.isna()] = 0.5

    data.loc[data["prop_location_score2"].isna(), "prop_location_score2"] = data.loc[
        data["prop_location_score2"].isna(),
        ["prop_location_score2", "srch_id"]
    ].apply(
        lambda x: mean_location_score[x["srch_id"]], axis=1
    )

    # create boolean: 0 will occur if the hotel was not sold in that period
    data["sold_prev_period_bool"] = np.where(data["prop_log_historical_price"] == 0, 0, 1)

    # fill empty prop_log_historical_price with mean prop price per prop id
    mean_prop_price = data.groupby("prop_id")["prop_log_historical_price"].mean()

    data.loc[data["prop_log_historical_price"] == 0, "prop_log_historical_price"] = data.loc[
        data["prop_log_historical_price"] == 0,
        ["prop_id", "prop_log_historical_price"]
    ].apply(
        lambda x: mean_prop_price[x["prop_id"]], axis=1
    )

    # cut price_usd into bins
    bin_size = 20
    bins = list(range(0, math.ceil(max(data["price_usd"])), bin_size))
    pd.cut(data["price_usd"], bins)

    # creating a column with the total count of travelers
    data["srch_travelers_count"] = data["srch_adults_count"] + data["srch_children_count"]

    # creating a column with the ratio guests per room
    data["guests_per_room"] = data["srch_travelers_count"] / data["srch_room_count"]

    data["srch_query_affinity_score"] = np.exp(data["srch_query_affinity_score"]).fillna(0)

    # fill orig_destination_distance
    # TODO: improve
    mean_distance_01 = data.groupby("srch_id")["orig_destination_distance"].mean()
    data.loc[data["orig_destination_distance"].isna(), "orig_destination_distance"] = data.loc[
        data["orig_destination_distance"].isna(),
        ["orig_destination_distance", "srch_id"],
    ].apply(
        lambda x: mean_distance_01[x["srch_id"]], axis=1
    )

    mean_distance_02 = data.groupby("visitor_location_country_id")["orig_destination_distance"].mean()
    data.loc[data["orig_destination_distance"].isna(), "orig_destination_distance"] = data.loc[
        data["orig_destination_distance"].isna(),
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

    # create a column with the number of competitors
    # Competitors that do not have availability will be removed from statistics

    for i in range(1, 9):
        data.loc[data[f"comp{i}_inv"] != 0, f"comp{i}_rate"] = np.nan

    columns_rate = ["comp{}_rate".format(i) for i in range(1, 9)]
    data["competitor_count"] = data[columns_rate].count(axis=1)
    data["competitor_lower_percent"] = (data[columns_rate] < 0).sum(axis=1)
    data["competitor_fraction_lower"] = data.competitor_lower_percent.div(data.competitor_count)
    data.loc[~np.isfinite(data["competitor_fraction_lower"]), "competitor_fraction_lower"] = 0


    # drop data
    columns_to_drop_list = []

    if "gross_bookings_usd" in data.columns:
        columns_to_drop_list.append("gross_bookings_usd")

    for i in range(1, 9):
        columns_to_drop_list.append("comp{}_rate".format(i))
        columns_to_drop_list.append("comp{}_inv".format(i))
        columns_to_drop_list.append("comp{}_rate_percent_diff".format(i))

    data = data.drop(columns=columns_to_drop_list)

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

    outliers = find_outlier(data)
    data = remove_outlier(data, outliers)

    return data
