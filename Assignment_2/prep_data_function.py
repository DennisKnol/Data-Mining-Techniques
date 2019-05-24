import pandas as pd
import numpy as np
import math
from collections import Counter


def find_outlier(data):
    """
    Function finding outliers with the IQR method.
    Returns a list of all rows that contain 2 or more outliers

    """
    columns = [
        "visitor_hist_adr_usd",
        "price_usd",
        # "srch_length_of_stay",
        # "srch_booking_window",
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


def normalize_feature(data, feature, wrt_feature):
    """
    Function normalizing features

    """
    data["norm_"+feature+"_wrt_"+wrt_feature] = data.groupby(wrt_feature)[feature].transform(
        lambda x: (x - x.mean()) / x.std()
    )

    data["norm_" + feature + "_wrt_" + wrt_feature] = data["norm_"+feature+"_wrt_"+wrt_feature].fillna(
        data["norm_" + feature + "_wrt_" + wrt_feature].mean()
    )

    return data


def prep_data(data):
    """
    Function preparing the data

    """
    outliers = find_outlier(data)
    data = remove_outlier(data, outliers)
    print("outlier")
    data[["date", "time"]] = data["date_time"].str.split(" ", expand=True)
    data[["year", "month", "day"]] = data["date"].str.split("-", expand=True)

    for feature in ["visitor_hist_starrating", "visitor_hist_adr_usd"]:
        first_quartile_hist = data.groupby("prop_country_id")[feature].quantile(0.25)
        first_quartile_fill = data.loc[data[feature].isna(), [feature, "prop_country_id"]
        ].apply(
            lambda x: first_quartile_hist[x["prop_country_id"]], axis=1
        )
        data.loc[data[feature].isna(), feature] = first_quartile_fill

    data["visitor_hist_starrating"] = data["visitor_hist_starrating"].fillna(0)
    data["visitor_hist_adr_usd"] = data["visitor_hist_adr_usd"].fillna(0)
    print("history")
    data["diff_starrating"] = np.abs(data["visitor_hist_starrating"] - data["prop_starrating"])
    data["abs_diff_price_usd_perc"] = np.abs(data["visitor_hist_adr_usd"] / data["price_usd"])

    data["diff_price_usd"] = data["visitor_hist_adr_usd"] - data["price_usd"]

    data["prop_starrating_bool"] = np.where(data["prop_starrating"] == 0, 0, 1)

    mean_prop_starrating = data.groupby("prop_id")["prop_starrating"].mean()
    data["mean_prop_starrating"] = data["prop_id"].apply(lambda x: mean_prop_starrating[x])

    data["prop_review_score"] = data["prop_review_score"].fillna(0)
    data["prop_review_score_bool"] = np.where(data["prop_review_score"] == 0, 0, 1)

    mean_prop_review_score = data.groupby("prop_id")["prop_review_score"].mean()
    data["mean_prop_review_score"] = data["prop_id"].apply(lambda x: mean_prop_review_score[x])

    # prop_location_score1
    mean_prop_location_score1 = data.groupby("prop_id")["prop_location_score1"].mean()
    data["mean_prop_location_score1"] = data["prop_id"].apply(lambda x: mean_prop_location_score1[x])
    print("prop_loc1")
    # prop_location_score2
    first_quartile_loc_score2_per_prop_country = data.groupby("prop_country_id")["prop_location_score2"].quantile(0.25)
    first_quartile_fill = data.loc[data["prop_location_score2"].isna(), ["prop_location_score2", "prop_country_id"]
    ].apply(
        lambda x: first_quartile_loc_score2_per_prop_country[x["prop_country_id"]], axis=1
    )
    data.loc[data["prop_location_score2"].isna(), "prop_location_score2"] = first_quartile_fill

    data["prop_location_score2"] = data["prop_location_score2"].fillna(data["prop_location_score2"].mean())
    print("prop_loc2")

    mean_prop_location_score2 = data.groupby("prop_id")["prop_location_score2"].mean()
    data["mean_prop_location_score2"] = data["prop_id"].apply(lambda x: mean_prop_location_score2[x])

    data["prop_location_score_combined"] = (
            (data["prop_location_score2"] + 0.0001) / (data["prop_location_score1"] + 0.0001)
    )

    mean_prop_location_score_comb = data.groupby("prop_id")["prop_location_score_combined"].mean()
    data["mean_prop_location_score_combined"] = data["prop_id"].apply(lambda x: mean_prop_location_score_comb[x])

    # create boolean: 0 will occur if the hotel was not sold in that period
    data["sold_prev_period_bool"] = np.where(data["prop_log_historical_price"] == 0, 0, 1)

    data["prop_historical_price"] = np.exp(data["prop_log_historical_price"])

    data["prop_price_diff"] = data["prop_historical_price"] - data["price_usd"]

    data["total_price"] = data["price_usd"] * data["srch_room_count"]

    mean_prop_price = data.groupby("prop_id")["price_usd"].mean()
    data["mean_prop_price"] = data["prop_id"].apply(lambda x: mean_prop_price[x])

    # creating a column with the total person count
    data["srch_person_count"] = data["srch_adults_count"] + data["srch_children_count"]

    data["fee_per_person"] = data["total_price"] / data["srch_person_count"]

    # creating a column with the ratio guests per room
    data["guests_per_room"] = data["srch_person_count"] / data["srch_room_count"]

    # fill "srch_query_affinity_score"
    data["srch_query_affinity_score"] = np.exp(data["srch_query_affinity_score"]).fillna(0)

    data["score1ma"] = data["srch_query_affinity_score"] * data["prop_location_score1"]
    data["score2ma"] = data["srch_query_affinity_score"] * data["prop_location_score2"]
    print("scorema")

    # cut price_usd into bins
    bin_size = 20
    bins = list(range(0, math.ceil(max(data["price_usd"])), bin_size))
    pd.cut(data["price_usd"], bins)

    # fill orig_destination_distance
    data['orig_destination_distance'] = data['orig_destination_distance'].fillna(data.orig_destination_distance.median())

    mean_destination_distance = data.groupby("srch_id")["orig_destination_distance"].mean()
    data["mean_destination_distance"] = data["srch_id"].apply(lambda x: mean_destination_distance[x])

    # create a column with the number of competitors
    # Competitors that do not have availability will be removed from statistics
    # https://www.kaggle.com/c/expedia-personalized-sort/discussion/5774

    # for i in range(1, 9):
    #     data.loc[data[f"comp{i}_inv"] != 0, f"comp{i}_rate"] = np.nan
    #
    # columns_rate = ["comp{}_rate".format(i) for i in range(1, 9)]
    # data["competitor_count"] = data[columns_rate].count(axis=1)
    # data["competitor_lower_percent"] = (data[columns_rate] < 0).sum(axis=1)
    # data["competitor_fraction_lower"] = data.competitor_lower_percent.div(data.competitor_count)
    # data.loc[~np.isfinite(data["competitor_fraction_lower"]), "competitor_fraction_lower"] = 0

    print("normalize")
    # normalize data
    normalize_features = ["price_usd", "prop_location_score1", "prop_location_score2", "prop_review_score"]
    wrt_features = ["srch_id", "prop_id", "srch_booking_window", "srch_destination_id", "site_id"]

    for feature in normalize_features:
        for wrt_feature in wrt_features:
            data = normalize_feature(data, feature=feature, wrt_feature=wrt_feature)

    # drop data
    print("drop")
    columns_to_drop_list = ["date_time"]

    if "gross_bookings_usd" in data.columns:
        columns_to_drop_list.append("gross_bookings_usd")

    for i in range(1, 9):
        columns_to_drop_list.append("comp{}_rate".format(i))
        columns_to_drop_list.append("comp{}_inv".format(i))
        columns_to_drop_list.append("comp{}_rate_percent_diff".format(i))

    data = data.drop(columns=columns_to_drop_list)

    return data


