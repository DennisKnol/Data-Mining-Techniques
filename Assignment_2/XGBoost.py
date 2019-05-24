import xgboost as xgb
import matplotlib.pyplot as plt

from prep_data_function import *
from RankingAlgorithms import *

df = pd.read_csv("prepped_training_set_VU_DM.csv")
df_test = pd.read_csv("prepped_test_set_VU_DM.csv")
df_test_whole = df_test

df = df[:1000000]
# df_test = df_test[:1000000]
#
# normalize_features = ["price_usd", "prop_location_score1", "prop_location_score2", "prop_review_score"]
# wrt_features = ["srch_id", "prop_id", "srch_booking_window", "srch_destination_id"]
#
# for feature in normalize_features:
#     for wrt_feature in wrt_features:
#         df = normalize_feature(df, feature=feature, wrt_feature=wrt_feature)
#         df_test = normalize_feature(df_test, feature=feature, wrt_feature=wrt_feature)
#         df_test_whole = normalize_feature(df_test_whole, feature=feature, wrt_feature=wrt_feature)
#

def make_submission(df_test, predictions):
    df = pd.DataFrame(predictions, columns=['preds'])
    merge = pd.concat([df_test, df], axis=1, sort=False)
    sorted_df = merge.sort_values(['srch_id', 'preds'], ascending=[True, False])
    submission = sorted_df[['srch_id', 'prop_id']]
    submission.to_csv('submission.csv', index=False)
    return print("submission file done")


def group_size(data):
    srch_value_counts = data["srch_id"].value_counts()
    srch_count = pd.DataFrame([srch_value_counts]).T.sort_index()
    return srch_count["srch_id"]

# dr uit: visitor_hist_adr_usd, date time alles, srch id, site id, abs_diff_price_usd,
# diff_price_usd_perc, srch_person_count, score1ma, prop_country_id, prop_id, visitor_location_country_id
# price_usd, srch_length_of_stay, srch_booking_window
# 'srch_adults_count', 'srch_children_count', 'srch_room_count',

columns_for_X = [
    'visitor_hist_starrating',
    'prop_starrating',
    'prop_review_score',
    'prop_brand_bool',
    'prop_location_score1',
    'prop_location_score2',
    'prop_log_historical_price',
    'promotion_flag',
    'srch_destination_id',
    'srch_booking_window',
    'srch_saturday_night_bool',
    'srch_query_affinity_score',
    'orig_destination_distance',
    'random_bool',
    'diff_starrating',
    'abs_diff_price_usd_perc',
    'diff_price_usd',
    'prop_starrating_bool',
    'mean_prop_starrating',
    'prop_review_score_bool',
    'mean_prop_review_score',
    'mean_prop_location_score1',
    'mean_prop_location_score2',
    'prop_location_score_combined',
    'mean_prop_location_score_combined',
    'sold_prev_period_bool',
    'prop_historical_price',
    'prop_price_diff',
    'total_price',
    'mean_prop_price',
    'fee_per_person',
    'score2ma',
    'mean_destination_distance',
    'norm_price_usd_wrt_srch_id',
    'norm_price_usd_wrt_prop_id',
    'norm_price_usd_wrt_srch_booking_window',
    'norm_price_usd_wrt_srch_destination_id',
    'norm_price_usd_wrt_site_id',
    'norm_prop_location_score1_wrt_srch_id',
    'norm_prop_location_score1_wrt_srch_booking_window',
    'norm_prop_location_score1_wrt_srch_destination_id',
    'norm_prop_location_score1_wrt_site_id',
    'norm_prop_location_score2_wrt_srch_id',
    'norm_prop_location_score2_wrt_prop_id',
    'norm_prop_location_score2_wrt_srch_booking_window',
    'norm_prop_location_score2_wrt_srch_destination_id',
    'norm_prop_location_score2_wrt_site_id',
    'norm_prop_review_score_wrt_srch_id',
    'norm_prop_review_score_wrt_srch_booking_window',
    'norm_prop_review_score_wrt_srch_destination_id',
    'norm_prop_review_score_wrt_site_id'
]

X = df[columns_for_X]
X_test = df_test_whole[columns_for_X]

y = df["booking_bool"]

# y_forest = df["booking_bool"] + df["click_bool"]
#


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
#
# xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
#                           max_depth=5, alpha=10, n_estimators=10)
#
# xg_reg.fit(X_train, y_train)
#
# preds = xg_reg.predict(X_test)

# ranking_split = [0.8, 0.2]
# xgb_ranking = rank_XGBoost(X, y, X_test, group_size(df), df_test)
# rf_ranking = rank_rf(X, y, X_test, df_test)
#
#
# combined_ranking = rf_ranking.join(xgb_ranking[['xgb_rank', 'xgb_preds']])
# combined_ranking['comb_rank'] = combined_ranking[['xgb_rank', 'rf_rank']].dot(ranking_split)

xgb_rank = xgb.XGBRanker(objective='rank:ndcg')
xgb_rank.fit(X, y, group_size(df))


preds = xgb_rank.predict(X_test)

make_submission(df_test_whole, preds)

fig, ax = plt.subplots(figsize=(14, 7))
xgb.plot_importance(xgb_rank, ax=ax)
plt.show()
