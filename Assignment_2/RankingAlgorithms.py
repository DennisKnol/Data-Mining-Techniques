import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


def group_size(data):
    srch_value_counts = data["srch_id"].value_counts()
    srch_count = pd.DataFrame([srch_value_counts]).T.sort_index()
    return srch_count["srch_id"]


def rank_rf(X, y, X_test, df_test):
    """ Makes RF classifier ranking"""

    rf_predictor = RandomForestClassifier(n_estimators=1000)
    rf_predictor.fit(X, y)
    rf_preds = np.sum(rf_predictor.predict_proba(X_test)[:, 1:], axis=1)

    rf_dataset = pd.DataFrame(rf_preds, columns=['rf_preds'])
    rf_merged = pd.concat([df_test[['srch_id', 'prop_id']], rf_dataset], axis=1, sort=False)
    rf_ranking = rf_merged.join(
        rf_merged.groupby('srch_id')['rf_preds'].rank(ascending=False).astype(int).rename('rf_rank'))

    return rf_ranking


def rank_XGBoost(X, y, X_test, group_size, df_test):
    """Makes XGBoost ranking"""

    xgb_rank = xgb.XGBRanker(objective='rank:ndcg')
    xgb_rank.fit(X, y, group_size)
    preds = xgb_rank.predict(X_test)

    # xgb.plot_importance(xgb_rank)
    # plt.show()

    # make submission
    xgb_dataset = pd.DataFrame(preds, columns=['xgb_preds'])

    xgb_merged = pd.concat([df_test[['srch_id', 'prop_id']], xgb_dataset], axis=1, sort=False)
    xgb_ranking = xgb_merged.join(
        xgb_merged.groupby('srch_id')['xgb_preds'].rank(ascending=False).astype(int).rename('xgb_rank'))

    return xgb_ranking