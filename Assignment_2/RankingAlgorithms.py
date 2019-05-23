import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def rank_rf(X, y, X_test):
    """ Makes RF classifier ranking"""

    rf_predictor = RandomForestClassifier()
    rf_predictor.fit(X, y)
    rf_preds = np.sum(rf_predictor.predict_proba(X_test)[:, 1:], axis=1)

    rf_dataset = pd.DataFrame(rf_preds, columns=['rf_preds'])
    rf_merged = pd.concat([df_test[['srch_id', 'prop_id']], rf_dataset], axis=1, sort=False)
    rf_ranking = rf_merged.join(
        rf_merged.groupby('srch_id')['rf_preds'].rank(ascending=False).astype(int).rename('rank'))

    return rf_ranking