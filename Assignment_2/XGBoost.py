import xgboost as xgb
import pandas as pd


df = pd.read_csv("training_set_VU_DM.csv")
df_test = pd.read_csv("test_set_VU_DM.csv")

xgb_rank = xgb.XGBRanker()


def group_size():
    """ hoeveel searches per srch_id"""
    return


X = df[[]]
y = df["booking_bool"]

X_test = df_test[[]]

xgb_rank.fit(X, y, group_size())

predictions = xgb_rank.predict()
