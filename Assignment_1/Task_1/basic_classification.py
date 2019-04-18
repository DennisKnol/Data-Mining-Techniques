import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


odi = pd.read_csv(
    "/Users/DennisK/PycharmProjects/Data-Mining-Techniques/Assignment_1/Task_1/ODI_2019_clean.csv",
    sep=','
)

odi = odi.drop(odi.columns[0], axis=1)

corr_mat = odi.corr().round(4)

categories = ["Sex", "Food", "Nice weather", "Alcohol", "Friends", "Sports", "Sleep"]

df = odi[['gender', 'good_day_1', 'good_day_2']].copy()
df = df[df["good_day_1"].isin(categories) & df["good_day_2"].isin(categories)]

mapping_1 = {
    "Sex": 0,
    "Food": 1,
    "Nice weather": 2,
    "Alcohol": 3,
    "Friends": 4,
    "Sports": 5,
    "Sleep": 6,
}

mapping_2 = {
    "Sex": 7,
    "Food": 8,
    "Nice weather": 9,
    "Alcohol": 10,
    "Friends": 11,
    "Sports": 12,
    "Sleep": 13,
}

df.good_day_1 = df.good_day_1.map(mapping_1)
df.good_day_2 = df.good_day_2.map(mapping_2)

df = pd.concat([df, pd.get_dummies(df["good_day_1"])], axis=1)
df = pd.concat([df, pd.get_dummies(df["good_day_2"])], axis=1)

# for feature selection
for cat in categories:
    print(cat, df["gender"][odi["good_day_1"] == cat].value_counts(normalize=True))
    print("\n")
    print(cat, df["gender"][odi["good_day_2"] == cat].value_counts(normalize=True))

corr_mat_2 = df.corr().round(4)

y = df["gender"]
X = df[[1, 3, 5, 6, 7, 9, 10, 11, 12, 13]]

corr_mat_3 = X.corr().round(4)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

lin_regr = LinearRegression()
lin_regr.fit(X_train, y_train)

score_linear = lin_regr.score(X_train, y_train)
y_pred_linear = lin_regr.predict(X_test)
pred_score_linear = r2_score(y_test, y_pred_linear)
