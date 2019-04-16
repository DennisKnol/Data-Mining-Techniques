import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ridge_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv('Consumo_cerveja.csv')
df = df.dropna()
df_shape = df.shape

col_names = [
    "date",
    "mean_temp",
    "min_temp",
    "max_temp",
    "precipitation",
    "weekend",
    "consumption"
]

for col in col_names:
    df[col] = df.iloc[:, (col_names.index(col))]

df[["year", "month", "day"]] = df['date'].str.split("-", expand=True)

# convert values from objects to integers
for col in ["year", "month", "day"]:
    df[col] = df[col].apply(pd.to_numeric, errors='coerce', downcast='integer')

# select only the renamed columns
df = df.iloc[:, [i for i in range(7, 17)]]

# replace comma with dot and convert to float
for col in ["mean_temp", "min_temp", "max_temp", "precipitation"]:
    df[col] = df[col].apply(lambda x: str(x))
    df[col] = df[col].apply(lambda x: x.replace(',', '.'))
    df[col] = df[col].apply(pd.to_numeric, errors='coerce', downcast='float')

# create correlation matrix
corr_mat = df.corr().round(4)

# visualize the prepared data
cols = ["consumption", "mean_temp", "min_temp", "max_temp", "precipitation", "weekend"]

sns.set()
sns.pairplot(df[cols], kind='reg')
plt.show()

df["consumption"].plot()
plt.show()

# prepared data divided in X and y
y = df["consumption"]
X = df[["max_temp", "precipitation", "weekend"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

lin_regr = LinearRegression()
lin_regr.fit(X_train, y_train)

score_linear = lin_regr.score(X_train, y_train)
y_pred_linear = lin_regr.predict(X_test)
pred_score = r2_score(y_test, y_pred_linear)

# X = df[[ "min_temp", "max_temp", "precipitation", "weekend"]]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#
# ridge_regr = ridge_regression()
# ridge_regr.fit(X_train, y_train)
#
# score_ridge = ridge_regr.score(X_train, y_train)
# y_pred_ridge = ridge_regr.predict(X_test)

