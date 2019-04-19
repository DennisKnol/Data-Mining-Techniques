import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

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
sns.pairplot(df, y_vars=cols[0], x_vars=cols[1:6], kind='reg')
plt.show()

plt.figure(figsize=(18, 5))
plt.subplot2grid((1, 3), (0, 0))
sns.boxplot(x='weekend', y='consumption', data=df, orient='v', width=0.5)
plt.title("Beer Consumption")
plt.xlabel("Weekend")
plt.ylabel("Liters")

plt.subplot2grid((1, 3), (0, 1))
df["consumption"].plot()
plt.title("Beer Consumption")
plt.ylabel("Liters")
plt.xlabel("Days")

plt.subplot2grid((1, 3), (0, 2))
df["max_temp"].plot()
plt.title("Maximum Temperature")
plt.ylabel("Degrees Celsius")
plt.xlabel("Days")

plt.show()


# prepared data divided in X and y
y = df["consumption"]
X = df[["max_temp", "precipitation", "weekend"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

lin_regr = LinearRegression()
lin_regr.fit(X_train, y_train)

score_linear = lin_regr.score(X_train, y_train)
y_pred_linear = lin_regr.predict(X_test)
pred_score_linear = r2_score(y_test, y_pred_linear)

plt.figure(figsize=(18, 4))

plt.subplot2grid((1, 2), (0, 0))
error_linear = y_test - y_pred_linear
sns.distplot(error_linear, hist=True, kde=False)
plt.xlabel("Error")
plt.title("Error distribution of the linear model")

mse_linear = mean_squared_error(y_test, y_pred_linear)
mae_linear = mean_absolute_error(y_test, y_pred_linear)

# define different set of explanatory and dependent variable for the logistic regression
y = df["weekend"]
X = df[["consumption", "mean_temp"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

log_regr = LogisticRegression()
log_regr.fit(X_train, y_train)

score_log = log_regr.score(X_train, y_train)
y_pred_log = log_regr.predict(X_test)
pred_score_log = r2_score(y_test, y_pred_log)

plt.subplot2grid((1, 2), (0, 1))
error_log = y_test - y_pred_log
sns.distplot(error_log, hist=True, kde=False)
plt.xlabel("Error")
plt.title("Error distribution of the logarithmic model")
plt.show()

mse_log = mean_squared_error(y_test, y_pred_log)
mae_log = mean_absolute_error(y_test, y_pred_log)

# results
print("MSE of linear regression = ", round(mse_linear, 4), '\n')
print("MAE of linear regression =  ", round(mae_linear, 4), '\n')
print("MSE of logistic regression = ", round(mse_log, 4), '\n')
print("MAE of logistic regression = ", round(mae_log, 4))



