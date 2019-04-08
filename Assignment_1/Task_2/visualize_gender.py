import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

plt.figure(figsize=(14,14))

plt.subplot2grid((3, 2), (0, 0))
train_data.Survived[train_data.Sex == "male"]\
    .value_counts(normalize=True)\
    .sort_index()\
    .plot(
    kind="bar",
    alpha=0.8,
    color='lightblue'
)
plt.title("Men Survived")

plt.subplot2grid((3, 2), (0, 1))
train_data.Survived[train_data.Sex == "female"]\
    .value_counts(normalize=True)\
    .sort_index()\
    .plot(
    kind="bar",
    alpha=0.8,
    color='pink'
)
plt.title("Women Survived")

plt.subplot2grid((3, 2), (1, 0))
train_data.Survived.value_counts().plot(kind="bar", alpha=0.5)
plt.title("Survived")

plt.subplot2grid((3, 2), (1, 1))
train_data.Sex[train_data.Survived == 1]\
    .value_counts(normalize=True)\
    .sort_index()\
    .plot(
    kind="bar",
    alpha=0.8,
    color=['pink','lightblue']
)
plt.title("Survived Per Gender")

plt.subplot2grid((3,2), (2,0), colspan=2)
[train_data.Survived[train_data.Sex == x].plot.kde(bw_method=0.3) for x in ["male", "female"]]
plt.legend(["male", "female"])
plt.title("Density Plot of Survived wrt Gender")

plt.show()
