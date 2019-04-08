import pandas as pd
import matplotlib.pyplot as plt


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

titanic_data = pd.concat([train_data, test_data], sort=False)


def missing_values_count(data):
    total = len(data)
    missing = (total - (data.count()[data.count() < total]))
    return missing


print("Missing data in train set\n", missing_values_count(train_data), "\n")
print("Missing data in test set \n", missing_values_count(test_data))


# Mean fare per class
print(test_data.groupby("Pclass")["Fare"].mean())


def prep_data(data):
    data.Fare = data.Fare.fillna()





#
#
# # Data exploration
# plt.subplot2grid((3, 2), (0, 0), colspan=2)
# train_data.Fare.value_counts().sort_index().plot(kind="bar", alpha=0.5)
# plt.xticks([])
# plt.title("Fare")
#
# plt.subplot2grid((3, 2), (1, 0), colspan=2)
# train_data.Embarked.value_counts().sort_index().plot(kind="bar", alpha=0.5)
# plt.title("Embarked")
#
# plt.subplot2grid((3, 2), (2, 0), colspan=2)
# # [train_data.Survived[train_data.Parch == i].value_counts().sort_index().plot.kde(bw_method=0.3) for i in [0, 1, 2, 3, 4, 5, 6]]
# [train_data.Survived[train_data.Parch == i].plot.kde(bw_method=0.3) for i in [0, 1, 2, 3, 5]]
# plt.legend(["0", "1", "2", "3", "5"])
# plt.title("Density plot")
#
# plt.show()
#
# print(max(train_data.Parch))
#
# print(train_data.Parch.value_counts().sort_index())