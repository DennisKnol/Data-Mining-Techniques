import pandas as pd
import numpy as np
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
mean_fare_per_class = titanic_data.groupby("Pclass")["Fare"].mean()
print(mean_fare_per_class[3])

print(test_data.Pclass[test_data.Fare.isnull()])

def prep_data(data):
    mean_fare_per_class = titanic_data.groupby("Pclass")["Fare"].mean()
    # for i in [test_data.Pclass[test_data.Fare.isnull()]:
        # data.Fare = data.Fare.fillna()





# def cleanData(data):
#   # If fare data is missing, replace it with the average from that class
#   data.Fare = data.Fare.map(lambda x: np.nan if x==0 else x)
#   classmeans = data.pivot_table('Fare', rows='Pclass', aggfunc='mean')
#   data.Fare = data[['Fare', 'Pclass']].apply(lambda x: classmeans[x['Pclass']] if pd.isnull(x['Fare']) else x['Fare'], axis=1 )
#
#
#   # Covert sex into a numberic value
#   data.Sex = data.Sex.apply(lambda sex: 0 if sex == "male" else 1)
#
#   return data


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