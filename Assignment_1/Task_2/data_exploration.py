import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train_data = pd.read_csv("train.csv")
train_data.info()


def missing_values_count(data):
    total = len(data)
    missing = (total - (data.count()[data.count() < total]))
    return missing


print("Missing data in train set\n", missing_values_count(train_data), "\n")


# Gender
percent_per_gender = train_data["Sex"].value_counts(normalize=True) * 100
percent_per_gender_survived = train_data["Sex"][train_data["Survived"] == 1].value_counts(normalize=True) * 100
print(percent_per_gender_survived)
#
# plt.figure(figsize=(14, 8))
#
# plt.subplot2grid((2, 2), (0, 0))
# train_data.Sex[train_data.Survived == 1]\
#     .value_counts(normalize=True)\
#     .sort_index()\
#     .plot(
#     kind="bar",
#     alpha=0.8,
#     color=['pink','lightblue']
# )
# plt.title("Survived Per Gender")
#
# plt.subplot2grid((2, 2), (0, 1))
# sns.barplot(x="Sex", y="Survived", data=train_data, alpha=0.8, palette=['lightblue', 'pink'])
# plt.title("Survival Rate per Gender")
#
# plt.subplot2grid((2,2), (1,0), colspan=2)
# [train_data.Survived[train_data.Sex == x].plot.kde(bw_method=0.3, color=c)
#  for x, c in zip(["male", "female"], ['lightblue', 'pink'])]
# plt.legend(["male", "female"])
# plt.title("Density Plot of Survived wrt Gender")
#
# plt.show()
#
#
# # Age and age categories
# plt.figure(figsize=(14, 10))
#
# plt.subplot2grid((3, 2), (0, 0))
# train_data.Age.value_counts().sort_index().plot(kind="bar", alpha=0.6)
# plt.xticks([])
# plt.ylabel("Count")
# plt.title("Age")
#
# plt.subplot2grid((3, 2), (0, 1))
# plt.scatter(train_data.Survived, train_data.Age, alpha=0.1)
# plt.ylabel("Age")
# plt.title("Survived vs Age")
#
# plt.subplot2grid((3, 2), (1, 0), colspan=2)
# [train_data.Age[train_data.Pclass == i].plot.kde(bw_method=0.3) for i in [1, 2, 3]]
# plt.legend(["First class", "Second class", "Third class"])
# plt.title("Density plot Age wrt Class of Travel")
#
# # Consider Age categories instead of specific ages
# train_data.Age = train_data.Age.fillna(-0.5)
#
# labels = ['Unknown', 'Babies', 'Children', 'Youth', 'Adults', 'Seniors']
# bins = [-1, 0, 5, 15, 24, 65, np.inf]
#
# train_data['AgeCategories'] = pd.cut(train_data["Age"], bins, labels=labels)
#
# survival_rate_per_category = (
#         train_data.AgeCategories[train_data.Survived == 1].value_counts(normalize=True).sort_index()/
#         train_data.AgeCategories[train_data.Survived == 0].value_counts(normalize=True).sort_index()
# )
#
# plt.subplot2grid((3, 2), (2, 0), colspan=2)
# survival_rate_per_category.plot(kind="bar", alpha=0.6)
# plt.ylabel("survival rate")
# plt.title("Survival Rate per Age Category")
#
# plt.show()
#
#
# # Class of travel
# plt.figure(figsize=(14, 7))
#
# plt.subplot2grid((3, 2), (0, 0))
# train_data.Pclass.value_counts().sort_index().plot(kind="bar", alpha=0.6)
# plt.title("Class of Travel, passenger count")
#
# plt.subplot2grid((3, 2), (0, 1))
# sns.barplot(x="Pclass", y="Survived", data=train_data, alpha=0.6)
# plt.title("Survival Rate per Class of Travel")
#
# plt.subplot2grid((3, 2), (1, 0), colspan=2)
# [train_data.Survived[train_data.Pclass == i].plot.kde(bw_method=0.3) for i in [1, 2, 3]]
# plt.legend(["First class", "Second class", "Third class"])
# plt.title("Density plot Survived wrt Class of Travel")
#
# plt.subplot2grid((3, 2), (2, 0), colspan=2)
# [train_data.Fare[train_data.Pclass == i].plot.kde(bw_method=0.3) for i in [1, 2, 3]]
# plt.legend(["First class", "Second class", "Third class"])
# plt.title("Density plot Fare wrt Class of Travel")
#
# plt.show()

#
train_data.Fare.value_counts().sort_index().plot(kind="bar", alpha=0.6)
plt.xticks([])
plt.ylabel("Count")
plt.title("Fare")
plt.show()