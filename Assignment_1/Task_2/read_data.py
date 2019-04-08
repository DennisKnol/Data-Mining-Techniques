import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# read in data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

print(train_data.count())
# missing values for Age, 714 (out of 891) are known
# missing values for Cabin, 204 (out of 891) are known

print(test_data.count())
# missing values for Age, 332 (out of 418) are known
# missing values for Cabin, 91 (out of 418) are known
# and 1 missing value for Fare

# Data exploration
plt.subplot2grid((3, 2), (0, 0))
train_data.Survived.value_counts().plot(kind="bar", alpha=0.5)
plt.title("Survived")

plt.subplot2grid((3, 2), (0, 1))
train_data.Pclass.value_counts().sort_index().plot(kind="bar", alpha=0.5)
plt.title("Class of Travel")

plt.subplot2grid((3, 2), (1, 0))
train_data.Sex.value_counts().plot(kind="bar", alpha=0.5)
plt.title("Gender")

plt.subplot2grid((3, 2), (1, 1))
train_data.Age.value_counts().sort_index().plot(kind="bar", alpha=0.5)
plt.xticks([])
plt.title("Age")

plt.subplot2grid((3, 2), (2, 0))
train_data.Fare.value_counts().sort_index().plot(kind="bar", alpha=0.5)
plt.xticks([])
plt.title("Fare")

plt.subplot2grid((3, 2), (2, 1))
train_data.Embarked.value_counts().sort_index().plot(kind="bar", alpha=0.5)
plt.title("Embarked")

plt.subplot2grid((3, 2), (3, 0))
plt.scatter(train_data.Survived, train_data.Age, alpha=0.1)
plt.title("Survived vs Age")

plt.show()


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
train_data.Sex[train_data.Survived == 1]\
    .value_counts(normalize=True)\
    .sort_index()\
    .plot(
    kind="bar",
    alpha=0.8,
    color=['pink','lightblue']
)
plt.title("Survived Per Gender")


plt.show()




