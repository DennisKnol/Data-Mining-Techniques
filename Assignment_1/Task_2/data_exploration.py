import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train_data = pd.read_csv("train.csv")
train_data.info()

test_data = pd.read_csv("test.csv")
test_data.info()

def missing_values_count(data):
    total = len(data)
    missing = (total - (data.count()[data.count() < total]))
    return missing


print("Missing data in train set\n", missing_values_count(train_data), "\n")
print("Missing data in test set\n", missing_values_count(test_data), "\n")


# Gender
percent_per_gender = train_data["Sex"].value_counts(normalize=True) * 100
percent_per_gender_survived = train_data["Sex"][train_data["Survived"] == 1].value_counts(normalize=True) * 100
print(percent_per_gender_survived)

plt.figure(figsize=(18, 8))

plt.subplot2grid((1, 3), (0, 0))
train_data.Sex.value_counts().sort_index().plot(
    kind='pie',
    colors=["pink", "lightblue"],
    autopct="%.1f%%",
    title="Gender"
)
plt.axis('off')

plt.subplot2grid((1, 3), (0, 1))
train_data.Embarked.value_counts().sort_index().plot(
    kind='pie',
    colors=["teal", "wheat", "lightblue"],
    autopct="%.1f%%",
    title="Embarked"
)
plt.axis('off')

plt.subplot2grid((1, 3), (0, 2))
train_data.Pclass.value_counts().plot(
    kind='pie',
    colors=["lightblue", "teal", "wheat"],
    autopct="%.1f%%",
    title="Class of travel"
)
plt.axis('off')

plt.show()


plt.figure(figsize=(18, 4))

plt.subplot2grid((1, 3), (0, 0))
sns.barplot(x="Sex", y="Survived", data=train_data, alpha=0.8, order=["female", "male"], palette=['pink', 'lightblue'])
plt.title("Chance of Survival per Gender")

plt.subplot2grid((1, 3), (0, 1))
sns.barplot(x="Embarked", y="Survived", data=train_data, alpha=0.8, order=["C", "Q", "S"], palette=["lightblue", "teal", "wheat"])
plt.title("Chance of Survival per Port")

plt.subplot2grid((1, 3), (0, 2))
sns.barplot(x="Pclass", y="Survived", data=train_data, alpha=0.8, palette=["lightblue", "teal", "wheat"])
plt.title("Chance of Survival per Class of Travel")

plt.show()

mean_fare_per_class = train_data.groupby("Pclass")["Fare"].mean()
mean_age_per_class = train_data.groupby("Pclass")["Age"].mean()

plt.figure(figsize=(18, 4))

plt.subplot2grid((1, 3), (0, 0))
sns.distplot(train_data.Age.dropna(), hist=True)
plt.title("Distribution plot of Age")

colors = ["lightblue", "teal", "grey"]

plt.subplot2grid((1, 3), (0, 1))
[train_data.Age[train_data.Pclass == i].plot.kde(bw_method=0.3, color=colors[i - 1]) for i in [1, 2, 3]]
plt.xlim((-20, 100))
plt.legend(["First class", "Second class", "Third class"])
plt.title("Density plot Age wrt Class of Travel")

# Consider Age categories instead of specific ages
train_data.Age = train_data.Age.fillna(-0.5)

labels = ['Unknown', 'Babies', 'Children', 'Youth', 'Adults', 'Seniors']
bins = [-1, 0, 5, 15, 24, 65, np.inf]

train_data['AgeCategories'] = pd.cut(train_data["Age"], bins, labels=labels)

survival_rate_per_category = (
        train_data.AgeCategories[train_data.Survived == 1].value_counts(normalize=True).sort_index()/
        train_data.AgeCategories[train_data.Survived == 0].value_counts(normalize=True).sort_index()
)

plt.subplot2grid((1, 3), (0, 2))
survival_rate_per_category.plot(kind="bar", alpha=0.6, color=["lightblue", "teal", "wheat", "beige", "grey", "lavender"])
plt.ylabel("survival rate")
plt.title("Survival Rate per Age Category")

plt.show()

# sns.set()
# cols = ["Survived", "Pclass", "Sex", "SibSp", "Parch", "Cabin", "Embarked", "Fare", "Age"]
# sns.pairplot(train_data[cols], height=2.5)
# plt.show()
