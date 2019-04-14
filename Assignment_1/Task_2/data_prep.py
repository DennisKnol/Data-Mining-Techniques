import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


def remove_outlier(data):
    outliers = []
    for col_name in ["Age", "Parch", "SibSp", "Fare"]:
        q1 = data[col_name].quantile(0.25)
        q3 = data[col_name].quantile(0.75)
        iqr = q3-q1
        lower_bound = q1-1.5*iqr
        upper_bound = q3+1.5*iqr
        # outliers.extend(data[(data[col_name] < lower_bound) | data[col_name] > upper_bound][col_name].index)
        print(data[(data[col_name] < lower_bound) | data[col_name] > upper_bound][col_name].index)
    data.drop(outliers, axis=0).reset_index(drop=True)
    return data


def prep_data(data):
    # convert sex: male is 0, female is 1
    data.loc[data["Sex"] == "male", "Sex"] = 0
    data.loc[data["Sex"] == "female", "Sex"] = 1

    # fill empty Fare with mean Fare in the corresponding Pclass
    mean_fare_per_class = data.groupby("Pclass")["Fare"].mean()
    data["Fare"] = data[["Fare", "Pclass"]].apply(
        lambda x: mean_fare_per_class[x["Pclass"]] if pd.isnull(x["Fare"]) else x["Fare"], axis=1
    )

    # sort Fare into bins
    data["FareBins"] = pd.cut(data["Fare"], 10, labels=[i+1 for i in range(10)])

    # fill empty Age with mean age in the corresponding Pclass
    mean_age_per_class = data.groupby("Pclass")["Age"].mean()
    data["Age"] = data[["Age", "Pclass"]].apply(
        lambda x: mean_age_per_class[x["Pclass"]] if pd.isnull(x["Age"]) else x["Age"], axis=1
    )

    # categorize age in
    labels = ['Baby', 'Child', 'Youth', 'Adult', 'Senior']
    bins = [0, 5, 15, 24, 65, np.inf]
    data['AgeCategories'] = pd.cut(data["Age"], bins, labels=labels)

    age_mapping = {
        "Baby": 1,
        "Child": 2,
        "Youth": 3,
        "Adult": 4,
        "Senior": 5
    }

    data["AgeCategories"] = data["AgeCategories"].map(age_mapping)

    # fill unknown Cabin with 0, known cabins with 1-8 (from most appearing to least appearing)
    data["Cabin"] = data["Cabin"].fillna("U")
    data['Cabin'] = data['Cabin'].apply(lambda x: x[0])

    cabin_mapping = {
        "U": 0,
        "C": 1,
        "B": 2,
        "D": 3,
        "E": 4,
        "A": 5,
        "F": 6,
        "G": 7,
        "T": 8,
    }

    data["Cabin"] = data["Cabin"].map(cabin_mapping)

    # fill empty Embarked with value that appears most often
    data.Embarked = data.Embarked.fillna(data.Embarked.mode()[0])

    # convert Embarked to integers
    data.loc[data["Embarked"] == "S", "Embarked"] = 0
    data.loc[data["Embarked"] == "C", "Embarked"] = 1
    data.loc[data["Embarked"] == "Q", "Embarked"] = 2

    # subtract title from name
    data["Title"] = data.Name.apply(lambda x: x.split('.')[0].split(',')[1].strip())
    data.loc[data["Title"] == "Ms", "Title"] = "Miss"
    data.loc[data["Title"] == "Mlle", "Title"] = "Miss"
    data.loc[data["Title"] == "Mme", "Title"] = "Mrs"
    data.loc[data["Title"] == "Mme", "Title"] = "Mrs"

    rare_titles = ["Dr", "Rev", "Col", "Major", "Jonkheer", "Don", "the Countess", "Lady", "Sir"]
    for title in rare_titles:
        data.loc[data["Title"] == title, "Title"] = "Rare"

    title_mapping = {
        "Unknown": 0,
        "Mr": 1,
        "Miss": 2,
        "Mrs": 3,
        "Master": 4,
        "Rare": 5,
        "Capt": 6
    }

    data["Title"] = data["Title"].map(title_mapping)
    data["Title"] = data["Title"].fillna(0)

    # drop Age, Name and Ticket
    data = data.drop(["Age"], axis=1)
    data = data.drop(["Fare"], axis=1)
    data = data.drop(["Name"], axis=1)
    data = data.drop(["Ticket"], axis=1)
    return data


train_data_outlier_removed = remove_outlier(train_data)
print(train_data_outlier_removed.shape)
train_data_prepped = prep_data(train_data_outlier_removed)

sns.set()
cols = ["Survived", "Pclass", "Sex", "SibSp", "Parch", "Cabin", "Embarked", "Fare", "Age"]
sns.pairplot(train_data[cols], height=2.5)
plt.show()


y = train_data_prepped["Survived"]
X = train_data_prepped[["Pclass", "Sex", "SibSp", "Parch", "Cabin", "Embarked", "FareBins", "AgeCategories", "Title"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

models = [
    DecisionTreeClassifier(),
    KNeighborsClassifier(),
    GradientBoostingClassifier(),
    SVC(),
    MLPClassifier(),
    RandomForestClassifier()
]

for model in models:
    model.fit(X_train, y_train)
    score = round(model.score(X_test, y_test), 2)
    print(score)

