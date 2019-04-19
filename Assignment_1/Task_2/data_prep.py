import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from collections import Counter


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


def find_outlier(data):
    """
    Function finding outliers with the IQR method.

    Returns a list of all rows that contain 3 or more outliers

    """

    outliers = []
    for col_name in ["Age", "SibSp", "Parch", "Fare"]:
        q1 = data[col_name].quantile(0.25)
        q3 = data[col_name].quantile(0.75)
        iqr = q3-q1
        lower_bound = q1 - (1.5*iqr)
        upper_bound = q3 + (1.5*iqr)
        outliers.extend(data[(data[col_name] < lower_bound) | (data[col_name] > upper_bound)].index)
    rows = list(k for k, v in Counter(outliers).items() if v > 2)
    return rows


def remove_outlier(data, outliers):
    """
    Function removing the outliers found by the function find_outlier()

    """
    data = data.drop(outliers, axis=0).reset_index(drop=True)
    print(len(outliers), " rows have been deleted")
    return data


def prep_data(data):
    """
    Function preparing the dataset for classification

    """
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

    # fill empty Age with mean age in the corresponding Title
    mean_age_per_title = data.groupby("Title")["Age"].mean()
    data["Age"] = data[["Age", "Title"]].apply(
        lambda x: mean_age_per_title[x["Title"]] if pd.isnull(x["Age"]) else x["Age"], axis=1
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

    # drop Age, Name and Ticket
    data = data.drop(["Age"], axis=1)
    data = data.drop(["Fare"], axis=1)
    data = data.drop(["Name"], axis=1)
    data = data.drop(["Ticket"], axis=1)
    return data


# find and remove outliers
train_data = remove_outlier(train_data, find_outlier(train_data))

# prepare train data and test data
train_data_prep = prep_data(train_data)
test_data_prepped = prep_data(test_data)

# prepared saved in csv file
train_data_prep.to_csv("train_prep.csv")
test_data_prepped.to_csv("test_prep.csv")


# # Density plot Age wrt Title
# plt.figure(figsize=(18, 6))
# colors = ["lightblue", "pink", "teal", "wheat", "grey", "lavender"]
# labels = ['Mr', "Miss", "Mrs", "Master", "Rare", "Capt"]
# [train_data_prepped.Age[train_data_prepped.Title == i].plot.kde(
#     bw_method=0.3,
#     color=colors[i-1],
#     label=labels[i-1]
# )
#     for i in [1, 2, 3, 5]
# ]
# plt.legend()
# plt.xlabel("Age")
# plt.xlim((-5, 85))
# plt.title("Density plot Age wrt Title")
# plt.show()
