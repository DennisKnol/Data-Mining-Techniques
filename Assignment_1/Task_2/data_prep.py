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


def prep_data(data):
    # convert sex to binary
    data.loc[data["Sex"] == "male", "Sex"] = 0
    data.loc[data["Sex"] == "female", "Sex"] = 1

    # fill empty Fare with mean Fare is the corresponding Pclass
    mean_fare_per_class = titanic_data.groupby("Pclass")["Fare"].mean()
    data["Fare"] = data[["Fare", "Pclass"]].apply(
        lambda x: mean_fare_per_class[x["Pclass"]] if pd.isnull(x["Fare"]) else x["Fare"], axis=1
    )

    # fill empty Cabin with "unknown"
    data.Cabin = data.Cabin.fillna("unknown")

    # fill empty Embarked with value that appears most often
    data.Embarked = data.Embarked.fillna(data.Embarked.mode()[0])
    
    # convert Embarked to integers
    data.loc[data["Embarked"] == "S", "Embarked"] = 0
    data.loc[data["Embarked"] == "C", "Embarked"] = 1
    data.loc[data["Embarked"] == "Q", "Embarked"] = 2

    # TODO: think of a way to fill unknown Ages, also the title of a passenger is correlated with ones survival

    data["Title"] = data.Name.apply(lambda x: x.split('.')[0].split(',')[1].strip())

    return data


prep_data(train_data)
prep_data(test_data)

print("Missing data in train set\n", missing_values_count(train_data), "\n")
print("Missing data in test set \n", missing_values_count(test_data), "\n")


# Test area:

print(train_data.Title.value_counts())
print(pd.crosstab(train_data["Title"], train_data["Sex"]))
print(train_data[["Title", "Survived"]].groupby(["Title"], as_index=False).mean())
