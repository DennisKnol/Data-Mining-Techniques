import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, tree
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


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

    # fill unknown Cabin with 0, known cabin with 1
    data["Cabin"] = (data["Cabin"].notnull()).astype('int')
    data["Cabin"] = data["Cabin"].fillna(0)

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


train_data_prepped = prep_data(train_data)

y = train_data_prepped["Survived"]
X = train_data_prepped[["Pclass", "Sex", "SibSp", "Parch", "Cabin", "Embarked", "FareBins", "AgeCategories", "Title"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

accuracy_dtc = []
accuracy_knn = []
accuracy_gbc = []
accuracy_svc = []
accuracy_nnc = []
accuracy_rfc = []

for _ in range(1):
    # Decision Tree Classifier
    dtc = tree.DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    y_predict = dtc.predict(X_test)
    accuracy_dtc.append(round(accuracy_score(y_predict, y_test)*100, 2))

    # Create and fit a nearest-neighbor classifier
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_predict = knn.predict(X_test)
    accuracy_knn.append(round(accuracy_score(y_predict, y_test)*100, 2))

    # Gradient boosting
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    y_predict = gbc.predict(X_test)
    accuracy_gbc.append(round(accuracy_score(y_predict, y_test)*100, 2))

    # Support Vector Classification
    svc = svm.SVC()
    svc.fit(X_train, y_train)
    y_predict = svc.predict(X_test)
    accuracy_svc.append(round(accuracy_score(y_predict, y_test) * 100, 2))

    # Neural Network Classification
    nnc = MLPClassifier()
    nnc.fit(X_train, y_train)
    y_predict = nnc.predict(X_test)
    accuracy_nnc.append(round(accuracy_score(y_predict, y_test) * 100, 2))

    # Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    y_predict = rfc.predict(X_test)
    accuracy_rfc.append(round(accuracy_score(y_predict, y_test) * 100, 2))

print("dtc:      ", np.mean(accuracy_dtc))
print("knn:      ", np.mean(accuracy_knn))
print("gbc:      ", np.mean(accuracy_gbc))
print("svm:      ", np.mean(accuracy_svc))
print("nnc:      ", np.mean(accuracy_nnc))
print("rfc:      ", np.mean(accuracy_rfc))

# dtc:       80.46240000000002
# knn:       83.39
# gbc:       83.7266
# svm:       81.35999999999999
# nnc:       81.65709999999999
# rfc:       81.06819999999999
# --> Best classifier is Gradient Boosting

test_data_prepped = prep_data(test_data)
X = test_data_prepped[["Pclass", "Sex", "SibSp", "Parch", "Cabin", "Embarked", "FareBins", "AgeCategories", "Title"]]

prediction_survived = pd.DataFrame(gbc.predict(X))
submission = pd.concat([test_data["PassengerId"], prediction_survived], axis=1)
submission.columns = ["PassengerId", "Survived"]
submission.to_csv('survived_submission.csv', index=False)

