import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier


train_data_prepped = pd.read_csv("train_prep.csv")
test_data_prepped = pd.read_csv("test_prep.csv")

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

X = test_data_prepped[["Pclass", "Sex", "SibSp", "Parch", "Cabin", "Embarked", "FareBins", "AgeCategories", "Title"]]

# create submission file
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
prediction_survived = pd.DataFrame(rfc.predict(X))
submission = pd.concat([test_data_prepped["PassengerId"], prediction_survived], axis=1)
submission.columns = ["PassengerId", "Survived"]
submission.to_csv('survived_submission.csv', index=False)


