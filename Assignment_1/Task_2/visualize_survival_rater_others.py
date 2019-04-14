import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


plt.figure(figsize=(14,14))

plt.subplot2grid((4, 2), (0, 0), colspan=2)
# sns.barplot(x="Parch", y="Survived", data=train_data, alpha=0.6)
# plt.title("Survival Rate for each Parent to Child Ratio")
plt.scatter(train_data.PassengerId, train_data.Parch)
plt.ylabel("Parch")
plt.xlabel("PassengerId")
plt.title("scatter of fare wrt passenger id")

plt.subplot2grid((4, 2), (1, 0), colspan=2)
#sns.barplot(x="SibSp", y="Survived", data=train_data, alpha=0.6)
# plt.title("Survival Rate for each Number of Sibling/Spouse aboard")
plt.scatter(train_data.PassengerId, train_data.SibSp)
plt.ylabel("SibSp")
plt.xlabel("PassengerId")
plt.title("scatter of fare wrt passenger id")

plt.subplot2grid((4, 2), (2, 0), colspan=2)
plt.scatter(train_data.PassengerId, train_data.Fare)
plt.ylabel("Fare")
plt.xlabel("PassengerId")
plt.title("scatter of fare wrt passenger id")

plt.subplot2grid((4, 2), (3, 0), colspan=2)
plt.scatter(train_data.PassengerId, train_data.Age)
plt.ylabel("Age")
plt.xlabel("PassengerId")
plt.title("scatter of age wrt passenger id")

plt.show()
