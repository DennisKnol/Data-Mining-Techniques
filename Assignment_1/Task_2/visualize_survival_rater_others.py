import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


plt.figure(figsize=(14,14))

plt.subplot2grid((3, 2), (0, 0), colspan=2)
sns.barplot(x="Parch", y="Survived", data=train_data, alpha=0.6)
plt.title("Survival Rate for each Parent to Child Ratio")

plt.subplot2grid((3, 2), (1, 0), colspan=2)
sns.barplot(x="SibSp", y="Survived", data=train_data, alpha=0.6)
plt.title("Survival Rate for each Number of Sibling/Spouse aboard")

plt.show()
