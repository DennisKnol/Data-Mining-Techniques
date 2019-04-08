import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

plt.figure(figsize=(14,14))

plt.subplot2grid((3, 2), (0, 0))
train_data.Age.value_counts().sort_index().plot(kind="bar", alpha=0.5)
plt.xticks([])
plt.ylabel("Count")
plt.title("Age")

plt.subplot2grid((3, 2), (0, 1))
plt.scatter(train_data.Survived, train_data.Age, alpha=0.1)
plt.ylabel("Age")
plt.title("Survived vs Age")

plt.subplot2grid((3, 2), (1, 0), colspan=2)
train_data.Age.value_counts().sort_index().plot(kind="bar", alpha=0.5)
plt.ylabel("Count")
plt.title("Age")

plt.subplot2grid((3, 2), (2, 0), colspan=2)
[train_data.Age[train_data.Pclass == i].plot.kde(bw_method=0.3) for i in [1, 2, 3]]
plt.legend(["First class", "Second class", "Third class"])
plt.title("Density plot Age wrt Class of Travel")

plt.show()

# to conclude:
# no a clear relation between age and survived
# However, we see that the average age increase from the third class to the first class
