import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

plt.figure(figsize=(14,14))

plt.subplot2grid((5, 2), (0, 0))
train_data.Age.value_counts().sort_index().plot(kind="bar", alpha=0.5)
plt.xticks([])
plt.ylabel("Count")
plt.title("Age")

plt.subplot2grid((5, 2), (0, 1))
plt.scatter(train_data.Survived, train_data.Age, alpha=0.1)
plt.ylabel("Age")
plt.title("Survived vs Age")

plt.subplot2grid((5, 2), (1, 0), colspan=2)
train_data.Age.value_counts().sort_index().plot(kind="bar", alpha=0.5)
plt.ylabel("Count")
plt.title("Age")

plt.subplot2grid((5, 2), (2, 0), colspan=2)
# [train_data.Age[train_data.Pclass == i].value_counts().sort_index().plot(kind="kde") for i in [1, 2, 3]]
train_data.Age[train_data.Pclass == 1].value_counts().sort_index().plot.kde(bw_method=0.3)
plt.legend(["First class"])
plt.title("Density plot Age wrt First Class of Travel")

plt.subplot2grid((5, 2), (3, 0), colspan=2)
train_data.Age[train_data.Pclass == 2].value_counts().sort_index().plot(kind="bar", alpha=0.8)
plt.legend(["Second class"])
plt.title("Density plot Age wrt Second Class of Travel")

plt.subplot2grid((5, 2), (4, 0), colspan=2)
train_data.Age[train_data.Pclass == 3].value_counts().sort_index().plot(kind="bar", alpha=0.8)
plt.legend(["Third class"])
plt.title("Density plot Age wrt Third Class of Travel")

plt.show()

# to conclude:
# no a clear relation between age and survived
# However, we see that the average age increase from the third class to the first class
