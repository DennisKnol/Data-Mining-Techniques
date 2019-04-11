import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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

# Consider Age categories instead of specific ages
train_data.Age = train_data.Age.fillna(-0.5)
test_data.Age = test_data.Age.fillna(-0.5)

labels = ['Unknown', 'Baby', 'Children', 'Youth', 'Adults', 'Seniors']
bins = [-1, 0, 5, 15, 24, 65, np.inf]

train_data['AgeCategories'] = pd.cut(train_data["Age"], bins, labels=labels)
test_data['AgeCategories'] = pd.cut(test_data["Age"], bins, labels=labels)

survival_rate_per_category = (
        train_data.AgeCategories[train_data.Survived == 1].value_counts(normalize=True).sort_index()/
        train_data.AgeCategories[train_data.Survived == 0].value_counts(normalize=True).sort_index()
)

survival_rate_per_category.plot(kind="bar", alpha=0.7)
plt.ylabel("survival rate")
plt.title("Survival Rate per Age Category")
plt.show()


# to conclude:
# no a clear relation between age and survived when considering the scatter plot
# However, when considering age categories we find that babies are most likely to survive and seniors least likely.
#
# Also, we see that the average age increase from the third class to the first class
