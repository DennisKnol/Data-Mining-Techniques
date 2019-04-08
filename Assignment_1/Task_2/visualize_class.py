import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

plt.figure(figsize=(14,14))

plt.subplot2grid((3, 2), (0, 0), colspan=2)
train_data.Pclass.value_counts().sort_index().plot(kind="bar", alpha=0.5)
plt.title("Class of Travel")

plt.subplot2grid((3, 2), (1, 0), colspan=2)
[train_data.Survived[train_data.Pclass == i].plot.kde(bw_method=0.3) for i in [1, 2, 3]]
plt.legend(["First class", "Second class", "Third class"])
plt.title("Density plot Survived wrt Class of Travel")

plt.subplot2grid((3, 2), (2, 0), colspan=2)
[train_data.Fare[train_data.Pclass == i].plot.kde(bw_method=0.3) for i in [1, 2, 3]]
plt.legend(["First class", "Second class", "Third class"])
plt.title("Density plot Fare wrt Class of Travel")

plt.show()

# to conclude:
# We see a clear difference in survival rates in the different classes
# Survival rates increases from the third class to the first
# Also, fare increases from the third class to the first
