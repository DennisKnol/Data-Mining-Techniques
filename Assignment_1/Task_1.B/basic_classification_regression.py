import pandas as pd

import seaborn as sns
import matplotlib.pylab as plt

from sklearn.model_selection import train_test_split, KFold, RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier


df = pd.read_csv(
    "/Users/DennisK/PycharmProjects/Data-Mining-Techniques/Assignment_1/Task_1.B/winequality-red.csv",
    sep=';'
)

print(df.info())
print(df.shape)
print(df.isnull().sum())

corr_mat = df.corr()

# visualize data
cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol', 'quality']

sns.set()
sns.pairplot(df, y_vars=cols[-1], x_vars=cols[0:11], kind='reg')
plt.show()

sns.distplot(df["quality"])
plt.show()

df.loc[df["quality"] <= 6, "quality"] = 0
df.loc[df["quality"] > 6, "quality"] = 1

sns.distplot(df["quality"])
plt.show()

y = df["quality"]
X = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=2652124)
# kf = KFold(n_splits=2, random_state=None)
#
# for train_index, test_index in kf.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

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
