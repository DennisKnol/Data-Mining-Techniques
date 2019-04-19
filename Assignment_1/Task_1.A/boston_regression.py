import matplotlib.pylab as plt

from sklearn import linear_model, datasets
from sklearn.model_selection import cross_val_predict

boston = datasets.load_boston()

y = boston.target
X = boston.data

model = linear_model.LinearRegression()
predicted = cross_val_predict(model, X, y, cv=10)

fig, ax = plt.subplots()
ax.scatter(y, predicted, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()