import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import r2_score


df = pd.read_csv('SmsCollection.csv', delimiter=';', usecols=[0, 1])

# convert to 0 and 1
df.loc[df["label"] == "ham", "label"] = 0
df.loc[df["label"] == "spam", "label"] = 1
df["label"] = df["label"].astype('int')

print(df["label"].value_counts(normalize=True))
print(df.isnull().sum())

df = df.dropna()

# convert to lowercase
df["text"] = df["text"].str.lower()

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.1, random_state=1)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)

svm = svm.SVC(C=1000)
svm.fit(X_train, y_train)

X_test = vectorizer.transform(X_test)
y_pred = svm.predict(X_test)
print(r2_score(y_test, y_pred))
