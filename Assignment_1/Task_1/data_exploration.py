import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pylab
import seaborn as sns


odi_raw = pd.read_csv(
    "/Users/DennisK/PycharmProjects/Data-Mining-Techniques/Assignment_1/Task_1/ODI-2019-csv.csv",
    sep=';'
)

shape_odi_raw = odi_raw.shape
print(odi_raw.info())

odi = pd.read_csv(
    "/Users/DennisK/PycharmProjects/Data-Mining-Techniques/Assignment_1/Task_1/ODI_2019_clean.csv",
    sep=','
)

odi = odi.drop(odi.columns[0], axis=1)

plt.figure(figsize=(15, 8))
odi["programme"].value_counts().plot(
    kind='bar',
    title='What programme are you in?'
)
plt.ylabel("Count")
plt.show()

print(round(odi["gender"].value_counts(normalize=True), 2))

plt.figure(figsize=(18, 8))

plt.subplot2grid((2, 3), (0, 0))
odi["machine_learning"].value_counts().sort_index().plot(
    kind='pie',
    labels=["no", "yes", "unknown"],
    colors=["lightblue", "teal", "wheat"],
    autopct="%.1f%%",
    title="Have you taken a course on machine learning?"
    )
plt.axis('off')

plt.subplot2grid((2, 3), (0, 1))
odi["information_retrieval"].value_counts().sort_index().plot(
    kind='pie',
    labels=["no", "yes", "unknown"],
    colors=["lightblue", "teal", "wheat"],
    autopct="%.1f%%",
    title="Have you taken a course on information retrieval?"
    )
plt.axis('off')

plt.subplot2grid((2, 3), (0, 2))
odi["statistics"].value_counts().sort_index().plot(
    kind='pie',
    labels=["no", "yes", "unknown"],
    colors=["lightblue", "teal", "wheat"],
    autopct="%.1f%%",
    title="Have you taken a course on statistics?"
    )
plt.axis('off')

plt.subplot2grid((2, 3), (1, 0))
odi["databases"].value_counts().sort_index().plot(
    kind='pie',
    labels=["no", "yes", "unknown"],
    colors=["lightblue", "teal", "wheat"],
    autopct="%.1f%%",
    title="Have you taken a course on databases?"
)
plt.axis('off')

plt.subplot2grid((2, 3), (1, 1))
odi["stand_up"].value_counts().sort_index().plot(
    kind='pie',
    labels=["no", "yes", "unknown"],
    colors=["lightblue", "teal", "wheat"],
    autopct="%.1f%%",
    title="Did you stand up?"
)
plt.axis('off')

plt.subplot2grid((2, 3), (1, 2))
odi["chocolate"].value_counts().sort_index().plot(
    kind='pie',
    labels=["fat", "slim", "neither", "I have no idea what you are talking about", "unknown"],
    colors=["lavender", "lightblue", "teal", "wheat", "beige"],
    autopct="%.1f%%",
    title="Chocolate makes you ..."
)
plt.axis('off')

plt.show()

