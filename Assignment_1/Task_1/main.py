import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pylab
import seaborn as sns


odi = pd.read_csv(
    "/Users/DennisK/PycharmProjects/Data-Mining-Techniques/Assignment_1/Task_1/ODI-2019-csv.csv",
    sep=';'
)

print(odi.shape)

def data_prep(data):
    # Split timestamp into day and time
    data[["day", "time"]] = data['Timestamp'].str.split(" ", expand=True)

    col_names = [
        "programme",
        "machine_learning",
        "information_retrieval",
        "statistics",
        "databases",
        "gender",
        "chocolate",
        "birthday",
        "number_of_neighbors",
        "stand_up",
        "deserve_money",
        "random_number",
        "bedtime",
        "good_day_1",
        "good_day_2",
        "stress_level"
        ]

    for col in col_names:
        data[col] = data.iloc[:, (col_names.index(col)+1)]

    # select only the renamed columns
    data = data.iloc[:, [i for i in range(17, 35)]]

    # convert answers to numbers
    data.loc[data["machine_learning"] == "no", "machine_learning"] = 0
    data.loc[data["machine_learning"] == "yes", "machine_learning"] = 1
    data.loc[data["machine_learning"] == "unknown", "machine_learning"] = 2

    data.loc[data["information_retrieval"] == "0", "information_retrieval"] = 0
    data.loc[data["information_retrieval"] == "1", "information_retrieval"] = 1
    data.loc[data["information_retrieval"] == "unknown", "information_retrieval"] = 2

    data.loc[data["statistics"] == "sigma", "statistics"] = 0
    data.loc[data["statistics"] == "mu", "statistics"] = 1
    data.loc[data["statistics"] == "unknown", "statistics"] = 2

    data.loc[data["databases"] == "nee", "databases"] = 0
    data.loc[data["databases"] == "ja", "databases"] = 1
    data.loc[data["databases"] == "unknown", "databases"] = 2

    data.loc[data["gender"] == "male", "gender"] = 0
    data.loc[data["gender"] == "female", "gender"] = 1
    data.loc[data["gender"] == "unknown", "gender"] = 2

    data.loc[data["stand_up"] == "no", "stand_up"] = 0
    data.loc[data["stand_up"] == "yes", "stand_up"] = 1
    data.loc[data["stand_up"] == "unknown", "stand_up"] = 2
    return data


odi = data_prep(odi)


# ODI_cleaned["stand_up"].value_counts().plot(kind='bar', title='did you stand up?')
# plt.show()

plt.figure(figsize=(10, 10))

plt.subplot2grid((3, 2), (0, 0))
odi["machine_learning"].value_counts().sort_index().plot(
    kind='pie',
    y='leeg',
    labels=["no", "yes", "unknown"],
    colors=["lightblue", "teal", "wheat"],
    autopct="%.1f%%",
    title="Have you taken a course on machine learning?"
    )
plt.axis('off')

plt.subplot2grid((3, 2), (0, 1))
odi["information_retrieval"].value_counts().sort_index().plot(
    kind='pie',
    labels=["no", "yes", "unknown"],
    colors=["lightblue", "teal", "wheat"],
    autopct="%.1f%%",
    title="Have you taken a course on information retrieval?"
    )
plt.axis('off')

plt.subplot2grid((3, 2), (1, 0))
odi["statistics"].value_counts().sort_index().plot(
    kind='pie',
    labels=["no", "yes", "unknown"],
    colors=["lightblue", "teal", "wheat"],
    autopct="%.1f%%",
    title="Have you taken a course on statistics?"
    )
plt.axis('off')

plt.subplot2grid((3, 2), (1, 1))
odi["databases"].value_counts().sort_index().plot(
    kind='pie',
    labels=["no", "yes", "unknown"],
    colors=["lightblue", "teal", "wheat"],
    autopct="%.1f%%",
    title="Have you taken a course on databases?"
)
plt.axis('off')

plt.subplot2grid((3, 2), (2, 0))
odi["chocolate"].value_counts().sort_index().plot(
    kind='pie',
    # labels=["no", "yes", "unknown"],
    colors=["lavender", "lightblue", "teal", "wheat", "beige"],
    autopct="%.1f%%",
    title="Chocolate makes you ..."
)
plt.axis('off')

plt.subplot2grid((3, 2), (2, 1))
odi["stand_up"].value_counts().sort_index().plot(
    kind='pie',
    labels=["no", "yes", "unknown"],
    colors=["lightblue", "teal", "wheat"],
    autopct="%.1f%%",
    title="Did you stand up?"
)
plt.axis('off')

plt.show()


