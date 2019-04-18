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
print(odi.info())

corr_mat = odi.corr().round(4)

plt.figure(figsize=(15, 8))
odi["programme"].value_counts().plot(
    kind='bar',
    title='What programme are you in?'
)
plt.ylabel("Count")
plt.show()

print(round(odi["gender"].value_counts(normalize=True), 2))

# Pie plots for multiple choice answers
plt.figure(figsize=(18, 8))

plt.subplot2grid((2, 4), (0, 0))
odi["machine_learning"].value_counts().sort_index().plot(
    kind='pie',
    labels=["no", "yes", "unknown"],
    colors=["lightblue", "teal", "wheat"],
    autopct="%.1f%%",
    title="Have you taken a course on machine learning?"
    )
plt.axis('off')

plt.subplot2grid((2, 4), (0, 1))
odi["information_retrieval"].value_counts().sort_index().plot(
    kind='pie',
    labels=["no", "yes", "unknown"],
    colors=["lightblue", "teal", "wheat"],
    autopct="%.1f%%",
    title="Have you taken a course on information retrieval?"
    )
plt.axis('off')

plt.subplot2grid((2, 4), (0, 2))
odi["statistics"].value_counts().sort_index().plot(
    kind='pie',
    labels=["no", "yes", "unknown"],
    colors=["lightblue", "teal", "wheat"],
    autopct="%.1f%%",
    title="Have you taken a course on statistics?"
    )
plt.axis('off')

plt.subplot2grid((2, 4), (1, 0))
odi["databases"].value_counts().sort_index().plot(
    kind='pie',
    labels=["no", "yes", "unknown"],
    colors=["lightblue", "teal", "wheat"],
    autopct="%.1f%%",
    title="Have you taken a course on databases?"
)
plt.axis('off')

plt.subplot2grid((2, 4), (1, 1))
odi["stand_up"].value_counts().sort_index().plot(
    kind='pie',
    labels=["no", "yes", "unknown"],
    colors=["lightblue", "teal", "wheat"],
    autopct="%.1f%%",
    title="Did you stand up?"
)
plt.axis('off')

plt.subplot2grid((2, 4), (1, 2))
odi["gender"].value_counts().sort_index().plot(
    kind='pie',
    labels=["male", "female", "unknown"],
    colors=["lightblue", "pink", "beige"],
    autopct="%.1f%%",
    title="What is your gender"
)
plt.axis('off')

plt.subplot2grid((2, 4), (0, 3))
odi["chocolate"][odi["gender"] == 0].value_counts().sort_index().plot(
    kind='pie',
    labels=["fat", "slim", "neither", "I have no idea what you are talking about", "unknown"],
    colors=["lavender", "lightblue", "teal", "wheat", "beige"],
    autopct="%.1f%%",
    title="Chocolate makes you ... (Male)"
)
plt.axis('off')

plt.subplot2grid((2, 4), (1, 3))
odi["chocolate"][odi["gender"] == 1].value_counts().sort_index().plot(
    kind='pie',
    labels=["fat", "slim", "neither", "I have no idea what you are talking about", "unknown"],
    colors=["lavender", "lightblue", "teal", "wheat", "beige"],
    autopct="%.1f%%",
    title="Chocolate makes you ... (Female)"
)
plt.axis('off')

plt.show()
#
# # room capacity is 343, meaning that the max numbers is 342
# # https://www.vu.nl/nl/Images/Zaalfaciliteiten_aug2018_tcm289-257362.pdf
odi["number_of_neighbors"] = odi["number_of_neighbors"].dropna()
odi["number_of_neighbors"] = odi["number_of_neighbors"].drop(
    odi["number_of_neighbors"][odi["number_of_neighbors"] > 342].index
)

# random number was supposed to be between 0 and 10
odi["random_number"] = odi["random_number"].dropna()
odi["random_number"] = odi["random_number"].drop(
    odi["random_number"][odi["random_number"] > 10].index
)

# 100 euro is the limit
odi["deserve_money"] = odi["deserve_money"].dropna()
odi["deserve_money"] = odi["deserve_money"].drop(
    odi["deserve_money"][odi["deserve_money"] > 100].index
)

# Density plot and bar plots
plt.figure(figsize=(18, 8))

plt.subplot2grid((2, 4), (0, 0))
sns.distplot(odi["number_of_neighbors"], bins=100)
plt.title("Density plot of the number of neighbors")

plt.subplot2grid((2, 4), (1, 0))
sns.barplot(x="gender", y="number_of_neighbors", data=odi, alpha=0.8, palette=['lightblue', 'pink', 'beige'])
plt.title("Mean number of neighbors per Gender")
plt.ylabel("Mean number of neighbors")

plt.subplot2grid((2, 4), (0, 1))
sns.distplot(odi["random_number"], bins=100)
plt.title("Density plot of random number from 0 to 10")

plt.subplot2grid((2, 4), (1, 1))
sns.barplot(x="gender", y="random_number", data=odi, alpha=0.8, palette=['lightblue', 'pink', 'beige'])
plt.title("Mean number of neighbors per Gender")
plt.ylabel("Mean number of neighbors")

plt.subplot2grid((2, 4), (0, 2))
sns.distplot(odi["deserve_money"], bins=100)
plt.title("Density plot of the amount of money student think they deserve")

plt.subplot2grid((2, 4), (1, 2))
sns.barplot(x="gender", y="deserve_money", data=odi, alpha=0.8, palette=['lightblue', 'pink', 'beige'])
plt.title("Mean amount of money per Gender")
plt.ylabel("Mean amount of money")

plt.subplot2grid((2, 4), (0, 3))
sns.distplot(odi["stress_level"], bins=100)
plt.title("Density plot of the level of stress")

plt.subplot2grid((2, 4), (1, 3))
sns.barplot(x="gender", y="stress_level", data=odi, alpha=0.8, palette=['lightblue', 'pink', 'beige'])
plt.title("Mean level of stress per Gender")
plt.ylabel("Mean stress level")

plt.show()


df_good_day_1 = odi[['gender', 'good_day_1']].copy()
df_good_day_1.columns = ['gender', 'good_day']

df_good_day_2 = odi[['gender', 'good_day_2']].copy()
df_good_day_2.columns = ['gender', 'good_day']

df_good_day = df_good_day_1.append(df_good_day_2, ignore_index=True)


plt.figure(figsize=(18, 8))

plt.subplot2grid((2, 3), (0, 0))
df_good_day["gender"][df_good_day["good_day"] == "Sex"].value_counts(normalize=True)\
    .sort_index()\
    .plot(
    kind="bar",
    alpha=0.8,
    color=['lightblue', 'pink', 'beige']
)
plt.title("'Sex' makes a good day, per gender ")

plt.subplot2grid((2, 3), (0, 1))
df_good_day["gender"][df_good_day["good_day"] == "Food"].value_counts(normalize=True)\
    .sort_index()\
    .plot(
    kind="bar",
    alpha=0.8,
    color=['lightblue', 'pink', 'beige']
)
plt.title("'Food' makes a good day, per gender ")

plt.subplot2grid((2, 3), (0, 2))
df_good_day["gender"][df_good_day["good_day"] == "Nice weather"].value_counts(normalize=True)\
    .sort_index()\
    .plot(
    kind="bar",
    alpha=0.8,
    color=['lightblue', 'pink', 'beige']
)
plt.title("'Nice weather' makes a good day, per gender ")

plt.subplot2grid((2, 3), (1, 0))
df_good_day["gender"][df_good_day["good_day"] == "Alcohol"].value_counts(normalize=True)\
    .sort_index()\
    .plot(
    kind="bar",
    alpha=0.8,
    color=['lightblue', 'pink', 'beige']
)
plt.title("'Alcohol' makes a good day, per gender ")

plt.subplot2grid((2, 3), (1, 1))
df_good_day["gender"][df_good_day["good_day"] == "Friends"].value_counts(normalize=True)\
    .sort_index()\
    .plot(
    kind="bar",
    alpha=0.8,
    color=['lightblue', 'pink', 'beige']
)
plt.title("'Friends' makes a good day , per gender ")

plt.subplot2grid((2, 3), (1, 2))
df_good_day["gender"][df_good_day["good_day"] == "Sports"].value_counts(normalize=True)\
    .sort_index()\
    .plot(
    kind="bar",
    alpha=0.8,
    color=['lightblue', 'pink', 'beige']
)
plt.title("'Sports' makes a good day , per gender ")

plt.show()

categories = ["Sex", "Food", "Nice weather", "Alcohol", "Friends", "Sports", "Sleep"]
df_good_day_cat = df_good_day[df_good_day["good_day"].isin(categories)]
df_good_day = df_good_day[df_good_day.gender < 2]

plt.figure(figsize=(18, 6))
sns.barplot(x='good_day', y='gender', data=df_good_day_cat, alpha=0.8)
plt.show()