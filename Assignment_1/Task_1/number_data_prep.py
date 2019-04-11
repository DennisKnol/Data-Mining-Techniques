import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

odi = pd.read_csv(
    "/Users/DennisK/PycharmProjects/Data-Mining-Techniques/Assignment_1/Task_1/odi-2019-csv.csv",
    sep=';'
)

# Split timestamp into day and time
odi[["day", "time"]] = odi['Timestamp'].str.split(" ", expand=True)

# Rename
odi["birthday"] = odi.iloc[:, 8]
odi["number_of_neighbors"] = odi.iloc[:, 9]
odi["deserve_money"] = odi.iloc[:, 11]
odi["random_number"] = odi.iloc[:, 12]
odi["bedtimes"] = odi.iloc[:, 13]




# clean birthdays
# print
#
# odi["clean_birthday"] = odi["birthday"].apply(lambda x: pd.to_datetime(x).strftime('%m/%d/%Y')[0])
# print(odi["clean_birthday"])

# clean stress level

odi["stress_level"].apply(lambda x: x.str.replace(',','.'))
odi["stress_level"] = odi.iloc[:, 16]
odi["stress_level"] = odi["stress_level"].apply(pd.to_numeric, errors='coerce', downcast='integer')


#
# odi["clean_stress_level"] = odi[~odi["stress_level"].isin([0, 100])]
# print(odi["clean_stress_level"])
# #
# #


odi = odi.drop(columns=['Timestamp'])
