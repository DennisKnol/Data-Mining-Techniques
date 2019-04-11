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
odi["deserve_money"] = odi.iloc[:, 11]
odi["random_number"] = odi.iloc[:, 12]
odi["bedtimes"] = odi.iloc[:, 13]

# stress level
odi["stress_level"] = odi.iloc[:, 16]
odi["stress_level"] = odi["stress_level"].apply(pd.to_numeric, errors='coerce', downcast='float')
odi["stress_level"] = odi["stress_level"]

# neighbors
odi["number_of_neighbors"] = odi.iloc[:, 9]
odi["number_of_neighbors"] = odi["number_of_neighbors"].apply(pd.to_numeric, errors='coerce', downcast='integer')
odi["number_of_neighbors"] = odi["number_of_neighbors"].round(2)

