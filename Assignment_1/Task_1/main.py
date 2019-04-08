import csv


ODI = "/Users/DennisK/PycharmProjects/Data-Mining-Techniques/Assignment_1/Task_1/ODI-2019-csv.csv"


def read_data(data):
    with open(data, 'rt') as f:
        csv_reader = csv.reader(f)

        for row in csv_reader:
            print(row)
    return

df = read_data(ODI)

