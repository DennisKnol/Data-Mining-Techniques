import csv


def read_data(data):
    with open(data, 'rt') as f:
        data = [row for row in csv.reader(f.read().splitlines())]
        return print(data)


ODI = "ODI-2019-csv.csv"
read_data(ODI)
