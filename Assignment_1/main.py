import csv


ODI = "ODI-2019-csv.csv"


def read_data(data):
    with open(data, 'rt') as f:
        csv_reader = csv.reader(f)

        for row in csv_reader:
            print(row)
    return

df=read_data(ODI)

