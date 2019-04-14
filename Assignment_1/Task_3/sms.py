import pandas as pd
import csv


with open("SmsCollection.csv", 'r') as file:
    reader = csv.reader(file, delimiter=';')
    for row in reader:
        print(row)


