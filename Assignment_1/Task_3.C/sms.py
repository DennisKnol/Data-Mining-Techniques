import pandas as pd
import csv


sms_data = []
with open("SmsCollection.csv", 'r') as file:
    reader = csv.reader(file, delimiter=';')
    for row in reader:
        sms_data.append(row)

sms_data = sms_data[:-1]





