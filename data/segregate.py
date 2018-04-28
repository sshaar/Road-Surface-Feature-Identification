import os
import csv

files = [f for f in os.listdir('.') if os.path.isfile(f)]

with open ('file_data.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for f in files:
        if (".csv" in f):
            writer.writerow([f] + ['0'])
