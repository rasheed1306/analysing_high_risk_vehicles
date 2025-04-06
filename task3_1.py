import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

with open('a1-datasets/filtered_vehicle.csv', 'r') as file: 
    reader = csv.reader(file)
    # Skips Header row
    header = next(reader) 
    # Reads the rest of the rows
    rows = [row for row in reader]

# Avoid Truncation of DataFrame
pd.set_option('display.max_columns', None)

# Convert the list of rows into a DataFrame
df = pd.DataFrame(rows[:11], columns=header)

print(df.loc[df.index, ['VEHICLE_YEAR_MANUF', 'VEHICLE_BODY_STYLE', 'VEHICLE_MAKE']])

# Counting number of unique crashes for each year of vehicle manufacture, body style, and manufacturer
counts_manuf = df['VEHICLE_YEAR_MANUF'].value_counts()
counts_style = df['VEHICLE_BODY_STYLE'].value_counts()
counts_make = df['VEHICLE_MAKE'].value_counts()

print(counts_manuf)
print(counts_style)
print(counts_make)