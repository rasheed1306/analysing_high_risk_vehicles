import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

with open('a1-datasets/filtered_vehicle_test.csv', 'r') as file: 
    reader = csv.reader(file)
    # Skips Header row
    header = next(reader) 
    # Reads the rest of the rows
    rows = [row for row in reader]

# Avoid Truncation of DataFrame
pd.set_option('display.max_columns', None)

# Convert the list of rows into a DataFrame
df = pd.DataFrame(rows, columns=header)
print(df[['VEHICLE_YEAR_MANUF', 'VEHICLE_BODY_STYLE', 'VEHICLE_MAKE']])

# Counting number of unique crashes for each year of vehicle manufacture
# df['VEHICLE_YEAR_MANUF'] = df['VEHICLE_YEAR_MANUF'].value_counts()
