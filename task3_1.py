import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DEBUG = True    

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


def task3_1(): 
    if DEBUG:
        print(df[['VEHICLE_YEAR_MANUF', 'VEHICLE_BODY_STYLE', 'VEHICLE_MAKE']])
    # Count number crashes for each car's unique year of manufacture, body style, and manufacturing company
    counts_manuf = df['VEHICLE_YEAR_MANUF'].value_counts()
    counts_style = df['VEHICLE_BODY_STYLE'].value_counts()
    counts_make = df['VEHICLE_MAKE'].value_counts()

    print(counts_manuf)
    print(counts_style)
    print(counts_make)

    # Graphically represent correlation between year of manufacture and unique vehicle types in crashes
    # Can identify how vehicles safety has improved over the years

    # Group unique vehicle types (body style and maker)
    types = zip(df['VEHICLE_BODY_STYLE'], df['VEHICLE_MAKE'])

    # Convert vehicle types to string and concatenate
    y = [type[0] + ' ' + type[1] for type in types]

    # Convert year of manufacture to int
    x = pd.to_numeric(df['VEHICLE_YEAR_MANUF'], errors='coerce').astype('Int64')

    plt.scatter(x, y)
    plt.xlabel('Year of Vehicle Manufacture ')
    plt.ylabel('Vehicle Types (Body Style and Maker)') 
    plt.title('Correlation between Year of Manufacture and Vehicle Types in Crashes')

    if DEBUG:
        plt.show()

    # Save the figure as a PNG file 
    plt.savefig('task3_1_scatter.png')

if __name__ == "__main__":
    task3_1()
