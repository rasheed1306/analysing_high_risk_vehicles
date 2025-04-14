from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from task3_1 import df
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DEBUG = True

# Truncating values to 2 decimal places 
pd.set_option('display.float_format', '{:.2f}'.format)

# Grouping vehicles by unique Year of Vehicle Manufacture, Body Style and Manufacturer combination
if DEBUG:
    print(df[['VEHICLE_YEAR_MANUF', 'VEHICLE_BODY_STYLE', 'VEHICLE_MAKE']])

# Changing Vehicle Year of Manufacture and Features to int
df['VEHICLE_YEAR_MANUF'] = pd.to_numeric(df['VEHICLE_YEAR_MANUF'], errors='coerce')
df['NO_OF_CYLINDERS'] = pd.to_numeric(df['NO_OF_CYLINDERS'], errors='coerce')
df['NO_OF_WHEELS'] = pd.to_numeric(df['NO_OF_WHEELS'], errors='coerce')
df['SEATING_CAPACITY'] = pd.to_numeric(df['SEATING_CAPACITY'], errors='coerce')
df['TARE_WEIGHT'] = pd.to_numeric(df['TARE_WEIGHT'], errors='coerce')
df['TOTAL_NO_OCCUPANTS'] = pd.to_numeric(df['TOTAL_NO_OCCUPANTS'], errors='coerce')

unique_vehicles = df.groupby(['VEHICLE_YEAR_MANUF', 'VEHICLE_BODY_STYLE', 'VEHICLE_MAKE'])


# Printing aggregrate statistics for each group 

features = ['NO_OF_CYLINDERS', 'NO_OF_WHEELS', 'SEATING_CAPACITY', 'TARE_WEIGHT', 'TOTAL_NO_OCCUPANTS']

unique_vehicles_agg = unique_vehicles[features].mean()

if DEBUG:
    print("Grouped Year of Manufacture, Body Style and Manufacturer Aggregated Data")
    print(unique_vehicles_agg)

# Max min normalising data 
scaler = MinMaxScaler()
unique_vehicles_norm = scaler.fit_transform(unique_vehicles_agg)

if DEBUG:
    # Printing normalised data
    print("Grouped Year of Manufacture Normalised Data")
    print(np.round(unique_vehicles_norm, 2))


# Performing KMeans clustering on the normalised data
distortions = []
k_range = range(1, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(unique_vehicles_norm)
    distortions.append(kmeans.inertia_)

plt.plot(k_range, distortions, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Distortion')
plt.grid()

if DEBUG:
    plt.show()

# The elbow method suggests that the optimal number of clusters is at k = 5

# Save the figure as a PNG file
plt.savefig('task3_2_elbow.png')
