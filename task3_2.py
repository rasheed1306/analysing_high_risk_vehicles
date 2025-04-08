from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from task3_1 import df

print(df.loc[df.index, ['VEHICLE_YEAR_MANUF', 'VEHICLE_BODY_STYLE', 'VEHICLE_MAKE']])

