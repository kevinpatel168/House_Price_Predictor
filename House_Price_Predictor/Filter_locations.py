import pandas as pd
import numpy as np
from array import *

df5 = pd.read_csv("Bengaluru_House_Data.csv")

location_stats = df5['location'].value_counts(ascending=False)
# print(location_stats.values.sum())

location_stats_less_than_10 = location_stats[location_stats<=10]
# print(len(location_stats_less_than_10))
# print(len(df5.location.unique()))

df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
x = df5.location.unique()
# y = final_locations.sort()
# print(y) 
x.tolist()
y = [str(item) for item in x ]
final_locations = sorted(y)

# y = final_locations.sort_values(by=['location'], ascending = True)
# print(y)
