# -*- coding: utf-8 -*-

# Author:
#    Antonio López Martínez-Carrasco <antoniolopezmc1995@gmail.com>

from pandas import DataFrame
from numpy import random, amin, amax
from sklearn.neighbors import NearestNeighbors

"""
The implementation of the Hopkins statistic is based on the following paper:
    1. Banerjee, A., Dave, R.: Validating clusters using the Hopkins statistic. In: IEEE International Conference on Fuzzy Systems (2004)
"""

def hopkins_statistic(pandas_dataframe, first_random_seed=1, second_random_seed=100):
    if type(pandas_dataframe) is not DataFrame:
        raise TypeError("Parameter 'pandas_dataframe' must be a pandas DataFrame.")
    # Datasets with 1 or 2 elements (i.e. less than 3).
    if len(pandas_dataframe) < 3:
        raise ValueError("Parameter 'pandas_dataframe' must have 3 rows or more.")
    # IMPORTANT: We use 'NearestNeighbors' (of 'scikit-learn') to obtain the nearest neighbor of an element.
    nearest_neighbor = NearestNeighbors()
    nearest_neighbor.fit(pandas_dataframe) # Fit with the complete dataset.
    # Number of rows in the set X (n) and number of rows in the set Y (m).
    n = len(pandas_dataframe)
    m = max(int(n*0.1), 1) # Variable 'm' cannot be 0. If the result of the multiplication is 0, then m=1.
    # Number of columns.
    nColumns = len(pandas_dataframe.columns)
    # Set Y (sampling placed at random in the d-dimensioned sampling window) and 'u' distances.
    sum_u_distances = 0
    random.seed(first_random_seed) # Set seed.
    for _ in range(m):
        # For each column of the random row, the value will be between the minimum and the maximum value of the corresponding column in 'pandas_dataframe'.
        random_row = random.uniform(amin(pandas_dataframe, axis=0), amax(pandas_dataframe, axis=0), nColumns).tolist()
        u_distance, _ = nearest_neighbor.kneighbors([random_row], n_neighbors = 1, return_distance=True)
        sum_u_distances = sum_u_distances + u_distance[0][0] # Distance to the nearest neighbor.
    # Second random sample (randomly selected elements in the set X) and 'w' distances.
    second_random_sample = pandas_dataframe.sample(m, random_state=second_random_seed)
    # VERY IMPORTANT: In this case, the nearest neighbor of an elements will be ALWAYS ITSELF (with distance=0). So, we have to use the second nearest neighbor.
    w_distances, _ = nearest_neighbor.kneighbors(second_random_sample, n_neighbors = 2, return_distance=True)
    sum_w_distances = 0
    for i in range(len(w_distances)):
        sum_w_distances = sum_w_distances + w_distances[i][1] # For each element in second_random_sample, the distance to the second nearest neighbor.
    # Compute H (Hopkins statistic).
    if (sum_u_distances + sum_w_distances) == 0: # Error case.
        raise ValueError("Division by zero (sum_u_distances + sum_w_distances is 0).")
    return sum_u_distances / (sum_u_distances + sum_w_distances)
