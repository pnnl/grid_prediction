'''
Date created: Feb 18, 2020
Date updated: Mar 25, 2021
Author: Sheng Guan, working with PNNL on PowerDrone project
'''

import numpy as np
import csv

def get_adjacency_matrix(distance_df_filename, num_of_vertices):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''

    with open(distance_df_filename, 'r') as f:
        reader = csv.reader(f)
        header = f.__next__()
        edges = [(int(i[0]), int(i[1])) for i in reader]

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1

    return A