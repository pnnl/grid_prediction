'''
Date created: Feb 18, 2020
Date updated: Mar 25, 2021
Author: Sheng Guan, working with PNNL on PowerDrone project
'''

import numpy as np
import matplotlib.pyplot as plt

def training_preparation(scenario_data, save_path,measure):


    Z_data = []
    count = 0
    for dataset in scenario_data:
        count += 1
        if count % 10 == 0:
            print('Done processing %d/%d datasets ...' % (count, len(scenario_data)))
        #change the shape of each scenario_data
        Z_data.append(np.reshape(dataset, (dataset.shape[0], dataset.shape[1])))

    print('[INFO]: Length of Z_data: ', len(Z_data))
    Z_flag = 0.0


    return Z_data,Z_flag
