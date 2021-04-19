# ---------------------------------------------------------------
"""
------- Code description ---------
This helper functions python file consists of all the necessary classes and functions
Storing contents of the .mat file into a tensor according to PyTorch Geometric
https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
Date created: Feb 18, 2020
Date updated: Mar 5, 2020
Location: PNNL, Richland, WA, USA
---------------------------------------------------------------
-------- Tensor variable description -------
power system PMU data
data.x          --> 3-D tensor [num_bus, num_node_feats, num_time_steps]
sparse representation of the edges
data.edge_indx  --> 2-D tensor [2, num_edges]
dynamic edge properties such as current flows (magnitude and angle)
data.edge_attr  --> 3-D tensor [num_edges, num_edge_feats, num_time_steps]
graph level stability or any equivalent labeling
data.y_graph    --> 1-D tensor [1, num_time_steps]
node level labeling
data.y_node     --> 2-D tensor [num_bus, num_time_steps]
node level control inputs
data.u          --> 2-D tensor [num_bus, num_time_steps]
---------------------------------------------------------------
------- Variable description ---------
num_bus        -- number of buses in the network (or it can be interpreted as number of nodes in a graph)
num_node_feats -- number of features (states) associated with each node in the network
num_time_steps -- number of time steps the data is available
num_edges      -- number of edges in the (power) network
num_edge_feats -- number of features (states) associates with each edge in the network
"""
# ---------------------------------------------------------------

import scipy.io as sio
import os
import numpy as np
import pandas as pd


class CyberPhysicalDataset:
    def __init__(self, root, gen_locations):
        """ This class file considers the mat file data pointed by the user and parses it into respective tensors"""
        # If the current code and data files are in the same directory
        current_file_path = os.path.dirname(os.path.abspath(__file__))
        # If the current code and data files are not in the same directory
        # current_file_path         = 'Repositories/GridSTAGE/code'

        mat_file_path = current_file_path + root + 'PMUData.mat'
        mat_file_contents = sio.loadmat(mat_file_path)
        PMU_locations = mat_file_contents['fmeas_con'][:, 1].astype(int)
        frequency_data = mat_file_contents['PMU']['f'][0][0][:, PMU_locations - 1].T
        voltage_data = mat_file_contents['PMU']['Vm'][0][0][:, PMU_locations - 1].T
        angle_data = mat_file_contents['PMU']['Va'][0][0][:, PMU_locations - 1].T
        num_time_steps, num_bus = mat_file_contents['PMU']['f'][0][0].shape

        scenario_description_file = current_file_path + root + 'ScenarioDescription.csv'

        agc_signals_file = current_file_path + root + 'ACEData.mat'
        agc_data = sio.loadmat(agc_signals_file)

        edge_attr_Im = None
        edge_attr_Ia = None
        edge_attr_Id = None

        for i in range(num_bus):
            if type(edge_attr_Im) == np.ndarray:
                edge_attr_Im = np.hstack((edge_attr_Im, mat_file_contents['PMU']['Im'][0][0][0][i]))
                edge_attr_Ia = np.hstack((edge_attr_Ia, mat_file_contents['PMU']['Ia'][0][0][0][i]))
                edge_attr_Id = np.hstack((edge_attr_Id, mat_file_contents['PMU']['Id'][0][0][0][i]))
            else:
                edge_attr_Im = mat_file_contents['PMU']['Im'][0][0][0][i]
                edge_attr_Ia = mat_file_contents['PMU']['Ia'][0][0][0][i]
                edge_attr_Id = mat_file_contents['PMU']['Id'][0][0][0][i]

        self.x = np.zeros([frequency_data.shape[1], frequency_data.shape[0], 3], dtype='float')
        num_buses = frequency_data.shape[0]
        num_time_steps = frequency_data.shape[1]
        for i in range(num_time_steps):
            for j in range(num_buses):
                self.x[i, j, 0] = frequency_data[j, i]
                self.x[i, j, 1] = voltage_data[j, i]
                self.x[i, j, 2] = angle_data[j, i]
        # print('X.shape')
        # print(self.x.shape)
        # self.x                    = np.transpose(np.array((frequency_data, voltage_data)), [1, 0, 2])

        self.scen_desc = pd.read_csv(scenario_description_file)

        self.edge_attr_Im = edge_attr_Im.T
        self.edge_attr_Ia = edge_attr_Ia.T
        self.edge_attr_Id = edge_attr_Id.T

        self.edge_attr = np.transpose(np.array((self.edge_attr_Im, self.edge_attr_Ia)), [1, 0, 2])
        self.u = np.zeros((num_bus, num_time_steps))
        self.u[gen_locations, :] = agc_data['tg_sig'][:, np.arange(0, 2 * num_time_steps, 2)]


# ---------------------------------------------------------------

class TransientDatasetwithControl:
    def __init__(self, root, gen_locations):
        """ This class file considers the mat file data pointed by the user and parses it into respective tensors"""
        # If the current code and data files are in the same directory
        current_file_path = os.path.dirname(os.path.abspath(__file__))
        # If the current code and data files are not in the same directory
        # current_file_path         = 'Repositories/GridSTAGE/code'

        mat_file_path = current_file_path + root + 'PMUData.mat'
        mat_file_contents = sio.loadmat(mat_file_path)
        pmu_locations = mat_file_contents['fmeas_con'][:, 1].astype(int)

        num_time_steps, num_bus = mat_file_contents['PMU']['f'][0][0].shape
        scenario_description = pd.read_csv(current_file_path + root + 'ScenarioDescription.csv')
        load_changes_start_times = scenario_description['Start time(s) for load changes'][0].split()
        time1 = int(load_changes_start_times[0])
        time2 = int(load_changes_start_times[1])

        frequency_data = [mat_file_contents['PMU']['f'][0][0][time1 * 50 + 25: time2 * 50 - 25, pmu_locations - 1],
                          mat_file_contents['PMU']['f'][0][0][time2 * 50 + 25: num_time_steps, pmu_locations - 1]]
        voltage_data = [mat_file_contents['PMU']['Vm'][0][0][time1 * 50 + 25: time2 * 50 - 25, pmu_locations - 1],
                        mat_file_contents['PMU']['Vm'][0][0][time2 * 50 + 25: num_time_steps, pmu_locations - 1]]
        angle_data = [mat_file_contents['PMU']['Va'][0][0][time1 * 50 + 25: time2 * 50 - 25, pmu_locations - 1],
                      mat_file_contents['PMU']['Va'][0][0][time2 * 50 + 25: num_time_steps, pmu_locations - 1]]

        self.F  = frequency_data
        self.Vm = voltage_data
        self.Va = angle_data
        agc_signals_file = current_file_path + root + 'ACEData.mat'
        agc_data = sio.loadmat(agc_signals_file)

        edge_attr_im = None
        edge_attr_ia = None
        edge_attr_id = None

        for i in range(num_bus):
            if type(edge_attr_im) == np.ndarray:
                edge_attr_im = np.hstack((edge_attr_im, mat_file_contents['PMU']['Im'][0][0][0][i]))
                edge_attr_ia = np.hstack((edge_attr_ia, mat_file_contents['PMU']['Ia'][0][0][0][i]))
                edge_attr_id = np.hstack((edge_attr_id, mat_file_contents['PMU']['Id'][0][0][0][i]))
            else:
                edge_attr_im = mat_file_contents['PMU']['Im'][0][0][0][i]
                edge_attr_ia = mat_file_contents['PMU']['Ia'][0][0][0][i]
                edge_attr_id = mat_file_contents['PMU']['Id'][0][0][0][i]

        self.scen_desc = scenario_description

        self.edge_attr_Im = edge_attr_im.T
        self.edge_attr_Ia = edge_attr_ia.T
        self.edge_attr_Id = edge_attr_id.T

        self.edge_attr = np.transpose(np.array((self.edge_attr_Im, self.edge_attr_Ia)), [1, 0, 2])
        self.u = np.zeros((num_bus, num_time_steps))
        self.u[gen_locations, :] = agc_data['tg_sig'][:, np.arange(0, 2 * num_time_steps, 2)]

# ---------------------------------------------------------------

class TransientDataset:
    def __init__(self, root):
        """ This class file considers the mat file data pointed by the user and parses it into respective tensors"""
        # If the current code and data files are in the same directory
        # current_file_path = os.path.dirname(os.path.abspath(__file__))
        # If the current code and data files are not in the same directory
        # current_file_path         = 'Repositories/GridSTAGE/code'

        mat_file_path =  root + 'PMUData.mat'
        mat_file_contents = sio.loadmat(mat_file_path)
        pmu_locations = mat_file_contents['fmeas_con'][:, 1].astype(int)

        num_time_steps, num_bus = mat_file_contents['PMU']['f'][0][0].shape
        scenario_description = pd.read_csv(root + 'ScenarioDescription.csv')
        load_changes_start_times = scenario_description['Start time(s) for load changes'][0]

#        frequency_data = mat_file_contents['PMU']['f'][0][0][load_changes_start_times * 50 + 25: num_time_steps, pmu_locations - 1]
#        voltage_data = mat_file_contents['PMU']['Vm'][0][0][load_changes_start_times * 50 + 25: num_time_steps, pmu_locations - 1]
#        angle_data = mat_file_contents['PMU']['Va'][0][0][load_changes_start_times * 50 + 25: num_time_steps, pmu_locations - 1]

        frequency_data = mat_file_contents['PMU']['f'][0][0][load_changes_start_times * 50 + 50: num_time_steps, pmu_locations - 1]
        voltage_data = mat_file_contents['PMU']['Vm'][0][0][load_changes_start_times * 50 + 50: num_time_steps, pmu_locations - 1]
        angle_data = mat_file_contents['PMU']['Va'][0][0][load_changes_start_times * 50 + 50: num_time_steps, pmu_locations - 1]

        self.F  = frequency_data
        self.Vm = voltage_data
        self.Va = angle_data

        edge_attr_im = None
        edge_attr_ia = None
        edge_attr_id = None

        for i in range(num_bus):
            if type(edge_attr_im) == np.ndarray:
                edge_attr_im = np.hstack((edge_attr_im, mat_file_contents['PMU']['Im'][0][0][0][i]))
                edge_attr_ia = np.hstack((edge_attr_ia, mat_file_contents['PMU']['Ia'][0][0][0][i]))
                edge_attr_id = np.hstack((edge_attr_id, mat_file_contents['PMU']['Id'][0][0][0][i]))
            else:
                edge_attr_im = mat_file_contents['PMU']['Im'][0][0][0][i]
                edge_attr_ia = mat_file_contents['PMU']['Ia'][0][0][0][i]
                edge_attr_id = mat_file_contents['PMU']['Id'][0][0][0][i]

        self.scen_desc = scenario_description

        self.edge_attr_Im = edge_attr_im.T
        self.edge_attr_Ia = edge_attr_ia.T
        self.edge_attr_Id = edge_attr_id.T

        self.edge_attr = np.transpose(np.array((self.edge_attr_Im, self.edge_attr_Ia)), [1, 0, 2])

# ---------------------------------------------------------------

class SystemData:
    def __init__(self, root):
        """ This class file considers the system data file data pointed by the user and parses it into respective variables"""
        """ System data files are independent of scenario information """
        # If the current code and data files are in the same directory
        current_file_path = os.path.dirname(os.path.abspath(__file__))
        # If the current code and data files are not in the same directory
        # current_file_path         = '/Repositories/GridSTAGE/code'

        bus_line_data_file = current_file_path + root + 'BusLineData_68bus.mat'
        bus68_data = sio.loadmat(bus_line_data_file)
        line_data = bus68_data['line']
        bus_data = bus68_data['bus']
        num_edges = line_data.shape[0]
        num_buses = bus_data.shape[0]

        gen_locations = np.nonzero(bus_data[:, 3])

        self.gen_locations = gen_locations[0]
        self.num_bus = num_buses
        self.num_edges = num_edges
        self.edge_indx = line_data[:, 0:2].T
# ---------------------------------------------------------------
