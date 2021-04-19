'''
Date created: Feb 18, 2020
Date updated: Mar 25, 2021
Author: Sheng Guan, working with PNNL on PowerDrone project
'''

import numpy as np
from memory_profiler import profile


class Dataset(object):
    def __init__(self, data, stats):
        self.__data = data
        self.mean = stats['mean']
        self.std = stats['std']

    def get_data(self, type):
        return self.__data[type]


    def get_stats(self):
        return {'mean': self.mean, 'std': self.std}


    def get_len(self, type):
        return len(self.__data[type])


    def z_inverse(self, type):
        return self.__data[type] * self.std + self.mean





def z_score(x, mean, std):
    '''
    Z-score normalization function: $z = (X - \mu) / \sigma $,
    where z is the z-score, X is the value of the element,
    $\mu$ is the population mean, and $\sigma$ is the standard deviation.
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score normalized array.
    '''
    return (x - mean) / std

    
@profile(precision=4)  
def second_data_gen_powerGrid(raw_feature, data_config, n_route, n_frame=51, p_start_t=75, n_his=50,val_pred_length=500,test_pred_length=500):
    n_train, n_val, n_test = data_config
    data_seq_train = np.zeros(raw_feature[0].shape[1])
    ###only keep the 0 to p_start_t + test_pred_legnth length data
    keep_length =  p_start_t + val_pred_length
    for i in range(len(raw_feature)-1):
        feats = raw_feature[i]
        feats = feats[0:keep_length,:]
        data_seq_train = np.vstack((data_seq_train, feats))
    data_seq_train = np.delete(data_seq_train, 0, axis=0)
    keep_test_length = p_start_t + test_pred_length
    data_seq_test = raw_feature[-1][0:keep_test_length]
    train_day_slot = p_start_t + val_pred_length
    test_day_slot = p_start_t + test_pred_length
    seq_train = seq_gen(n_train, data_seq_train, 0, n_frame, n_route, train_day_slot)
    seq_val = seq_gen(n_val, data_seq_train, n_train, n_frame, n_route, train_day_slot)
    seq_test = seq_gen(n_test, data_seq_test, 0, n_frame, n_route, test_day_slot)

    x_stats = {'mean': np.mean(seq_train), 'std': np.std(seq_train)}
    x_train = z_score(seq_train, x_stats['mean'], x_stats['std'])
    x_val = z_score(seq_val, x_stats['mean'], x_stats['std'])
    x_test = z_score(seq_test, x_stats['mean'], x_stats['std'])

    x_data = {'train': x_train, 'val': x_val, 'test': x_test}
    dataset = Dataset(x_data, x_stats)
    return dataset

def seq_gen(len_seq, data_seq, offset, n_frame, n_route, day_slot, C_0=1):
    '''
    Generate data in the form of standard sequence unit.
    :param len_seq: int, the length of target date sequence.
    :param data_seq: np.ndarray, source data / time-series.
    :param offset:  int, the starting index of different dataset type.
    :param n_frame: int, the number of frame within a standard sequence unit,
                         which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
    :param n_route: int, the number of routes in the graph.
    :param day_slot: int, the number of time slots per day, controlled by the time window (5 min as default).
    :param C_0: int, the size of input channel.
    :return: np.ndarray, [len_seq, n_frame, n_route, C_0].
    '''
    n_slot = day_slot - n_frame + 1

    tmp_seq = np.zeros((len_seq * n_slot, n_frame, n_route, C_0))
    for i in range(len_seq):
        for j in range(n_slot):
            sta = (i + offset) * day_slot + j
            end = sta + n_frame
            tmp_seq[i * n_slot + j, :, :, :] = np.reshape(data_seq[sta:end, :], [n_frame, n_route, C_0])
    return tmp_seq
    
def seq_test_gen(len_seq, data_seq, offset, n_frame, n_route,train_day_slot, day_slot, C_0=1):
    '''
    Generate data in the form of standard sequence unit.
    :param len_seq: int, the length of target date sequence.
    :param data_seq: np.ndarray, source data / time-series.
    :param offset:  int, the starting index of different dataset type.
    :param n_frame: int, the number of frame within a standard sequence unit,
                         which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
    :param n_route: int, the number of routes in the graph.
    :param day_slot: int, the number of time slots per day, controlled by the time window (5 min as default).
    :param C_0: int, the size of input channel.
    :return: np.ndarray, [len_seq, n_frame, n_route, C_0].
    '''
    n_slot = day_slot - n_frame + 1

    tmp_seq = np.zeros((len_seq * n_slot, n_frame, n_route, C_0))
    for i in range(len_seq):
        for j in range(n_slot):
            sta = (i + offset) * train_day_slot + j
            end = sta + n_frame
            tmp_seq[i * n_slot + j, :, :, :] = np.reshape(data_seq[sta:end, :], [n_frame, n_route, C_0])
    return tmp_seq
