# @Author   : Sheng Guan
# @FileName : math_utils.py
# @This project refers to:
# @Github   : https://github.com/VeritasYin/Project_Orion
'''
Date created: Feb 18, 2020
Date updated: Mar 25, 2021
Author: Sheng Guan, working with PNNL on PowerDrone project
'''

import numpy as np


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


def z_inverse(x, mean, std):
    '''
    The inverse of function z_score().
    :param x: np.ndarray, input to be recovered.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score inverse array.
    '''
    return x * std + mean


def MAPE(v, v_):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAPE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v) / v)*100


def RMSE(v, v_):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, RMSE averages on all elements of input.
    '''
    return np.sqrt(np.mean((v_ - v) ** 2))


def MAE(v, v_):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v))


def evaluation(y, y_, x_stats):
    '''
    Evaluation function: interface to calculate MAPE, MAE and RMSE between ground truth and prediction.
    Extended version: multi-step prediction can be calculated by self-calling.
    :param y: np.ndarray or int, ground truth.
    :param y_: np.ndarray or int, prediction.
    :param x_stats: dict, paras of z-scores (mean & std).
    :return: np.ndarray, averaged metric values.
    '''
    dim = len(y_.shape)

    if dim == 3:
        # single_step case
        v = z_inverse(y, x_stats['mean'], x_stats['std'])
        v_ = z_inverse(y_, x_stats['mean'], x_stats['std'])
        return np.array([MAPE(v, v_), MAE(v, v_), RMSE(v, v_)])
    else:
        # multi_step case
        tmp_list = []
        # y -> [time_step, batch_size, n_route, 1]
        y = np.swapaxes(y, 0, 1)
        # recursively call
        for i in range(y_.shape[0]):
            tmp_res = evaluation(y[i], y_[i], x_stats)
            tmp_list.append(tmp_res)
        #for tmp_list. futher compute the average score of MAPE, MAE,RMSE
        #added by Sheng
        score_list = np.zeros((3,1))
        mape_list =[]
        mae_list = []
        rmse_list =[]
        for score in tmp_list:
            mape_list.append(score[0])
            mae_list.append(score[1])
            rmse_list.append(score[2])
        mape_mean = np.mean(mape_list)
        mae_mean = np.mean(mae_list)
        rmse_mean = np.mean(rmse_list)
        score_list[0][0] =  mape_mean
        score_list[1][0] =   mae_mean
        score_list[2][0] = rmse_mean
        score_list=score_list.squeeze()
        return score_list

def power_grid_evaluation(y, y_, x_stats,x_stat):
    '''
    Evaluation function: interface to calculate MAPE, MAE and RMSE between ground truth and prediction.
    Extended version: multi-step prediction can be calculated by self-calling.
    :param y: np.ndarray or int, ground truth.
    :param y_: np.ndarray or int, prediction.
    :param x_stats: dict, paras of z-scores (mean & std).
    :return: np.ndarray, averaged metric values.
    '''
    dim = len(y_.shape)

    # single_step case
    v = z_inverse(y, x_stats['mean'], x_stats['std'])
    x_mean = x_stat
    x_mean = np.tile(x_mean,(v.shape[0],1))

    x_mean = x_mean.reshape((x_mean.shape[0],x_mean.shape[1],1))

    v_ = z_inverse(y_, x_stats['mean'], x_stats['std'])


    return np.array([MAPE(v, v_), MAE(v, v_), RMSE(v, v_)])
    
def transient_power_grid_evaluation(y_, x_stats,x_stat, val_ground_truth_all):
    '''
    Evaluation function: interface to calculate MAPE, MAE and RMSE between ground truth and prediction.
    Extended version: multi-step prediction can be calculated by self-calling.
    :param y: np.ndarray or int, ground truth.
    :param y_: np.ndarray or int, prediction.
    :param x_stats: dict, paras of z-scores (mean & std).
    :return: np.ndarray, averaged metric values.
    '''
    v = val_ground_truth_all

    v_ = z_inverse(y_, x_stats['mean'], x_stats['std'])
    v_ = v_[:,:,0]


    return np.array([MAPE(v, v_), MAE(v, v_), RMSE(v, v_)])


