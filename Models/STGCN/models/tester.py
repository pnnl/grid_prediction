# @Author   : Sheng Guan
# @FileName : tester.py
# @This project refers to:
# @Github   : https://github.com/VeritasYin/Project_Orion
'''
Date created: Feb 18, 2020
Date updated: Mar 25, 2021
Author: Sheng Guan, working with PNNL on PowerDrone project
'''


from data_loader.data_utils import gen_batch
from utils.math_utils import evaluation, power_grid_evaluation,transient_power_grid_evaluation
from os.path import join as pjoin
from models.base_model import build_model, model_save

import tensorflow as tf
import numpy as np
import time
from visualization.draw_actual_vs_prediction import *
from visualization.draw_all_buses_prediction import *
from visualization.draw_all_buses_prediction_accu import *
from visualization.draw_transient_actual_vs_prediction import *


def multi_pred_process(sess, y_pred, seq, batch_size, n_his, n_pred, step_idx, dynamic_batch=True):
    '''
       Multi_prediction function.
       :param sess: tf.Session().
       :param y_pred: placeholder.
       :param seq: np.ndarray, [len_seq, n_frame, n_route, C_0].
       :param batch_size: int, the size of batch.
       :param n_his: int, size of historical records for training.
       :param n_pred: int, the length of prediction.
       :param step_idx: int or list, index for prediction slice.
       :param dynamic_batch: bool, whether changes the batch size in the last one if its length is less than the default.
       :return y_ : tensor, 'sep' [len_inputs, n_route, 1]; 'merge' [step_idx, len_inputs, n_route, 1].
               len_ : int, the length of prediction.
       '''
    pred_list = []
    for i in gen_batch(seq, min(batch_size, len(seq)), dynamic_batch=dynamic_batch):
        # Note: use np.copy() to avoid the modification of source data.
        test_seq = np.copy(i[:, 0:n_his + 1, :, :])

        step_list = []
        ### for the prediction part, the logic has passed unit test!!! By Sheng 09/08/2020
        for j in range(n_pred):
            pred = sess.run(y_pred,
                            feed_dict={'data_input:0': test_seq, 'keep_prob:0': 1.0})
            if isinstance(pred, list):
                pred = np.array(pred[0])
            test_seq[:, 0:n_his - 1, :, :] = test_seq[:, 1:n_his, :, :]
            # make sure that using pred as the input to feed to the prediction
            test_seq[:, n_his - 1, :, :] = pred
            step_list.append(pred)
        pred_list.append(step_list)
    #  pred_array -> [n_pred, batch_size, n_route, C_0)
    pred_array = np.concatenate(pred_list, axis=1)
    return pred_array, pred_array.shape[1]


def multi_pred(sess, y_pred, seq, batch_size, n_his, n_pred, step_idx, dynamic_batch=True):
    '''
    Multi_prediction function.
    :param sess: tf.Session().
    :param y_pred: placeholder.
    :param seq: np.ndarray, [len_seq, n_frame, n_route, C_0].
    :param batch_size: int, the size of batch.
    :param n_his: int, size of historical records for training.
    :param n_pred: int, the length of prediction.
    :param step_idx: int or list, index for prediction slice.
    :param dynamic_batch: bool, whether changes the batch size in the last one if its length is less than the default.
    :return y_ : tensor, 'sep' [len_inputs, n_route, 1]; 'merge' [step_idx, len_inputs, n_route, 1].
            len_ : int, the length of prediction.
    '''
    pred_list = []
    for i in gen_batch(seq, min(batch_size, len(seq)), dynamic_batch=dynamic_batch):
        # Note: use np.copy() to avoid the modification of source data.
        test_seq = np.copy(i[:, 0:n_his+1, :, :])

        step_list = []
        for j in range(n_pred):
            pred = sess.run(y_pred,
                            feed_dict={'data_input:0': test_seq, 'keep_prob:0': 1.0})
            if isinstance(pred, list):
                pred = np.array(pred[0])
            test_seq[:, 0:n_his - 1, :, :] = test_seq[:, 1:n_his, :, :]
            #make sure that using pred as the input to feed to the prediction
            test_seq[:, n_his -1, :, :] = pred
            step_list.append(pred)
        pred_list.append(step_list)
    #  pred_array -> [n_pred, batch_size, n_route, C_0)
    pred_array = np.concatenate(pred_list, axis=1)
    return pred_array[step_idx], pred_array.shape[1]


def transient_model_inference(sess, pred, inputs, batch_size, n_his, n_pred, step_idx, min_va_val, min_val, p_start_t,x_mean,epoch,global_steps,model_feature,val_pred_length,test_pred_length,scenario_data,test_data):
    '''
    Model inference function.
    :param sess: tf.Session().
    :param pred: placeholder.
    :param inputs: instance of class Dataset, data source for inference.
    :param batch_size: int, the size of batch.
    :param n_his: int, the length of historical records for training.
    :param n_pred: int, the length of prediction.
    :param step_idx: int or list, index for prediction slice.
    :param min_va_val: np.ndarray, metric values on validation set.
    :param min_val: np.ndarray, metric values on test set.
    '''
    min_val_copy = min_val.copy()
    start_time = time.time()
    print("Validation evaluation starts!!!")
    x_val, x_test, x_stats = inputs.get_data('val'), inputs.get_data('test'), inputs.get_stats()
    
    if n_his+1 > x_val.shape[1]:
        raise ValueError(f'ERROR: the value of n_pred "{n_pred}" exceeds the length limit.')


    val_predicion_array = []
    val_rmse_array = []

    prediction_length = val_pred_length
    i_stop = int(prediction_length / n_pred)
    
    if i_stop != prediction_length:
        raise ValueError(f'ERROR: the value of n_pred "{n_pred}" is not equal to 1.')


    x_val_target = p_start_t - n_his
    x_val_process= x_val[x_val_target-1:x_val_target,:,:,:]
    batch_size =1
    #need at least n_his round to make sure that all the prediction now is updated
    temp_pred = n_his
    y_pred, len_pred = multi_pred_process(sess, pred, x_val_process, batch_size, n_his, temp_pred, step_idx)
    val_prediction = y_pred[:, 0, :, :]
    val_predicion_array.append(val_prediction)
    val_as_input_array = val_prediction.reshape(1, val_prediction.shape[0], val_prediction.shape[1],
                                                  val_prediction.shape[2])


    ###evaluate n_his size prediction and ground truthe next n_his size
    load_path = './output/models/'

    #continue to process the rest of the remaining prediction nodes
    i_remain = prediction_length - val_prediction.shape[0]
    x_val_dummy = val_as_input_array[:, n_his - 1:n_his, :, :]
    x_val_process = val_as_input_array
    x_val_process = np.concatenate((x_val_process, x_val_dummy), axis=1)
    
    for i in range(0, i_remain):
        y_pred, len_pred = multi_pred(sess, pred, x_val_process, batch_size, n_his, n_pred, step_idx)
        #update x_val_process
        x_val_process[:, 0:n_his - 1, :, :] = x_val_process[:, 1:n_his, :, :]
        x_val_process[:, n_his - 1, :, :] = y_pred
        val_prediction = y_pred[:, 0, :, :]
        val_predicion_array[0] = np.concatenate((val_predicion_array[0], val_prediction),axis=0)


    val_prediction_all = val_predicion_array[0]
    #write to get the val_ground_truth
    val_ground_truth_all_bak = scenario_data[p_start_t-1:p_start_t+val_pred_length-1,:]
    #print("verify the output")
    #print(val_ground_truth_all_bak.shape)
    #print(val_prediction_all.shape)

    scenario='valid'
    #draw figure
    #draw_transient_actual_vs_prediction(val_prediction_all, x_stats, p_start_t, load_path, x_mean, epoch,scenario, val_ground_truth_all_bak)





    print(f'Validation Inference Time {time.time() - start_time:.3f}s')

    evl_val = transient_power_grid_evaluation(val_prediction_all, x_stats,x_mean, val_ground_truth_all_bak)
    
    print("###########Validation Performance Metrics Values are #######################")
    print(f'MAPE validation {evl_val[0]:7.3%}')
    print(f'MAE  validation {evl_val[1]:14.8f}')
    print(f'RMSE validation {evl_val[2]:6.3f}')


    # chks: indicator that reflects the relationship of values between evl_val and min_va_val.
    chks = evl_val < min_va_val
    # update the metric on test set, if model's performance got improved on the validation.
    #Always draw testing figures for now
    if True:
        #print("current epoch number is %s"%(epoch))
        start_t_time = time.time()
        #print("testing evaluation starts!!!")
        min_va_val[chks] = evl_val[chks]
        ##starting the test part

        test_predicion_array = []
        test_rmse_array = []

        prediction_length = test_pred_length
        i_stop = int(prediction_length / n_pred)
        if i_stop != prediction_length:
            raise ValueError(f'ERROR: the value of n_pred "{n_pred}" is not equal to 1.')

        x_test_target = p_start_t - n_his
        x_test_process = x_test[x_test_target - 1:x_test_target, :, :, :]
        batch_size = 1
        # need at least n_his round to make sure that all the prediction now is updated
        temp_pred = n_his
        y_test_pred, len_test_pred = multi_pred_process(sess, pred, x_test_process, batch_size, n_his, temp_pred, step_idx)
        test_prediction = y_test_pred[:, 0, :, :]
        test_predicion_array.append(test_prediction)
        test_as_input_array = test_prediction.reshape(1, test_prediction.shape[0], test_prediction.shape[1],
                                                    test_prediction.shape[2])

        ###evaluate n_his size prediction and ground truthe next n_his size
        # draw the figures to show the difference between ground truth and prediction
        load_path = './output/models/'

        # continue to process the rest of the remaining prediction nodes
        i_remain = prediction_length - test_prediction.shape[0]
        x_test_dummy = test_as_input_array[:, n_his - 1:n_his, :, :]
        x_test_process = test_as_input_array
        x_test_process = np.concatenate((x_test_process, x_test_dummy), axis=1)
        
        for i in range(0, i_remain):
            y_test_pred, len_test_pred = multi_pred(sess, pred, x_test_process, batch_size, n_his, n_pred, step_idx)
            # update x_val_process
            x_test_process[:, 0:n_his - 1, :, :] = x_test_process[:, 1:n_his, :, :]
            x_test_process[:, n_his - 1, :, :] = y_test_pred
            test_prediction = y_test_pred[:, 0, :, :]
            test_predicion_array[0] = np.concatenate((test_predicion_array[0], test_prediction), axis=0)

        test_prediction_all = test_predicion_array[0]
        inference_time = time.time() - start_t_time
        #print(f'Testing Inference Time {inference_time:.3f}s')
        #save the testing inference time to a file
        with open(load_path+"inference_time.txt", mode='w') as file:
            file.write('Testing Inference Time %s seconds.\n' % 
               (inference_time))
        # write to get the test_ground_truth
        test_ground_truth_all_bak = test_data[p_start_t - 1:p_start_t + test_pred_length - 1, :]

        '''
        x_mean =1
        scenario='test_valid'
        draw_transient_actual_vs_prediction(test_prediction_all, x_stats, p_start_t, load_path, x_mean,
                                  epoch,scenario, test_ground_truth_all_bak)
        '''
        
        evl_pred = transient_power_grid_evaluation(test_prediction_all, x_stats,x_mean,test_ground_truth_all_bak)
        test_chks = evl_pred < min_val_copy
        #print("###########tesing prediction values is")
        #print(f'MAPE testing {evl_pred[0]:7.3%}')
        #print(f'MAE  testing {evl_pred[1]:14.8f}')
        #print(f'RMSE testing {evl_pred[2]:6.3f}')
        
        #save the model for each epoch
        print("saving the model")
        print('current epoch is %d'%epoch)
        model_save(sess, global_steps, 'STGCN', n_his, n_pred,model_feature)
        
 
        #using MAE as the indicator to save the model
        if sum(test_chks):
            #print("saving the model")
            #print('current epoch is %d'%epoch)
            #model_save(sess, global_steps, 'STGCN', n_his, n_pred,model_feature)
            min_val_copy[test_chks]= evl_pred[test_chks]
            
    return min_va_val, min_val_copy

    
def model_test_withGroundTruth(inputs, args, blocks, batch_size, n_his, n_pred, inf_mode, p_start_t, model_pred, scenario_data, \
test_pred_length, load_path='./output/models/'):
    '''
    Load and test saved model from the checkpoint.
    :param inputs: instance of class Dataset, data source for test.
    :param batch_size: int, the size of batch.
    :param n_his: int, the length of historical records for training.
    :param n_pred: int, the length of prediction.
    :param inf_mode: str, test mode - 'merge / multi-step test' or 'separate / single-step test'.
    :param load_path: str, the path of loaded model.
    '''
    load_path = load_path+ 'savedModel'+'/'
    start_time = time.time()
    model_path = tf.train.get_checkpoint_state(load_path).model_checkpoint_path

    test_graph = tf.Graph()

    with test_graph.as_default():
        saver = tf.train.import_meta_graph(pjoin(f'{model_path}'+'.meta'))

    with tf.Session(graph=test_graph) as test_sess:
        saver.restore(test_sess, tf.train.latest_checkpoint(load_path))
        print(f'>> Loading saved model from {model_path} ...')
        pred = test_graph.get_collection('y_pred')

        if inf_mode == 'sep':
            # for inference mode 'sep', the type of step index is int.
            step_idx = n_pred - 1
            tmp_idx = [step_idx]
        elif inf_mode == 'merge':
            # for inference mode 'merge', the type of step index is np.ndarray.
            step_idx = np.arange(1, n_pred + 1, 1) - 1
            tmp_idx = [n_pred - 1]
        else:
            raise ValueError(f'ERROR: test mode "{inf_mode}" is not defined.')

        x_test, x_stats = inputs.get_data('test'), inputs.get_stats()
        start_t_time = time.time()
        print("testing evaluation starts!!!")

        batch_size = 1
        
        test_ground_truth = scenario_data[2]

        load_path = './output/models/'

        test_predicion_array=[]
        rmse_array =[]
        
        prediction_length = test_pred_length
        i_stop = int(prediction_length/n_pred)
        i_remain = prediction_length%n_pred

        ''''/////////////////////////////////////////'''


        x_test_target = p_start_t - n_his
        x_test_process = x_test[x_test_target-1:x_test_target, :, :, :]
        y_pred, len_pred = multi_pred(test_sess, pred, x_test_process, batch_size, n_his, n_pred, step_idx)
        test_prediction = y_pred[:, 0, :, :]
        test_predicion_array.append(test_prediction)
        test_as_input_array = test_prediction.reshape(1,test_prediction.shape[0], test_prediction.shape[1],
                                         test_prediction.shape[2])
        for i in range(1, i_stop):
            x_test_dummy = test_as_input_array[:,n_pred-1:n_pred,:,:]
            x_test_process = test_as_input_array
            x_test_process = np.concatenate((x_test_process,x_test_dummy),axis=1)
            new_n_his = n_his -1

            y_pred, len_pred = multi_pred(test_sess, pred, x_test_process, batch_size, new_n_his, n_pred, step_idx)
            test_prediction = y_pred[:, 0, :, :]

            test_predicion_array.append(test_prediction)
            test_as_input_array = test_prediction.reshape(1, test_prediction.shape[0], test_prediction.shape[1],
                                                          test_prediction.shape[2])





        ####concatenate all the prediction and draw a complete figure
        test_prediction_array = test_predicion_array
        test_prediction = np.zeros((1,test_prediction_array[0].shape[1],1))
        for i in range(len(test_prediction_array)):
            test_prediction = np.concatenate((test_prediction,test_prediction_array[i]),axis=0)
        test_prediction = np.delete(test_prediction, 0, axis=0)
        print(test_prediction.shape)
        rmse = draw_all_buses_prediction_accu(test_ground_truth, test_prediction, x_stats, p_start_t, load_path)
        
        for ix in tmp_idx:
            print(f'Future Time Step {ix + 1}: '
                  f'RMSE testing {rmse:16.8f}.')
        print(f'Model Test Time {time.time() - start_time:.3f}s')
    print('Testing model finished!')



                        
def transient_model_test(inputs, args, blocks, batch_size, n_his, n_pred, inf_mode, p_start_t, model_pred, scenario_data, \
test_pred_length, test_scenario, test_data, load_path='./output/models/'):

    print("Starting the inference process")
    load_path = load_path + 'savedModel' + '/'
    start_time = time.time()
    model_path = tf.train.get_checkpoint_state(load_path).model_checkpoint_path
    test = tf.train.get_checkpoint_state(load_path)

    test_graph = tf.Graph()

    with test_graph.as_default():
        saver = tf.train.import_meta_graph(pjoin(f'{model_path}' + '.meta'))

    with tf.Session(graph=test_graph) as test_sess:
        saver.restore(test_sess, tf.train.latest_checkpoint(load_path))
        print(f'>> Loading saved model from {model_path} ...')
        pred = test_graph.get_collection('y_pred')

        if inf_mode == 'sep':
            # for inference mode 'sep', the type of step index is int.
            step_idx = n_pred - 1
            tmp_idx = [step_idx]
        elif inf_mode == 'merge':
            # for inference mode 'merge', the type of step index is np.ndarray.
            step_idx = np.arange(1, n_pred + 1, 1) - 1
            tmp_idx = [n_pred - 1]
        else:
            raise ValueError(f'ERROR: test mode "{inf_mode}" is not defined.')

        x_test, x_stats = inputs.get_data('test'), inputs.get_stats()
        start_t_time = time.time()
        print("testing evaluation starts!!!")


        test_predicion_array = []
        test_rmse_array = []

        prediction_length = test_pred_length
        i_stop = int(prediction_length / n_pred)
        if i_stop != prediction_length:
            raise ValueError(f'ERROR: the value of n_pred "{n_pred}" is not equal to 1.')

        x_test_target = p_start_t - n_his
        x_test_process = x_test[x_test_target - 1:x_test_target, :, :, :]
        batch_size = 1
        # need at least n_his round to make sure that all the prediction now is updated
        temp_pred = n_his
        y_pred, len_pred = multi_pred_process(test_sess, pred, x_test_process, batch_size, n_his, temp_pred, step_idx)
        test_prediction = y_pred[:, 0, :, :]
        test_predicion_array.append(test_prediction)
        test_as_input_array = test_prediction.reshape(1, test_prediction.shape[0], test_prediction.shape[1],
                                                    test_prediction.shape[2])

        ###evaluate n_his size prediction and ground truthe next n_his size
        # draw the figures to show the difference between ground truth and prediction
        load_path = './output/models/'
        

        # continue to process the rest of the remaining prediction nodes
        i_remain = prediction_length - test_prediction.shape[0]
        x_test_dummy = test_as_input_array[:, n_his - 1:n_his, :, :]
        x_test_process = test_as_input_array
        x_test_process = np.concatenate((x_test_process, x_test_dummy), axis=1)
        
        for i in range(0, i_remain):
            y_pred, len_pred = multi_pred(test_sess, pred, x_test_process, batch_size, n_his, n_pred, step_idx)
            # update x_val_process
            x_test_process[:, 0:n_his - 1, :, :] = x_test_process[:, 1:n_his, :, :]
            x_test_process[:, n_his - 1, :, :] = y_pred
            test_prediction = y_pred[:, 0, :, :]
            test_predicion_array[0] = np.concatenate((test_predicion_array[0], test_prediction), axis=0)


        test_prediction_all = test_predicion_array[0]
        print(f'Inference Time {time.time() - start_t_time:.3f}s')
        # write to get the test_ground_truth
        test_ground_truth_all_bak = test_data[p_start_t - 1:p_start_t + test_pred_length - 1, :]

        x_mean = 1
        epoch =1
        draw_transient_actual_vs_prediction(test_prediction_all, x_stats, p_start_t, load_path, x_mean,
                                  epoch,test_scenario, test_ground_truth_all_bak)