# @Author: Sheng Guan
# This project refers to the project
# @Github   : https://github.com/VeritasYin/Project_Orion

'''
Date created: Feb 18, 2020
Date updated: Mar 25, 2021
Author: Sheng Guan, working with PNNL on PowerDrone project
'''

from data_loader.data_utils import gen_batch
from models.tester import transient_model_inference
from models.base_model import build_model, model_save
from os.path import join as pjoin

import tensorflow as tf
import numpy as np
import time


    
def transient_model_train(inputs, blocks, args, x_mean, scenario_data, test_data, sum_path='./output/tensorboard'):
    '''
    Train the base model.
    :param inputs: instance of class Dataset, data source for training.
    :param blocks: list, channel configs of st_conv blocks.
    :param args: instance of class argparse, args for training.
    '''
    n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
    Ks, Kt = args.ks, args.kt
    model_feature = args.feature
    batch_size, epoch, inf_mode, opt = args.batch_size, args.epoch, args.inf_mode, args.opt
    prediction_start_time = args.p_start_t
    val_pred_length = args.val_pred_length
    test_pred_length = args.test_pred_length

    # Placeholder for model training
    # placeholder data_input the size dimension is fixed
    x = tf.placeholder(tf.float32, [None, n_his+1, n, 1], name='data_input')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # Define model loss
    train_loss, pred = build_model(x, n_his, Ks, Kt, blocks, keep_prob)
    tf.summary.scalar('train_loss', train_loss)
    copy_loss = tf.add_n(tf.get_collection('copy_loss'))
    tf.summary.scalar('copy_loss', copy_loss)

    # Learning rate settings
    global_steps = tf.Variable(0, trainable=False)
    len_train = inputs.get_len('train')
    if len_train % batch_size == 0:
        epoch_step = len_train / batch_size
    else:
        epoch_step = int(len_train / batch_size) + 1
    # Learning rate decay with rate 0.7 every 5 epochs.
    lr = tf.train.exponential_decay(args.lr, global_steps, decay_steps=5 * epoch_step, decay_rate=0.7, staircase=True)
    tf.summary.scalar('learning_rate', lr)
    step_op = tf.assign_add(global_steps, 1)
    with tf.control_dependencies([step_op]):
        if opt == 'RMSProp':
            train_op = tf.train.RMSPropOptimizer(lr).minimize(train_loss)
        elif opt == 'ADAM':
            train_op = tf.train.AdamOptimizer(lr).minimize(train_loss)
        else:
            raise ValueError(f'ERROR: optimizer "{opt}" is not defined.')

    merged = tf.summary.merge_all()
    
    '''
    print("Get the training model parameters")
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(shape)
        print(len(shape))
        variable_parameters = 1
        for dim in shape:
            print(dim)
            variable_parameters *= dim.value
        print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)
    total_comp =np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print("comparison")
    print(total_comp)
    '''

    with tf.Session() as sess:
        
        ##need to recomment if you normally train the model
        writer = tf.summary.FileWriter(pjoin(sum_path, 'train'), sess.graph)
        sess.run(tf.global_variables_initializer())
        
        ###add tbe segment that loads a model and resumes training
        '''
        load_path = './output/models/'
        load_path = load_path+ str(n_his)+str(1)+'/'
        model_path = tf.train.get_checkpoint_state(load_path).model_checkpoint_path
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(load_path))
        print(f'>> Loading saved model from {model_path} ...')
        graph = tf.get_default_graph()
        writer = tf.summary.FileWriter(pjoin(sum_path, 'train'), sess.graph)
        '''
        

        if inf_mode == 'sep':
            # for inference mode 'sep', the type of step index is int.
            step_idx = n_pred - 1
            tmp_idx = [step_idx]
            min_val = min_va_val = np.array([4e1, 1e5, 1e5])
        elif inf_mode == 'merge':
            # for inference mode 'merge', the type of step index is np.ndarray.
            step_idx = tmp_idx = np.arange(1, n_pred + 1, 1) - 1
            tmp_idx =  [n_pred - 1]
            min_val = min_va_val = np.array([4e1, 1e5, 1e5])


        else:
            raise ValueError(f'ERROR: test mode "{inf_mode}" is not defined.')

        train_time_starter = time.time()
        for i in range(epoch):
            start_time = time.time()
            for j, x_batch in enumerate(
                    gen_batch(inputs.get_data('train'), batch_size, dynamic_batch=True, shuffle=True)):
                summary, _ = sess.run([merged, train_op], feed_dict={x: x_batch[:, 0:n_his+1, :, :], keep_prob: 1.0})
                writer.add_summary(summary, i * epoch_step + j)
                if j % 50 == 0:
                    loss_value = \
                        sess.run([train_loss, copy_loss],
                                 feed_dict={x: x_batch[:, 0:n_his + 1, :, :], keep_prob: 1.0})
                    print(f'Epoch {i:2d}, Step {j:3d}: [Train_loss {loss_value[0]:.3f}]')
            print(f'Epoch {i:2d} Training Time {time.time() - start_time:.3f}s')

            min_va_val, min_val = \
                transient_model_inference(sess, pred, inputs, batch_size, n_his, n_pred, step_idx, min_va_val, min_val, prediction_start_time,x_mean,i,global_steps,model_feature,val_pred_length,test_pred_length,scenario_data,test_data)
            
            for ix in tmp_idx:
                va, te = min_va_val, min_val
                '''
                print(f'Future Time Step {ix + 1}: '
                      f'MAPE validation {va[0]:7.3%}, testing {te[0]:7.3%}; '
                      f'MAE  validation {va[1]:14.8f}, testing {te[1]:14.8f}; '
                      f'RMSE validation {va[2]:16.8f}, testing {te[2]:16.8f}.')
                '''

        writer.close()
        total_train_time = time.time()- train_time_starter
        load_path = './output/models/'
        with open(load_path+"training_time.txt", mode='w') as file:
            file.write('Training Time %s seconds.\n' % 
               (total_train_time))
        #print("Maximum memory usage")
        #memory_usage=sess.run(tf.contrib.memory_stats.MaxBytesInUse())
        #print(memory_usage)
    print('Training model finished!')
