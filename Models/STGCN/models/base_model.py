# @Author   : Sheng Guan
# @This project refers to the project:
# @Github   : https://github.com/VeritasYin/Project_Orion
'''
Date created: Feb 18, 2020
Date updated: Mar 25, 2021
Author: Sheng Guan, working with PNNL on PowerDrone project
'''

from models.layers import *
from os.path import join as pjoin
import tensorflow as tf
import tensorflow.contrib.slim as slim


def build_model(inputs, n_his, Ks, Kt, blocks, keep_prob):
    '''
    Build the base model.
    :param inputs: placeholder.
    :param n_his: int, size of historical records for training.
    :param Ks: int, kernel size of spatial convolution.
    :param Kt: int, kernel size of temporal convolution.
    :param blocks: list, channel configs of st_conv blocks.
    :param keep_prob: placeholder.
    '''
    x = inputs[:, 0:n_his, :, :]

    # Ko>0: kernel size of temporal convolution in the output layer.
    Ko = n_his
    # ST-Block
    for i, channels in enumerate(blocks):
        x = st_conv_block(x, Ks, Kt, channels, i, keep_prob, act_func='GLU')
        Ko -= 2 * (Ks - 1)

    # Output Layer
    if Ko > 1:
        y = output_layer(x, Ko, 'output_layer')
    else:
        raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".')

    tf.add_to_collection(name='copy_loss',
                         value=tf.nn.l2_loss(inputs[:, n_his - 1:n_his, :, :] - inputs[:, n_his:n_his + 1, :, :]))
    train_loss = tf.nn.l2_loss(y - inputs[:, n_his:n_his + 1, :, :])
    single_pred = y[:, 0, :, :]
    tf.add_to_collection(name='y_pred', value=single_pred)
    return train_loss, single_pred


def model_save(sess, global_steps, model_name, n_his, n_pred, model_feature,save_path='./output/models/'):
    '''
    Save the checkpoint of trained model.
    :param sess: tf.Session().
    :param global_steps: tensor, record the global step of training in epochs.
    :param model_name: str, the name of saved model.
    :param save_path: str, the path of saved model.
    :return:
    '''
    save_path = save_path+ str(n_his)+str(n_pred)+'/'
    saver = tf.train.Saver(max_to_keep=3)
    prefix_path = saver.save(sess, pjoin(save_path, model_name+str(model_feature)+str(n_his)+str(n_pred)), global_step=global_steps)
    print(f'<< Saving model to {prefix_path} ...')


def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
