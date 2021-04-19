import scipy.io as sio
import tensorflow as tf
import numpy as np
import math
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

"""
Date created:  December 10, 2020
Date Modified: December 10, 2020
---- Inputs to this code ----
current_data              -- vector [#states - 136]
num_time_steps prediction -- positive integer

---- Content of this code ----
Takes the current state space data, transforms to latent space,
forwards num_time_steps with states inclusive

---- Outputs of this code ----
predicted_data -- vector [#states x #time - 136xnum_time_steps]

---- Assumptions ----
The input data to this code is normalized and the output data
of this code is normalized as well
"""

dir_name = 'Models'
mat_file_contents = sio.loadmat(dir_name+'/Robust_deepDMD.mat')
K = mat_file_contents['K']
encoder_weights = mat_file_contents['Weights'][0]

encoder_weights[1] = np.reshape(encoder_weights[1],(encoder_weights[1].shape[1],))
encoder_weights[3] = np.reshape(encoder_weights[3],(encoder_weights[3].shape[1],))
encoder_weights[5] = np.reshape(encoder_weights[5],(encoder_weights[5].shape[1],))
encoder_weights[7] = np.reshape(encoder_weights[7],(encoder_weights[7].shape[1],))
encoder_weights[9] = np.reshape(encoder_weights[9],(encoder_weights[9].shape[1],))

class DenseLayer(layers.Layer):

    def __init__(self, initial_weights, initial_bias):
        super(DenseLayer, self).__init__(dtype = 'float64')
        self.w = tf.Variable(initial_value=tf.convert_to_tensor(initial_weights, dtype=tf.float64),trainable=False)
        self.b = tf.Variable(initial_value=tf.convert_to_tensor(initial_bias, dtype=tf.float64),trainable=False)

    def call(self, inputs):
        x = tf.matmul(inputs, self.w) + self.b
        return tf.nn.elu(x)

class LinearLayer(layers.Layer):

    def __init__(self, initial_weights, initial_bias):
        super(LinearLayer, self).__init__(dtype = 'float64')
        self.w = tf.Variable(initial_value=tf.convert_to_tensor(initial_weights, dtype=tf.float64),trainable=False)
        self.b = tf.Variable(initial_value=tf.convert_to_tensor(initial_bias, dtype=tf.float64),trainable=False)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


# Neural Network
class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Encoder, self).__init__(dtype = 'float64', name = 'Encoder')
        self.input_layer   = DenseLayer(encoder_weights[0], encoder_weights[1])
        self.hidden_layer1 = DenseLayer(encoder_weights[2], encoder_weights[3])
        self.hidden_layer2 = DenseLayer(encoder_weights[4], encoder_weights[5])
        self.hidden_layer3 = DenseLayer(encoder_weights[6], encoder_weights[7])
        self.output_layer  = LinearLayer(encoder_weights[8], encoder_weights[9])
        
    def call(self, input_data):
        fx = self.input_layer(input_data)
        fx = self.hidden_layer1(fx)
        fx = self.hidden_layer2(fx)
        fx = self.hidden_layer3(fx)
        return self.output_layer(fx)

class NeuralNetworkModel(tf.keras.Model):
    def __init__(self):
        super(NeuralNetworkModel, self).__init__(dtype = 'float64')
        self.EN = Encoder()
        
    def call(self, inputs):
        X        = inputs[0]
        Y        = inputs[1]
        
        Psi_X    = self.EN(X)
        Psi_Y    = self.EN(Y)
        
        PSI_X    = tf.concat([X, Psi_X], 1)
        PSI_Y    = tf.concat([Y, Psi_Y], 1)
        return PSI_X, PSI_Y
        
class SurrogateModel:
    def __init__(self):
        # load neural network model
        self.Model = NeuralNetworkModel()
        
    def predict(self, X0, num_time_steps):
        # Transform the current_data (which is in state space) to Latent space
        PSI_X, PSI_Y = self.Model([X0, X0])

        X_pred = np.zeros((num_time_steps, K.shape[0]))
        X_pred[0,:] = PSI_X

        for k in range(num_time_steps-1):
            X_pred[k+1, :] = np.matmul(X_pred[k, :], K)

        return X_pred
