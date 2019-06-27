# Origial Paper : https://arxiv.org/pdf/1512.03385.pdf

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense

from ResNet_convblock import ConvLayer, BNLayer, ConvBlock
from ResNet_identity_block import IdentityBlock
from ResNet_first_layer import ReLuLayer, MaxPoolLayer

# Average pooling layer class
class AvgPool:

    def __init__(self, ksize):
        self.ksize = ksize

    def forward(self, X):
        return tf.nn.avg_pool(value = X, ksize = [1, ksize, ksize, 1], strides = [1, 1, 1, 1], padding = 'VALID')

    def get_params(self):
        return []

# Flatten layer class
class Flatten:

    def forward(self, X):
        return tf.contrib.layers.flatten(X)

    def get_params(self):
        return []

# Dense layer class
class DenseLayer:

    def __init__(self, mi, mo):
        self.W = tf.Variable((np.random.randn(mi, mo) * np.sqrt(2.0/mi)).astype(np.float32))
        self.b = tf.Variable(np.zeros(mo, dtype = np.float32))

    def forward(self, X):
        return tf.matmul(X, self.W) + self.b

    def copy_keras_layers(self, layer):
        W, b = layer.get_weights()
        op1 = self.W.assign(W)
        op2 = self.b.assign(b)
        self.session.run((op1, op2))

    def get_params(self):
        return [self.W, self.b]


def custom_softmax(x):
    m = tf.reduce_max(x, 1)
    X = X - m
    e = tf.exp(X)
    return e / tf.reduce_sum(e, -1)


class ResNet:

    def __init__(self, ):
        self.layer = [# Conv1
                      ConvLayer(f = 7, mi = 3, mo = 64, stride = 2, padding = 'SAME'),
                      BNLayer(64),
                      ReLuLayer(),
                      MaxPoolLayer(dims = 3)
                      # Conv2
                      ConvBlock(mi = 64, fm_sizes= [64, 64, 256], stride = 1),
                      IdentityBlock(mi = 256, fm_sizes= [64, 64, 256]),
                      IdentityBlock(mi = 256, fm_sizes= [64, 64, 256]),
                      # Conv3
                      ConvBlock(mi = 256, fm_sizes= [128, 128, 512], stride = 2),
                      IdentityBlock(mi = 512, fm_sizes= [128, 128, 512]),
                      IdentityBlock(mi = 512, fm_sizes= [128, 128, 512]),
                      IdentityBlock(mi = 512, fm_sizes= [128, 128, 512]),
                      # Conv4
                      ConvBlock(mi = 512, fm_sizes= [256, 256, 1024], stride = 2),
                      IdentityBlock(mi = 1024, fm_sizes= [256, 256, 1024]),
                      IdentityBlock(mi = 1024, fm_sizes= [256, 256, 1024]),
                      IdentityBlock(mi = 1024, fm_sizes= [256, 256, 1024])
                      IdentityBlock(mi = 1024, fm_sizes= [256, 256, 1024])
                      IdentityBlock(mi = 1024, fm_sizes= [256, 256, 1024])
                      # Conv5
                      ConvBlock(mi = 1024, fm_sizes= [512, 512, 2048], stride = 2),
                      IdentityBlock(mi = 2048, fm_sizes= [512, 512, 2048]),
                      IdentityBlock(mi = 2048, fm_sizes= [512, 512, 2048]),
                      # Average pooling / Fully connection / softmax
                      AvgPool(ksize= 7),
                      Flatten(),
                      DenseLayer(mi = 2048, mo = 1000)
                      ]

    def copy_keras_layers(self, layers):
        self.layers[0].copy_keras_layers(layers[1])         # ConvLayer at Conv1
        self.layers[1].copy_keras_layers(layers[2])         # BNLayer at Conv1
        self.layers[4].copy_keras_layers(layers[5:17])      # ConvBlock at Conv2
        self.layers[5].copy_keras_layers(layers[17:27])     # IdentityBlock at Conv2
        self.layers[6].copy_keras_layers(layers[27:37])
        self.layers[7].copy_keras_layers(layers[37:49])     # ConvBlock at Conv3
        self.layers[8].copy_keras_layers(layers[49:59])     # IdentityBlock at Conv3
        self.layers[9].copy_keras_layers(layers[59:69])
        self.layers[10].copy_keras_layers(layers[69:79])
        self.layers[11].copy_keras_layers(layers[79:91])    # ConvBlock at Conv4
        self.layers[12].copy_keras_layers(layers[91:101])   # IdentityBlock at Conv4
        self.layers[13].copy_keras_layers(layers[101:111])
        self.layers[14].copy_keras_layers(layers[111:121])
        self.layers[15].copy_keras_layers(layers[121:131])
        self.layers[16].copy_keras_layers(layers[131:141])
        self.layers[17].copy_keras_layers(layers[141:153])  # ConvBlock at Conv5
        self.layers[18].copy_keras_layers(layers[153:163])  # IdentityBlock at Conv5
        self.layers[19].copy_keras_layers(layers[163:173])
        self.layers[22].copy_keras_layers(layers[175])     # DenseLayer


    def forward(self, layers):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def predict(self, X):
        assert self.session != None
        return self.session.run(self.output, feed_dict = {self.input : X})

    def set_session(self, session):
        self.session = session
        for layer in self.layers:
            # Setting session should be done separatly with _Block instances and other layers
            if isinstance(layer, ConvBlock) or isinstance(layer, IdentityBlock):
                layer.set_session(session)
            else:
                layer.session = session

    def get_params(self):
        params = []
        for layer in self.layers:
            params += layer.get_params()


if __name__=='__main__':

    resnet = ResNet50(weights = 'imagenet')

    # Getting the output just before the softmax layer of the original resnet
    x = resnet.layers[-2].output
    # Get the weights of the dense layer
    W, b = resnet.layers[-1].get_weights()
    # Add dense layer to the final layer (No softmax)
    y = Dense(1000)(x)
    # Rebuild a resnet model just without softmax
    resnet = Model(inputs = resnet.input, outputs = y)
    resnet.layers[-1].set_weights([W, b])
    output = resnet.layers[175].output

    resnet_no_softmax = Model(inputs = resnet.input, outputs = output)

    # Fake input image
    X = np.random.random((1, 224, 224, 3))
    keras_output = resnet_no_softmax.predict(X)

    # Create an instance of my ResNet!
    my_resnet = ResNet()

    init = tf.variables_initializer(my_resnet.get_params())
    session = keras.backend.get_session()
    my_resnet.set_session(session)
    session.run(init)

    # first, just make sure we can get any output
    test_output = my_resnet.predict(X)
    print("test_output.shape:", test_output.shape)

    # copy params from Keras model
    my_resnet.copy_keras_layers(resnet_no_softmax.layers)

    # compare the 2 models
    output = my_resnet.predict(X)
    diff = np.abs(output - keras_output).sum()

    if diff < 1e-10:
        print("Everything's great!")
    else:
        print("diff = %s" % diff)
