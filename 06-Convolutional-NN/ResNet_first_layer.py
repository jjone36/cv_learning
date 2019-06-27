
import numpy as np
import tensorflow as tf

import keras
from keras.applications.resnet50 import ResNet50
from keras.models import Model

from ResNet_convblock import ConvBlock, ConvLayer, BNLayer

class ReLuLayer:

    def forward(self, X):
        return tf.nn.relu(X)

    def get_params(self):
        return []


class MaxPoolLayer:

    def __init__(self, dim):
        self.dim = dim

    def forward(self, X):
        X = tf.nn.max_pool(value = X,
                          ksize = [1, self.dim, self.dim, 1],
                          strides = [1, 2, 2, 1],
                          padding = 'VALID')
        return X

    def get_params(self):
        return []


class ResNetHead:

    def __init__(self):
        self.layers = [# Conv1
                        ConvLayer(f = 7, mi = 3, mo = 64, stride = 2, padding = 'SAME'),
                        BNLayer(64),
                        ReLuLayer(),
                        MaxPoolLayer(dim = 3),
                        # Conv2
                        ConvBlock(mi = 64, fm_sizes=[64, 64, 256], stride = 1)]

        self.input = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
        self.output = self.forward(self.input)

    # This is for sanity check later
    def copy_keras_layers(self, layers):
        self.layers[0].copy_keras_layers(layers[1])
        self.layers[1].copy_keras_layers(layers[2])
        self.layers[4].copy_keras_layers(layers[5:])

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def predict(self, X):
        assert self.session != None
        return self.session.run(self.output, feed_dict = {self.input : X})

    def set_session(self, session):
        self.session = session
        self.layers[0].session = session
        self.layers[1].session = session
        self.layers[4].set_session(session)

    def get_params(self):
        params = []
        for layer in self.layers:
            params += layer.get_params()


if __name__=='__main__':

    # Get the weights from the ResNet50
    resnet = ResNet50(weights = 'imagenet')
    partial_resnet = Model(input = resnet.input, outputs = resnet.layers[16].output)

    # Fake input image
    X = np.random.random((1, 224, 224, 3))
    keras_output = partial_resnet.predict(X)

    # Create an instance of the frist layer!
    my_resnet_head = ResNetHead()

    init = tf.variables_initializer(my_resnet_head.get_params())
    session = keras.backend.get_session()
    my_resnet_head.set_session(session)
    session.run(init)

    # first, just make sure we can get any output
    test_output = my_resnet_head.predict(X)
    print("test_output.shape:", test_output.shape)

    # copy params from Keras model
    my_resnet_head.copy_keras_layers(partial_resnet.layers)

    # compare the 2 models
    output = my_resnet_head.predict(X)
    diff = np.abs(output - keras_output).sum()

    if diff < 1e-10:
        print("Everything's great!")
    else:
        print("diff = %s" % diff)
