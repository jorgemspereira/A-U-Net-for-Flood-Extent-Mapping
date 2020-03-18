# Keras Implementation of Mish Activation Function.

from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
import keras.backend as K
import tensorflow as tf
import keras.layers as KL

class Swish(KL.Layer):
    def call(self, inputs): return tf.nn.swish(inputs)

class Mish(Activation):
    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'

def mish(x): return x*K.tanh(K.softplus(x))

def swish(x): return tf.nn.swish(x)

get_custom_objects().update({'Mish': Mish(mish)})
get_custom_objects().update({'Swish': Swish()})