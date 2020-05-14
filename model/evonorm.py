import tensorflow as tf
from keras import backend as K
from keras.layers import *
from keras.initializers import *

DEFAULT_EPSILON_VALUE = 1e-5

def instance_std(x, eps=DEFAULT_EPSILON_VALUE):
    _, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    return K.sqrt(var + eps)

def group_std(inputs, groups=32, eps=DEFAULT_EPSILON_VALUE, axis=-1):
    groups = min(inputs.shape[axis], groups)
    input_shape = K.shape(inputs)
    group_shape = [input_shape[i] for i in range(4)]
    group_shape[axis] = input_shape[axis] // groups
    group_shape.insert(axis, groups)
    group_shape = K.stack(group_shape)
    grouped_inputs = K.reshape(inputs, group_shape)
    _, var = tf.nn.moments(grouped_inputs, [1, 2, 4], keepdims=True)
    std = K.sqrt(var + eps)
    std = tf.broadcast_to(std, K.shape(grouped_inputs))
    return K.reshape(std, input_shape)

class EvoNormB0(Layer):

    def __init__(self, channels=None, momentum=0.9, epsilon=DEFAULT_EPSILON_VALUE, name=None, **kwargs):
        super(EvoNormB0, self).__init__(name=name)
        self.momentum = momentum
        self.epsilon = epsilon
        self.channels = channels
        if channels is not None:
           self.gamma = self.add_weight(name="gamma", shape=(1, 1, 1, channels), initializer=Ones())
           self.beta = self.add_weight(name="beta", shape=(1, 1, 1, channels), initializer=Zeros())
           self.v_1 = self.add_weight(name="v1", shape=(1, 1, 1, channels), initializer=Ones())
           self.running_average_std = self.add_variable(trainable=False, shape=(1, 1, 1, channels), initializer=Ones())

    def call(self, inputs, training=True):
        if self.channels is None:
          self.channels = inputs.shape[ len(inputs.shape) - 1 ]
          self.gamma = self.add_weight(name="gamma", shape=(1, 1, 1, self.channels), initializer=Ones())
          self.beta = self.add_weight(name="beta", shape=(1, 1, 1, self.channels), initializer=Zeros())
          self.v_1 = self.add_weight(name="v1", shape=(1, 1, 1, self.channels), initializer=Ones())
          self.running_average_std = self.add_variable(trainable=False, shape=(1, 1, 1, self.channels), initializer=Ones())
        var = self.running_average_std
        if training:
            _, var = tf.nn.moments(inputs, [0, 1, 2], keepdims=True)
            self.running_average_std.assign(self.momentum * self.running_average_std + (1 - self.momentum) * var)
        else: pass
        denominator = tf.maximum(instance_std(inputs) + self.v_1 * inputs, K.sqrt(var + self.epsilon), )
        return inputs * self.gamma / denominator + self.beta

    def get_config(self):
        config = { 'channels': self.channels, }
        base_config = super(EvoNormS0, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class EvoNormS0(Layer):

    def __init__(self, channels=None, groups=8, name=None, **kwargs):
        super(EvoNormS0, self).__init__(name=name)
        self.groups = groups
        self.channels = channels
        if channels is not None:
          self.gamma = self.add_weight(name="gamma", shape=(1, 1, 1, channels), initializer=Ones())
          self.beta = self.add_weight(name="beta", shape=(1, 1, 1, channels), initializer=Zeros())
          self.v_1 = self.add_weight(name="v1", shape=(1, 1, 1, channels), initializer=Ones())

    def call(self, inputs, training=True):
        if self.channels is None:
          self.channels = inputs.shape[ len(inputs.shape) - 1 ]
          self.gamma = self.add_weight(name="gamma", shape=(1, 1, 1, self.channels), initializer=Ones())
          self.beta = self.add_weight(name="beta", shape=(1, 1, 1, self.channels), initializer=Zeros())
          self.v_1 = self.add_weight(name="v1", shape=(1, 1, 1, self.channels), initializer=Ones())
        return (inputs * K.sigmoid(self.v_1 * inputs)) / group_std(inputs, groups=self.groups) * self.gamma + self.beta

    def get_config(self):
        config = { 'group': self.groups, 'channels': self.channels, }
        base_config = super(EvoNormS0, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
