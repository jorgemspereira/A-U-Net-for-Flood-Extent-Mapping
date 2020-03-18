from keras.initializers import he_normal, Initializer
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, add, Permute, Conv2D, LeakyReLU, BatchNormalization
from keras import backend as K
from model.mish import Mish, Swish
 
class ANInitializer(Initializer):
    def __init__(self, scale=0.1, bias=0., seed=1997):
        super(ANInitializer, self).__init__()
        self.scale = scale
        self.bias = bias
        self.seed = seed

    def __call__(self, shape, dtype=None):
        dtype = dtype or K.floatx()
        return self.scale * K.random_normal(shape=shape, mean=0.0, stddev=1., seed=self.seed) + self.bias

class AttentiveNormalization(BatchNormalization):
    
    def __init__(self, n_mixture=5, momentum=0.99, epsilon=0.1, axis=-1, **kwargs):
        super(AttentiveNormalization, self).__init__(momentum=momentum, epsilon=epsilon, axis=axis, center=False, scale=False, **kwargs)
        if self.axis == -1: self.data_format = 'channels_last'
        else: self.data_format = 'channel_first'
        self.n_mixture = n_mixture
        
    def build(self, input_shape):
        if len(input_shape) != 4 and len(input_shape) != 3: raise ValueError('expected 3D or 4D input, got shape {}'.format(input_shape))
        super(AttentiveNormalization, self).build(input_shape)        
        dim = input_shape[self.axis]
        shape = (self.n_mixture, dim) # K x C 
        self.FC = layers.Dense(self.n_mixture, activation="sigmoid")
        self.FC.build(input_shape) # (N, C)
        if len(input_shape) == 4: self.GlobalAvgPooling = layers.GlobalAveragePooling2D(self.data_format)
        else: self.GlobalAvgPooling = layers.GlobalAveragePooling1D(self.data_format)
        self.GlobalAvgPooling.build(input_shape)
        self._trainable_weights = self.FC.trainable_weights
        self.learnable_weights = self.add_weight(name='gamma2', shape=shape, initializer=ANInitializer(scale=0.1, bias=1.), trainable=True)
        self.learnable_bias = self.add_weight(name='bias2', shape=shape, initializer=ANInitializer(scale=0.1, bias=0.), trainable=True)
        
    def call(self, input):
        # input is a batch of shape : (N, H, W, C)
        avg = self.GlobalAvgPooling(input) # N x C 
        attention = self.FC(avg) # N x K 
        gamma_readjust = K.dot(attention, self.learnable_weights) # N x C
        beta_readjust  = K.dot(attention, self.learnable_bias)  # N x C
        out_BN = super(AttentiveNormalization, self).call(input) # rescale input, N x H x W x C
        if K.int_shape(input)[0] is None or K.int_shape(input)[0] > 1:
            if len(input_shape) == 4:
                gamma_readjust = gamma_readjust[:, None, None, :]
                beta_readjust  = beta_readjust[:, None, None, :]
            else:
                gamma_readjust = gamma_readjust[:, None, :]
                beta_readjust  = beta_readjust[:, None, :]
        return gamma_readjust * out_BN + beta_readjust

    def get_config(self):
        config = { 'n_mixture' : self.n_mixture }
        base_config = super(AttentiveNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def squeeze_excite_block(input, init_seed=None, ratio=2):
    ''' Create a channel-wise squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
    Returns: a keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)
    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer=he_normal(seed=init_seed), use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer=he_normal(seed=init_seed), use_bias=False)(se)
    if K.image_data_format() == 'channels_first': se = Permute((3, 1, 2))(se)
    x = multiply([init, se])
    return x


def spatial_squeeze_excite_block(input, init_seed=None):
    ''' Create a spatial squeeze-excite block
    Args:
        input: input tensor
    Returns: a keras tensor
    References
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    '''
    se = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False, kernel_initializer=he_normal(seed=init_seed))(input)
    x = multiply([input, se])
    return x


def channel_spatial_squeeze_excite(input, init_seed=None, ratio=16):
    ''' Create a spatial squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
    Returns: a keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    '''
    cse = squeeze_excite_block(input, init_seed, ratio)
    sse = spatial_squeeze_excite_block(input, init_seed)
    x = add([cse, sse])
    return x
