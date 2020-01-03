from keras.initializers import he_normal
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, add, Permute, Conv2D, LeakyReLU
from keras import backend as K
from model.mish import Mish

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
    se = Dense(filters // ratio, activation='Mish', kernel_initializer=he_normal(seed=init_seed), use_bias=False)(se)
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
