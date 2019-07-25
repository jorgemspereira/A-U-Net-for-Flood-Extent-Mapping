from keras.initializers import he_uniform
from keras.layers import *
from keras.models import Model

from model.se import channel_spatial_squeeze_excite


def conv2d_compress_block(input_tensor, n_filters, init_seed=None):
    x = Conv2D(filters=n_filters, kernel_size=(1, 1), kernel_initializer=he_uniform(seed=init_seed),
               bias_initializer=he_uniform(seed=init_seed), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def conv2d_transpose_block(input_tensor, n_filters):
    x = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def conv2d_super_block(input_tensor, n_filters, init_seed=None):
    x = Conv2D(filters=n_filters, kernel_size=(3, 3), kernel_initializer=he_uniform(seed=init_seed),
               bias_initializer=he_uniform(seed=init_seed), padding="same")(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    y = concatenate([x, input_tensor])
    y = Conv2D(filters=n_filters, kernel_size=(3, 3), kernel_initializer=he_uniform(seed=init_seed),
               bias_initializer=he_uniform(seed=init_seed), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    z = concatenate([x, y, input_tensor])
    z = conv2d_compress_block(z, n_filters, init_seed=init_seed)
    return z


def unet_model(n_classes=5, init_seed=None, im_sz=160, n_channels=8, n_filters_start=32, growth_factor=2, droprate=0.5):
    inputs = Input((im_sz, im_sz, n_channels))

    # Block1
    n_filters = n_filters_start
    conv1 = conv2d_super_block(inputs, n_filters, init_seed=init_seed)
    conv1 = channel_spatial_squeeze_excite(conv1, init_seed=init_seed)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(droprate * 0.5)(pool1)

    # Block2
    n_filters *= growth_factor
    conv2 = conv2d_super_block(pool1, n_filters, init_seed=init_seed)
    conv2 = channel_spatial_squeeze_excite(conv2, init_seed=init_seed)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(droprate)(pool2)

    # Block3
    n_filters *= growth_factor
    conv3 = conv2d_super_block(pool2, n_filters, init_seed=init_seed)
    conv3 = channel_spatial_squeeze_excite(conv3, init_seed=init_seed)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(droprate)(pool3)

    # Block4
    n_filters *= growth_factor
    conv4 = conv2d_super_block(pool3, n_filters, init_seed=init_seed)
    conv4 = channel_spatial_squeeze_excite(conv4, init_seed=init_seed)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(droprate)(pool4)

    # Block5
    n_filters *= growth_factor
    conv5 = conv2d_super_block(pool4, n_filters, init_seed=init_seed)
    conv5 = channel_spatial_squeeze_excite(conv5, init_seed=init_seed)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    pool5 = Dropout(droprate)(pool5)

    # Block6
    n_filters *= growth_factor
    conv6 = conv2d_super_block(pool5, n_filters, init_seed=init_seed)

    # Block7
    n_filters //= growth_factor
    up7_aux = concatenate([AveragePooling2D(pool_size=(2, 2))(conv4),
                           AveragePooling2D(pool_size=(4, 4))(conv3),
                           AveragePooling2D(pool_size=(8, 8))(conv2),
                           AveragePooling2D(pool_size=(16, 16))(conv1)])
    up7_aux = conv2d_compress_block(up7_aux, n_filters, init_seed=init_seed)

    up7 = conv2d_transpose_block(conv6, n_filters)
    up7 = concatenate([up7, up7_aux, conv5])
    up7 = conv2d_compress_block(up7, n_filters, init_seed=init_seed)

    conv7 = Dropout(droprate)(up7)
    conv7 = conv2d_super_block(conv7, n_filters, init_seed=init_seed)
    conv7 = channel_spatial_squeeze_excite(conv7, init_seed=init_seed)

    # Block8
    n_filters //= growth_factor
    up8_aux = concatenate([UpSampling2D(size=(2, 2), interpolation="bilinear")(conv5),
                           UpSampling2D(size=(4, 4), interpolation="bilinear")(conv6),
                           AveragePooling2D(pool_size=(2, 2))(conv3),
                           AveragePooling2D(pool_size=(4, 4))(conv2),
                           AveragePooling2D(pool_size=(8, 8))(conv1)])
    up8_aux = conv2d_compress_block(up8_aux, n_filters, init_seed=init_seed)

    up8 = conv2d_transpose_block(conv7, n_filters)
    up8 = concatenate([up8, up8_aux, conv4])
    up8 = conv2d_compress_block(up8, n_filters, init_seed=init_seed)

    conv8 = Dropout(droprate)(up8)
    conv8 = conv2d_super_block(conv8, n_filters, init_seed=init_seed)
    conv8 = channel_spatial_squeeze_excite(conv8, init_seed=init_seed)

    # Block9
    n_filters //= growth_factor
    up9_aux = concatenate([UpSampling2D(size=(2, 2), interpolation="bilinear")(conv4),
                           UpSampling2D(size=(4, 4), interpolation="bilinear")(conv5),
                           UpSampling2D(size=(8, 8), interpolation="bilinear")(conv6),
                           AveragePooling2D(pool_size=(2, 2))(conv2),
                           AveragePooling2D(pool_size=(4, 4))(conv1)])
    up9_aux = conv2d_compress_block(up9_aux, n_filters, init_seed=init_seed)

    up9 = conv2d_transpose_block(conv8, n_filters)
    up9 = concatenate([up9, up9_aux, conv3])
    up9 = conv2d_compress_block(up9, n_filters, init_seed=init_seed)

    conv9 = Dropout(droprate)(up9)
    conv9 = conv2d_super_block(conv9, n_filters, init_seed=init_seed)
    conv9 = channel_spatial_squeeze_excite(conv9, init_seed=init_seed)

    # Block10
    n_filters //= growth_factor
    up10_aux = concatenate([UpSampling2D(size=(2, 2), interpolation="bilinear")(conv3),
                            UpSampling2D(size=(4, 4), interpolation="bilinear")(conv4),
                            UpSampling2D(size=(8, 8), interpolation="bilinear")(conv5),
                            UpSampling2D(size=(16, 16), interpolation="bilinear")(conv6),
                            AveragePooling2D(pool_size=(2, 2))(conv1)])
    up10_aux = conv2d_compress_block(up10_aux, n_filters, init_seed=init_seed)

    up10 = conv2d_transpose_block(conv9, n_filters)
    up10 = concatenate([up10, up10_aux, conv2])
    up10 = conv2d_compress_block(up10, n_filters, init_seed=init_seed)

    conv10 = Dropout(droprate)(up10)
    conv10 = conv2d_super_block(conv10, n_filters, init_seed=init_seed)
    conv10 = channel_spatial_squeeze_excite(conv10, init_seed=init_seed)

    # Block11
    n_filters //= growth_factor
    up11_aux = concatenate([UpSampling2D(size=(2, 2), interpolation="bilinear")(conv2),
                            UpSampling2D(size=(4, 4), interpolation="bilinear")(conv3),
                            UpSampling2D(size=(8, 8), interpolation="bilinear")(conv4),
                            UpSampling2D(size=(16, 16), interpolation="bilinear")(conv5),
                            UpSampling2D(size=(32, 32), interpolation="bilinear")(conv6)])
    up11_aux = conv2d_compress_block(up11_aux, n_filters, init_seed=init_seed)

    up11 = conv2d_transpose_block(conv10, n_filters)
    up11 = concatenate([up11, up11_aux, conv1])
    up11 = conv2d_compress_block(up11, n_filters, init_seed=init_seed)

    conv11 = Dropout(droprate)(up11)
    conv11 = conv2d_super_block(conv11, n_filters, init_seed=init_seed)
    conv11 = channel_spatial_squeeze_excite(conv11, init_seed=init_seed)

    conv11 = Conv2D(n_classes, (1, 1), activation='sigmoid')(conv11)

    return Model(inputs=inputs, outputs=conv11)
