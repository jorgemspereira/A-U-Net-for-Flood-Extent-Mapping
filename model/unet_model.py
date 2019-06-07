from keras.initializers import he_uniform
from keras.layers import *
from keras.models import Model

from model.se import channel_spatial_squeeze_excite


def unet_model(n_classes=5, init_seed=None, im_sz=160, n_channels=8, n_filters_start=32, growth_factor=2, droprate=0.3):
    # Block1
    n_filters = n_filters_start
    inputs = Input((im_sz, im_sz, n_channels), name='input')
    conv1 = Conv2D(n_filters, (3, 3), padding='same', name='conv1_1', kernel_initializer=he_uniform(seed=init_seed),
                   bias_initializer=he_uniform(seed=init_seed))(inputs)
    actv1 = LeakyReLU(name='actv1_1')(conv1)
    conv1 = Conv2D(n_filters, (3, 3), padding='same', name='conv1_2', kernel_initializer=he_uniform(seed=init_seed),
                   bias_initializer=he_uniform(seed=init_seed))(actv1)
    actv1 = LeakyReLU(name='actv1_2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='maxpool1')(actv1)
    pool1 = channel_spatial_squeeze_excite(pool1, init_seed=init_seed)

    # Block2
    n_filters *= growth_factor
    pool1 = BatchNormalization(name='bn1')(pool1)
    conv2 = Conv2D(n_filters, (3, 3), padding='same', name='conv2_1', kernel_initializer=he_uniform(seed=init_seed),
                   bias_initializer=he_uniform(seed=init_seed))(pool1)
    actv2 = LeakyReLU(name='actv2_1')(conv2)
    conv2 = Conv2D(n_filters, (3, 3), padding='same', name='conv2_2', kernel_initializer=he_uniform(seed=init_seed),
                   bias_initializer=he_uniform(seed=init_seed))(actv2)
    actv2 = LeakyReLU(name='actv2_2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='maxpool2')(actv2)
    pool2 = channel_spatial_squeeze_excite(pool2, init_seed=init_seed)
    pool2 = Dropout(droprate, name='dropout2')(pool2)

    # Block3
    n_filters *= growth_factor
    pool2 = BatchNormalization(name='bn2')(pool2)
    conv3 = Conv2D(n_filters, (3, 3), padding='same', name='conv3_1', kernel_initializer=he_uniform(seed=init_seed),
                   bias_initializer=he_uniform(seed=init_seed))(pool2)
    actv3 = LeakyReLU(name='actv3_1')(conv3)
    conv3 = Conv2D(n_filters, (3, 3), padding='same', name='conv3_2', kernel_initializer=he_uniform(seed=init_seed),
                   bias_initializer=he_uniform(seed=init_seed))(actv3)
    actv3 = LeakyReLU(name='actv3_2')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='maxpool3')(actv3)
    pool3 = channel_spatial_squeeze_excite(pool3, init_seed=init_seed)
    pool3 = Dropout(droprate, name='dropout3')(pool3)

    # Block4
    n_filters *= growth_factor
    pool3 = BatchNormalization(name='bn3')(pool3)
    conv4_0 = Conv2D(n_filters, (3, 3), padding='same', name='conv4_1', kernel_initializer=he_uniform(seed=init_seed),
                     bias_initializer=he_uniform(seed=init_seed))(pool3)
    actv4_0 = LeakyReLU(name='actv4_1')(conv4_0)
    conv4_0 = Conv2D(n_filters, (3, 3), padding='same', name='conv4_0_2', kernel_initializer=he_uniform(seed=init_seed),
                     bias_initializer=he_uniform(seed=init_seed))(actv4_0)
    actv4_0 = LeakyReLU(name='actv4_2')(conv4_0)
    pool4_1 = MaxPooling2D(pool_size=(2, 2), name='maxpool4')(actv4_0)
    pool4_1 = channel_spatial_squeeze_excite(pool4_1, init_seed=init_seed)
    pool4_1 = Dropout(droprate, name='dropout4')(pool4_1)

    # Block5
    n_filters *= growth_factor
    pool4_1 = BatchNormalization(name='bn4')(pool4_1)
    conv4_1 = Conv2D(n_filters, (3, 3), padding='same', name='conv5_1', kernel_initializer=he_uniform(seed=init_seed),
                     bias_initializer=he_uniform(seed=init_seed))(pool4_1)
    actv4_1 = LeakyReLU(name='actv5_1')(conv4_1)
    conv4_1 = Conv2D(n_filters, (3, 3), padding='same', name='conv5_2', kernel_initializer=he_uniform(seed=init_seed),
                     bias_initializer=he_uniform(seed=init_seed))(actv4_1)
    actv4_1 = LeakyReLU(name='actv5_2')(conv4_1)
    pool4_2 = MaxPooling2D(pool_size=(2, 2), name='maxpool5')(actv4_1)
    pool4_2 = channel_spatial_squeeze_excite(pool4_2, init_seed=init_seed)
    pool4_2 = Dropout(droprate, name='dropout5')(pool4_2)

    # Block6
    n_filters *= growth_factor
    conv5 = Conv2D(n_filters, (3, 3), padding='same', name='conv6_1', kernel_initializer=he_uniform(seed=init_seed),
                   bias_initializer=he_uniform(seed=init_seed))(pool4_2)
    actv5 = LeakyReLU(name='actv6_1')(conv5)
    conv5 = Conv2D(n_filters, (3, 3), padding='same', name='conv6_2', kernel_initializer=he_uniform(seed=init_seed),
                   bias_initializer=he_uniform(seed=init_seed))(actv5)
    actv5 = LeakyReLU(name='actv6_2')(conv5)

    # Block7
    n_filters //= growth_factor
    up6_1_aux = concatenate([actv4_1,
                             AveragePooling2D(pool_size=(2, 2))(actv4_0),
                             AveragePooling2D(pool_size=(4, 4))(actv3),
                             AveragePooling2D(pool_size=(8, 8))(actv2),
                             AveragePooling2D(pool_size=(16, 16))(actv1)])
    up6_1_aux = Conv2D(n_filters, (1, 1), padding='same', kernel_initializer=he_uniform(seed=init_seed),
                       bias_initializer=he_uniform(seed=init_seed))(up6_1_aux)
    up6_1_aux = LeakyReLU()(up6_1_aux)

    up6_1 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same', name='up7')(actv5),
                         up6_1_aux], name='concat7')
    up6_1 = BatchNormalization(name='bn7')(up6_1)
    conv6_1 = Conv2D(n_filters, (3, 3), padding='same', name='conv7_1', kernel_initializer=he_uniform(seed=init_seed),
                     bias_initializer=he_uniform(seed=init_seed))(up6_1)
    actv6_1 = LeakyReLU(name='actv7_1')(conv6_1)
    conv6_1 = Conv2D(n_filters, (3, 3), padding='same', name='conv7_2', kernel_initializer=he_uniform(seed=init_seed),
                     bias_initializer=he_uniform(seed=init_seed))(actv6_1)
    conv6_1 = LeakyReLU(name='actv7_2')(conv6_1)
    conv6_1 = channel_spatial_squeeze_excite(conv6_1, init_seed=init_seed)
    conv6_1 = Dropout(droprate, name='dropout7')(conv6_1)

    # Block8
    n_filters //= growth_factor
    up6_2_aux = concatenate([actv4_0,
                             UpSampling2D(size=(2, 2), interpolation="bilinear")(actv4_1),
                             UpSampling2D(size=(4, 4), interpolation="bilinear")(actv5),
                             AveragePooling2D(pool_size=(2, 2))(actv3),
                             AveragePooling2D(pool_size=(4, 4))(actv2),
                             AveragePooling2D(pool_size=(8, 8))(actv1)])
    up6_2_aux = Conv2D(n_filters, (1, 1), padding='same', kernel_initializer=he_uniform(seed=init_seed),
                       bias_initializer=he_uniform(seed=init_seed))(up6_2_aux)
    up6_2_aux = LeakyReLU()(up6_2_aux)

    up6_2 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same', name='up8')(conv6_1),
                         up6_2_aux], name='concat8')
    up6_2 = BatchNormalization(name='bn8')(up6_2)
    conv6_2 = Conv2D(n_filters, (3, 3), padding='same', name='conv8_1', kernel_initializer=he_uniform(seed=init_seed),
                     bias_initializer=he_uniform(seed=init_seed))(up6_2)
    actv6_2 = LeakyReLU(name='actv8_1')(conv6_2)
    conv6_2 = Conv2D(n_filters, (3, 3), padding='same', name='conv8_2', kernel_initializer=he_uniform(seed=init_seed),
                     bias_initializer=he_uniform(seed=init_seed))(actv6_2)
    conv6_2 = LeakyReLU(name='actv8_2')(conv6_2)
    conv6_2 = channel_spatial_squeeze_excite(conv6_2, init_seed=init_seed)
    conv6_2 = Dropout(droprate, name='dropout8')(conv6_2)

    # Block9
    n_filters //= growth_factor
    up7_aux = concatenate([actv3,
                           UpSampling2D(size=(2, 2), interpolation="bilinear")(actv4_0),
                           UpSampling2D(size=(4, 4), interpolation="bilinear")(actv4_1),
                           UpSampling2D(size=(8, 8), interpolation="bilinear")(actv5),
                           AveragePooling2D(pool_size=(2, 2))(actv2),
                           AveragePooling2D(pool_size=(4, 4))(actv1)])
    up7_aux = Conv2D(n_filters, (1, 1), padding='same', kernel_initializer=he_uniform(seed=init_seed),
                     bias_initializer=he_uniform(seed=init_seed))(up7_aux)
    up7_aux = LeakyReLU()(up7_aux)

    up7 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same', name='up9')(conv6_2),
                       up7_aux], name='concat9')
    up7 = BatchNormalization(name='bn9')(up7)
    conv7 = Conv2D(n_filters, (3, 3), padding='same', name='conv9_1', kernel_initializer=he_uniform(seed=init_seed),
                   bias_initializer=he_uniform(seed=init_seed))(up7)
    actv7 = LeakyReLU(name='actv9_1')(conv7)
    conv7 = Conv2D(n_filters, (3, 3), padding='same', name='conv9_2', kernel_initializer=he_uniform(seed=init_seed),
                   bias_initializer=he_uniform(seed=init_seed))(actv7)
    conv7 = LeakyReLU(name='actv9_2')(conv7)
    conv7 = channel_spatial_squeeze_excite(conv7, init_seed=init_seed)
    conv7 = Dropout(droprate, name='dropout9')(conv7)

    # Block10
    n_filters //= growth_factor
    up8_aux = concatenate([actv2,
                           UpSampling2D(size=(2, 2), interpolation="bilinear")(actv3),
                           UpSampling2D(size=(4, 4), interpolation="bilinear")(actv4_0),
                           UpSampling2D(size=(8, 8), interpolation="bilinear")(actv4_1),
                           UpSampling2D(size=(16, 16), interpolation="bilinear")(actv5),
                           AveragePooling2D(pool_size=(2, 2))(actv1)])
    up8_aux = Conv2D(n_filters, (1, 1), padding='same', kernel_initializer=he_uniform(seed=init_seed),
                     bias_initializer=he_uniform(seed=init_seed))(up8_aux)
    up8_aux = LeakyReLU()(up8_aux)

    up8 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same', name='up10')(conv7),
                       up8_aux], name='concat10')
    up8 = BatchNormalization(name='bn10')(up8)
    conv8 = Conv2D(n_filters, (3, 3), padding='same', name='conv10_1', kernel_initializer=he_uniform(seed=init_seed),
                   bias_initializer=he_uniform(seed=init_seed))(up8)
    actv8 = LeakyReLU(name='actv10_1')(conv8)
    conv8 = Conv2D(n_filters, (3, 3), padding='same', name='conv10_2', kernel_initializer=he_uniform(seed=init_seed),
                   bias_initializer=he_uniform(seed=init_seed))(actv8)
    conv8 = LeakyReLU(name='actv10_2')(conv8)
    conv8 = channel_spatial_squeeze_excite(conv8, init_seed=init_seed)
    conv8 = Dropout(droprate, name='dropout10')(conv8)

    # Block11
    n_filters //= growth_factor
    up9_aux = concatenate([actv1,
                           UpSampling2D(size=(2, 2), interpolation="bilinear")(actv2),
                           UpSampling2D(size=(4, 4), interpolation="bilinear")(actv3),
                           UpSampling2D(size=(8, 8), interpolation="bilinear")(actv4_0),
                           UpSampling2D(size=(16, 16), interpolation="bilinear")(actv4_1),
                           UpSampling2D(size=(32, 32), interpolation="bilinear")(actv5)])
    up9_aux = Conv2D(n_filters, (1, 1), padding='same', kernel_initializer=he_uniform(seed=init_seed),
                     bias_initializer=he_uniform(seed=init_seed))(up9_aux)
    up9_aux = LeakyReLU()(up9_aux)

    up9 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same', name='up11')(conv8),
                       up9_aux], name='concat11')
    conv9 = Conv2D(n_filters, (3, 3), padding='same', name='conv11_1', kernel_initializer=he_uniform(seed=init_seed),
                   bias_initializer=he_uniform(seed=init_seed))(up9)
    actv9 = LeakyReLU(name='actv11_1')(conv9)
    conv9 = Conv2D(n_filters, (3, 3), padding='same', name='conv11_2', kernel_initializer=he_uniform(seed=init_seed),
                   bias_initializer=he_uniform(seed=init_seed))(actv9)
    actv9 = LeakyReLU(name='actv11_2')(conv9)
    actv9 = channel_spatial_squeeze_excite(actv9, init_seed=init_seed)

    conv10 = Conv2D(n_classes, (1, 1), activation='sigmoid', name='output1')(actv9)

    return Model(inputs=inputs, outputs=conv10)
