# u-net model with up-convolution or up-sampling and weighted binary-crossentropy as loss func

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, LeakyReLU
from keras.optimizers import Adam
from keras.initializers import he_uniform

from keras.utils import plot_model
from keras import backend as K

def unet_model(n_classes=5, im_sz=160, n_channels=8, n_filters_start=32, growth_factor=2):
    droprate=0.25

    #Block1
    n_filters = n_filters_start
    inputs = Input((im_sz, im_sz, n_channels), name='input')
    #inputs = BatchNormalization()(inputs)
    conv1 = Conv2D(n_filters, (3, 3),  padding='same', name = 'conv1_1', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(inputs)
    actv1 = LeakyReLU(name = 'actv1_1')(conv1)
    conv1 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv1_2', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv1)
    actv1 = LeakyReLU(name = 'actv1_2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name = 'maxpool1')(actv1)
    #pool1 = Dropout(droprate)(pool1)

    #Block2
    n_filters *= growth_factor
    pool1 = BatchNormalization(name = 'bn1')(pool1)
    conv2 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv2_1', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(pool1)
    actv2 = LeakyReLU(name = 'actv2_1')(conv2)
    conv2 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv2_2', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv2)
    actv2 = LeakyReLU(name = 'actv2_2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name = 'maxpool2')(actv2)
    pool2 = Dropout(droprate, name = 'dropout2')(pool2)

    #Block3
    n_filters *= growth_factor
    pool2 = BatchNormalization(name = 'bn2')(pool2)
    conv3 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv3_1', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(pool2)
    actv3 = LeakyReLU(name = 'actv3_1')(conv3)
    conv3 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv3_2', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv3)
    actv3 = LeakyReLU(name = 'actv3_2')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name = 'maxpool3')(actv3)
    pool3 = Dropout(droprate, name = 'dropout3')(pool3)

    #Block4
    n_filters *= growth_factor
    pool3 = BatchNormalization(name = 'bn3')(pool3)
    conv4_0 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv4_1', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(pool3)
    actv4_0 = LeakyReLU(name = 'actv4_1')(conv4_0)
    conv4_0 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv4_0_2', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv4_0)
    actv4_0 = LeakyReLU(name = 'actv4_2')(conv4_0)
    pool4_1 = MaxPooling2D(pool_size=(2, 2), name = 'maxpool4')(actv4_0)
    pool4_1 = Dropout(droprate, name = 'dropout4')(pool4_1)

    #Block5
    n_filters *= growth_factor
    pool4_1 = BatchNormalization(name = 'bn4')(pool4_1)
    conv4_1 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv5_1', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(pool4_1)
    actv4_1 = LeakyReLU(name = 'actv5_1')(conv4_1)
    conv4_1 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv5_2', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv4_1)
    actv4_1 = LeakyReLU(name = 'actv5_2')(conv4_1)
    pool4_2 = MaxPooling2D(pool_size=(2, 2), name = 'maxpool5')(actv4_1)
    pool4_2 = Dropout(droprate, name = 'dropout5')(pool4_2)

    #Block6
    n_filters *= growth_factor
    conv5 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv6_1', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(pool4_2)
    actv5 = LeakyReLU(name = 'actv6_1')(conv5)
    conv5 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv6_2', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv5)
    actv5 = LeakyReLU(name = 'actv6_2')(conv5)

    #Block7
    n_filters //= growth_factor
    up6_1 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same', name = 'up7')(actv5), actv4_1], name = 'concat7')
    up6_1 = BatchNormalization(name = 'bn7')(up6_1)
    conv6_1 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv7_1', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(up6_1)
    actv6_1 = LeakyReLU(name = 'actv7_1')(conv6_1)
    conv6_1 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv7_2', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv6_1)
    actv6_1 = LeakyReLU(name = 'actv7_2')(conv6_1)
    conv6_1 = Dropout(droprate, name = 'dropout7')(actv6_1)

    #Block8
    n_filters //= growth_factor
    up6_2 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same', name = 'up8')(conv6_1), actv4_0], name = 'concat8')
    up6_2 = BatchNormalization(name = 'bn8')(up6_2)
    conv6_2 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv8_1', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(up6_2)
    actv6_2 = LeakyReLU(name = 'actv8_1')(conv6_2)
    conv6_2 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv8_2', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv6_2)
    actv6_2 = LeakyReLU(name = 'actv8_2')(conv6_2)
    conv6_2 = Dropout(droprate, name = 'dropout8')(actv6_2)

    #Block9
    n_filters //= growth_factor
    up7 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same', name = 'up9')(conv6_2), actv3], name = 'concat9')
    up7 = BatchNormalization(name = 'bn9')(up7)
    conv7 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv9_1', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(up7)
    actv7 = LeakyReLU(name = 'actv9_1')(conv7)
    conv7 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv9_2', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv7)
    actv7 = LeakyReLU(name = 'actv9_2')(conv7)
    conv7 = Dropout(droprate, name = 'dropout9')(actv7)

    #Block10
    n_filters //= growth_factor
    up8 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same', name = 'up10')(conv7), actv2], name = 'concat10')
    up8 = BatchNormalization(name = 'bn10')(up8)
    conv8 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv10_1', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(up8)
    actv8 = LeakyReLU(name = 'actv10_1')(conv8)
    conv8 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv10_2', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv8)
    actv8 = LeakyReLU(name = 'actv10_2')(conv8)
    conv8 = Dropout(droprate, name = 'dropout10')(actv8)

    #Block11
    n_filters //= growth_factor
    up9 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same', name = 'up11')(conv8), actv1], name = 'concat11')
    conv9 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv11_1', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(up9)
    actv9 = LeakyReLU(name = 'actv11_1')(conv9)
    conv9 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv11_2', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv9)
    actv9 = LeakyReLU(name = 'actv11_2')(conv9)

    conv10 = Conv2D(n_classes, (1, 1), activation='softmax', name = 'output1')(actv9)

    model = Model(inputs=inputs, outputs=conv10)

    def dice_coef(y_true, y_pred, smooth=1e-7):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    """This simply calculates the dice score for each individual label, and then sums them together, and includes the background."""
    def dice_coef_multilabel(y_true, y_pred):
        dice=n_classes
        for index in range(n_classes):
            dice -= dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
        return dice/n_classes

    model.compile(optimizer=Adam(lr = 10e-5), loss=dice_coef_multilabel)
    return model
