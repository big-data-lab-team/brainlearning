from keras import Input, Model
from keras import Input, Model
from keras import backend as K
from keras.layers import LeakyReLU, Dropout, BatchNormalization, concatenate, UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.utils import multi_gpu_model


def model_name():
    return 'f_1i_mal_y'


def build_model(number_of_gpus=2):
    # Constants
    HEIGHT = 320
    WIDTH = 320
    CHANNELS = 1
    DROPOUT_RATE = 0.5
    EPSILON = 1e-4

    inputs = Input(shape=(HEIGHT, WIDTH, CHANNELS))

    down1, down1_res = down( 64, inputs, EPSILON, DROPOUT_RATE)
    down2, down2_res = down(128, down1, EPSILON, DROPOUT_RATE)
    down3, down3_res = down(256, down2, EPSILON, DROPOUT_RATE)
    down4, down4_res = down(512, down3, EPSILON, DROPOUT_RATE)

    center = Conv2D(1024, (3, 3), padding='same')(down4)
    center = BatchNormalization(epsilon=EPSILON)(center)
    center = LeakyReLU(alpha=0.2)(center)
    center = Dropout(rate=DROPOUT_RATE)(center)

    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization(epsilon=EPSILON)(center)
    center = LeakyReLU(alpha=0.2)(center)
    center = Dropout(rate=DROPOUT_RATE)(center)

    up4 = up(512, center, down4_res, EPSILON, DROPOUT_RATE)
    up3 = up(256, up4, down3_res, EPSILON, DROPOUT_RATE)
    up2 = up(128, up3, down2_res, EPSILON, DROPOUT_RATE)
    up1 = up( 64, up2, down1_res, EPSILON, DROPOUT_RATE)

    classify = Conv2D(1, (1, 1), activation='sigmoid', name='final_layer')(up1)

    model = Model(inputs=inputs, outputs=classify)

    model = multi_gpu_model(model, gpus=number_of_gpus)
    model.compile(loss=bce_dice_loss, optimizer=Adam(lr=EPSILON), metrics=[dice_coef])

    return model


def down(filters, input_, epsilon, dropout_rate):
    down_ = Conv2D(filters, (3, 3), padding='same')(input_)
    down_ = BatchNormalization(epsilon=epsilon)(down_)
    down_ = LeakyReLU(alpha=0.2)(down_)
    down_ = Dropout(rate=dropout_rate)(down_)

    down_ = Conv2D(filters, (3, 3), padding='same')(down_)
    down_ = BatchNormalization(epsilon=epsilon)(down_)
    down_ = LeakyReLU(alpha=0.2)(down_)
    down_res = Dropout(rate=dropout_rate)(down_)

    down_pool = MaxPooling2D((2, 2))(down_)
    return down_pool, down_res


def up(filters, input_, down_, epsilon, dropout_rate):
    up_ = UpSampling2D((2, 2))(input_)

    up_ = concatenate([down_, up_], axis=3)

    up_ = Conv2D(filters, (3, 3), padding='same')(up_)
    up_ = BatchNormalization(epsilon=epsilon)(up_)
    up_ = LeakyReLU(alpha=0.2)(up_)
    up_ = Dropout(rate=dropout_rate)(up_)

    up_ = Conv2D(filters, (3, 3), padding='same')(up_)
    up_ = BatchNormalization(epsilon=epsilon)(up_)
    up_ = LeakyReLU(alpha=0.2)(up_)
    up_ = Dropout(rate=dropout_rate)(up_)

    return up_


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)

