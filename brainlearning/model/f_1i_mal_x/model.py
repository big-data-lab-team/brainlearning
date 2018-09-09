import tensorflow as tf
from keras import Input, Model
from keras.layers import LeakyReLU, Dropout, BatchNormalization, Flatten, Dense, Reshape, concatenate, UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers.core import Activation
from keras.losses import binary_crossentropy
from keras.optimizers import SGD, Adam
from keras.utils import multi_gpu_model
from keras import backend as K


def model_name():
    return 'f_1i_mal_x'


def build_model(number_of_gpus=2):
    # dropout_rate = 0.5

    # model_input = Input(shape=(320, 320, 1))
    # # Result x320 x1
    #
    # # Layer E1
    # # E1_1
    # e1_1 = Conv2D(64, (3, 3), padding='same')(model_input)
    # e1_1 = Activation("sigmoid")(e1_1)
    # e1_1 = BatchNormalization()(e1_1)
    # e1_1 = Dropout(rate=dropout_rate)(e1_1)
    #
    # e1_2 = Conv2D(64, (3, 3), padding='same', strides=(2, 2))(e1_1)
    # e1_2 = LeakyReLU(alpha=0.2)(e1_2)
    # e1_2 = BatchNormalization()(e1_2)
    # e1_res = Dropout(rate=dropout_rate)(e1_2)
    #
    # e1_pool = MaxPooling2D(pool_size=(2, 2))(e1_res)
    # # Result x160 x64
    #
    # # Layer E2
    # # E2_1
    # e2_1 = Conv2D(128, (3, 3), padding='same')(e1_pool)
    # e2_1 = LeakyReLU(alpha=0.2)(e2_1)
    # e2_1 = BatchNormalization()(e2_1)
    # e2_1 = Dropout(rate=dropout_rate)(e2_1)
    #
    # e2_2 = Conv2D(128, (3, 3), padding='same', strides=(2, 2))(e2_1)
    # e2_2 = LeakyReLU(alpha=0.2)(e2_2)
    # e2_2 = BatchNormalization()(e2_2)
    # e2_2 = Dropout(rate=dropout_rate)(e2_2)
    #
    # e2_out = MaxPooling2D(pool_size=(2, 2))(e2_2)
    # # Result x80 x128
    #
    # # Layer E3
    # # E3_1
    # e3_1 = Conv2D(256, (3, 3), padding='same')(e2_out)
    # e3_1 = LeakyReLU(alpha=0.2)(e3_1)
    # e3_1 = BatchNormalization()(e3_1)
    # e3_1 = Dropout(rate=dropout_rate)(e3_1)
    #
    # e3_2 = Conv2D(256, (3, 3), padding='same', strides=(2, 2))(e3_1)
    # e3_2 = LeakyReLU(alpha=0.2)(e3_2)
    # e3_2 = BatchNormalization()(e3_2)
    # e3_out = Dropout(rate=dropout_rate)(e3_2)
    #
    # # e3_out = MaxPooling2D(pool_size=(2, 2))(e3_2)
    # # Result x40 x256
    #
    # # Layer E4
    # # E4_1
    # e4_1 = Conv2D(512, (3, 3), padding='same')(e3_out)
    # e4_1 = BatchNormalization()(e4_1)
    # e4_1 = Dropout(rate=dropout_rate)(e4_1)
    # e4_1 = LeakyReLU(alpha=0.2)(e4_1)
    #
    # e4_2 = Conv2D(512, (3, 3), padding='same', strides=(2, 2))(e4_1)
    # e4_2 = BatchNormalization()(e4_2)
    # e4_2 = Dropout(rate=dropout_rate)(e4_2)
    # e4_out = LeakyReLU(alpha=0.2)(e4_2)
    #
    # # e4_out = MaxPooling2D(pool_size=(2, 2))(e4_2)
    # # Result x20 x512
    #
    # # Layer E5
    # # E5_1
    # # e5_1 = Conv2D(1024, (3, 3), padding='same')(e4_out)
    # # e5_1 = BatchNormalization()(e5_1)
    # # e5_1 = Dropout(rate=dropout_rate)(e5_1)
    # # e5_1 = LeakyReLU(alpha=0.2)(e5_1)
    # #
    # # e5_2 = Conv2D(1024, (3, 3), padding='same', strides=(2, 2))(e5_1)
    # # e5_2 = BatchNormalization()(e5_2)
    # # e5_2 = Dropout(rate=dropout_rate)(e5_2)
    # # e5_out = LeakyReLU(alpha=0.2)(e5_2)
    # #
    # # e5_out = MaxPooling2D(pool_size=(2, 2))(e5_2)
    # # Result x10 x1024
    #
    # # Layer E6
    # # E6_1
    # # e6_1 = Conv2D(1024, (3, 3), padding='same')(e5_out)
    # # e6_1 = BatchNormalization()(e6_1)
    # # e6_1 = Dropout(rate=dropout_rate)(e6_1)
    # # e6_1 = LeakyReLU(alpha=0.2)(e6_1)
    # #
    # # e6_2 = Conv2D(1024, (3, 3), padding='same')(e6_1)
    # # e6_2 = BatchNormalization()(e6_2)
    # # e6_2 = Dropout(rate=dropout_rate)(e6_2)
    # # e6_2 = LeakyReLU(alpha=0.2)(e6_2)
    # #
    # # e6_out = MaxPooling2D(pool_size=(2, 2))(e6_2)
    # # Result x5 x1024
    #
    # # Layer E7
    # # e7 = Flatten()(e3_out)
    # # e7 = Dense(2000)(e7)
    # # e7_out = Activation("sigmoid")(e7)
    #
    # # Middle Layer
    # mid_1 = Conv2D(1024, (3, 3), padding='same')(e4_out)
    # mid_1 = BatchNormalization()(mid_1)
    # mid_1 = Dropout(rate=dropout_rate)(mid_1)
    # mid_1 = LeakyReLU(alpha=0.2)(mid_1)
    #
    # mid_2 = Conv2D(1024, (3, 3), padding='same')(mid_1)
    # mid_2 = BatchNormalization()(mid_2)
    # mid_2 = Dropout(rate=dropout_rate)(mid_2)
    # mid_out = LeakyReLU(alpha=0.2)(mid_2)
    #
    #
    # # Layer D7
    # # d7 = Dense(32000)(e7_out)
    # # d7 = Activation("sigmoid")(d7)
    # # d7_out = Reshape((40, 40, -1))(d7)
    # # Result x5 x1024
    #
    # # Layer D6
    # # D6_1
    # # d6_2 = concatenate([e6_out, d7_out], axis=-1)
    # #
    # # d6_2 = Conv2DTranspose(1024, (3, 3), padding='same')(d6_2)
    # # d6_2 = BatchNormalization()(d6_2)
    # # d6_2 = Dropout(rate=dropout_rate)(d6_2)
    # # d6_2 = LeakyReLU(alpha=0.2)(d6_2)
    # #
    # # d6_1 = Conv2DTranspose(1024, (3, 3), padding='same', strides=(2, 2))(d6_2)
    # # d6_1 = BatchNormalization()(d6_1)
    # # d6_1 = Dropout(rate=dropout_rate)(d6_1)
    # # d6_out = LeakyReLU(alpha=0.2)(d6_1)
    # # Result x10 x1024
    #
    # # Layer D5
    # # D5_1
    #
    # # d5_2 = concatenate([e5_out, d6_out], axis=-1)
    # #
    # # d5_2 = Conv2DTranspose(512, (3, 3), padding='same')(d5_2)
    # # d5_2 = BatchNormalization()(d5_2)
    # # d5_2 = Dropout(rate=dropout_rate)(d5_2)
    # # d5_2 = LeakyReLU(alpha=0.2)(d5_2)
    # #
    # # d5_1 = Conv2DTranspose(512, (3, 3), padding='same', strides=(2, 2))(d5_2)
    # # d5_1 = BatchNormalization()(d5_1)
    # # d5_1 = Dropout(rate=dropout_rate)(d5_1)
    # # d5_out = LeakyReLU(alpha=0.2)(d5_1)
    # # Result x20 x512
    #
    # # Layer D4
    # # D4_1
    # d4_2 = concatenate([e4_out, mid_out], axis=-1)
    #
    # d4_2 = Conv2DTranspose(512, (3, 3), padding='same')(d4_2)
    # d4_2 = BatchNormalization()(d4_2)
    # d4_2 = Dropout(rate=dropout_rate)(d4_2)
    # d4_2 = LeakyReLU(alpha=0.2)(d4_2)
    #
    # d4_1 = Conv2DTranspose(512, (3, 3), padding='same', strides=(2, 2))(d4_2)
    # d4_1 = BatchNormalization()(d4_1)
    # d4_1 = Dropout(rate=dropout_rate)(d4_1)
    # d4_out = LeakyReLU(alpha=0.2)(d4_1)
    # # Result x40 x256
    #
    # # Layer D3
    # # D3_1
    # d3_2 = concatenate([e3_out, d4_out], axis=-1)
    #
    # d3_2 = Conv2DTranspose(256, (3, 3), padding='same')(d3_2)
    # d3_2 = LeakyReLU(alpha=0.2)(d3_2)
    # d3_2 = BatchNormalization()(d3_2)
    # d3_2 = Dropout(rate=dropout_rate)(d3_2)
    #
    # d3_1 = Conv2DTranspose(256, (3, 3), padding='same', strides=(2, 2))(d3_2)
    # d3_1 = LeakyReLU(alpha=0.2)(d3_1)
    # d3_1 = BatchNormalization()(d3_1)
    # d3_out = Dropout(rate=dropout_rate)(d3_1)
    # # Result x80 x128
    #
    # # Layer D2
    # # D2_1
    # d2_2 = concatenate([e2_out, d3_out], axis=-1)
    #
    # d2_2 = Conv2DTranspose(128, (3, 3), padding='same')(d2_2)
    # d2_2 = LeakyReLU(alpha=0.2)(d2_2)
    # d2_2 = BatchNormalization()(d2_2)
    # d2_2 = Dropout(rate=dropout_rate)(d2_2)
    #
    # d2_1 = Conv2DTranspose(128, (3, 3), padding='same', strides=(2, 2))(d2_2)
    # d2_1 = LeakyReLU(alpha=0.2)(d2_1)
    # d2_1 = BatchNormalization()(d2_1)
    # d2_out = Dropout(rate=dropout_rate)(d2_1)
    # # Result x160 x64
    #
    # # Layer D1
    # d1_2 = concatenate([e1_out, d2_out], axis=-1)
    #
    # d1_2 = Conv2DTranspose(64, (3, 3), padding='same')(d1_2)
    # d1_2 = LeakyReLU(alpha=0.2)(d1_2)
    # d1_2 = BatchNormalization()(d1_2)
    # d1_2 = Dropout(rate=dropout_rate)(d1_2)
    #
    # d1_1 = Conv2DTranspose(64, (3, 3), padding='same', strides=(2, 2))(d1_2)
    # d1_1 = Activation("sigmoid")(d1_1)
    # d1_1 = BatchNormalization()(d1_1)
    # d1_out = Dropout(rate=dropout_rate)(d1_1)
    # # Result x320
    #
    # # Layer out
    # model_output = Conv2DTranspose(2, (3, 3), padding='same')(d1_out)
    # model_output = BatchNormalization()(model_output)
    # model_output = Reshape((2, ))(model_output)
    # model_output = Activation("softmax")(model_output)
    # model_output = Dropout(rate=dropout_rate)(model_output)
    #
    # # Build the Model
    # model = Model(inputs=model_input, outputs=model_output)
    # model = multi_gpu_model(model, gpus=number_of_gpus)
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Constants
    HEIGHT = 320
    WIDTH = 320
    CHANNELS = 1
    DROPOUT_RATE = 0.5
    EPSILON = 1e-4

    inputs = Input(shape=(HEIGHT, WIDTH, CHANNELS))

    # down0, down0_res = down(24, inputs, EPSILON, DROPOUT_RATE)
    down1, down1_res = down( 64, inputs, EPSILON, DROPOUT_RATE)
    down2, down2_res = down(128, down1, EPSILON, DROPOUT_RATE)
    down3, down3_res = down(256, down2, EPSILON, DROPOUT_RATE)
    down4, down4_res = down(512, down3, EPSILON, DROPOUT_RATE)
    # down5, down5_res = down(768, down4, EPSILON, DROPOUT_RATE)

    center = Conv2D(1024, (3, 3), padding='same')(down4)
    center = BatchNormalization(epsilon=EPSILON)(center)
    center = LeakyReLU(alpha=0.2)(center)
    center = Dropout(rate=DROPOUT_RATE)(center)

    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization(epsilon=EPSILON)(center)
    center = LeakyReLU(alpha=0.2)(center)
    center = Dropout(rate=DROPOUT_RATE)(center)

    # up5 = up(768, center, down5_res, EPSILON, DROPOUT_RATE)
    up4 = up(512, center, down4_res, EPSILON, DROPOUT_RATE)
    up3 = up(256, up4, down3_res, EPSILON, DROPOUT_RATE)
    up2 = up(128, up3, down2_res, EPSILON, DROPOUT_RATE)
    up1 = up( 64, up2, down1_res, EPSILON, DROPOUT_RATE)
    # up0 = up(24, up1, down0_res, EPSILON, DROPOUT_RATE)

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

    # up_ = Conv2D(filters, (3, 3), padding='same')(up_)
    # up_ = BatchNormalization(epsilon=epsilon)(up_)
    # up_ = LeakyReLU(alpha=0.2)(up_)
    # up_ = Dropout(rate=dropout_rate)(up_)
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

