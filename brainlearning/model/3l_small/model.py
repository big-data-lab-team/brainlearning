import tensorflow as tf
from keras.engine import Layer
from keras.layers import LeakyReLU, Dropout, BatchNormalization, Flatten, Dense, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers.core import Activation
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import multi_gpu_model


def model_name():
    return 'ldnss'


def encoder_model(model):
    dropout_rate = 0.5

    # Input
    model.add(Layer(input_shape=(320, 320, 3)))

    # Layer E1
    # E1_1
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=dropout_rate))
    model.add(LeakyReLU(alpha=0.2))
    # E1_2
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=dropout_rate))
    model.add(LeakyReLU(alpha=0.2))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Result x160

    # Layer E2
    # E2_1
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=dropout_rate))
    model.add(LeakyReLU(alpha=0.2))
    # E2_2
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=dropout_rate))
    model.add(LeakyReLU(alpha=0.2))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Result x80

    # Layer E3
    # E3_1
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=dropout_rate))
    model.add(LeakyReLU(alpha=0.2))
    # E3_2
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=dropout_rate))
    model.add(LeakyReLU(alpha=0.2))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Result x40

    # Layer E4
    # E4_1
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=dropout_rate))
    model.add(LeakyReLU(alpha=0.2))
    # E4_2
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=dropout_rate))
    model.add(LeakyReLU(alpha=0.2))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Result x20

    # Layer E5
    # E5_1
    model.add(Conv2D(1024, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=dropout_rate))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Result x10

    # Layer E6
    # E6_1
    model.add(Conv2D(1024, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=dropout_rate))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Result x5

    # Layer E7
    model.add(Flatten())
    model.add(Dense(5000))
    model.add(LeakyReLU(alpha=0.2))

    return model


def decoder_model(model):
    dropout_rate = 0.3

    # Layer D7
    model.add(Layer(input_shape=(5000,)))
    model.add(Dense(5000))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((5, 5, -1)))
    # Result x5

    # Layer D6
    # D6_1
    model.add(Conv2DTranspose(1024, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=dropout_rate))
    model.add(LeakyReLU(alpha=0.2))
    # D6_2
    model.add(Conv2DTranspose(1024, (3, 3), padding='same', strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(rate=dropout_rate))
    model.add(LeakyReLU(alpha=0.2))
    # Result x10

    # Layer D5
    # D5_1
    model.add(Conv2DTranspose(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=dropout_rate))
    model.add(LeakyReLU(alpha=0.2))
    # D5_2
    model.add(Conv2DTranspose(512, (3, 3), padding='same', strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(rate=dropout_rate))
    model.add(LeakyReLU(alpha=0.2))
    # Result x20

    # Layer D4
    # D4_1
    model.add(Conv2DTranspose(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=dropout_rate))
    model.add(LeakyReLU(alpha=0.2))
    # D4_2
    model.add(Conv2DTranspose(256, (3, 3), padding='same', strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(rate=dropout_rate))
    model.add(LeakyReLU(alpha=0.2))
    # Result x40

    # Layer D3
    # D3_1
    model.add(Conv2DTranspose(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=dropout_rate))
    model.add(LeakyReLU(alpha=0.2))
    # D3_2
    model.add(Conv2DTranspose(128, (3, 3), padding='same', strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(rate=dropout_rate))
    model.add(LeakyReLU(alpha=0.2))
    # Result x80

    # Layer D2
    # D2_1
    model.add(Conv2DTranspose(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=dropout_rate))
    model.add(LeakyReLU(alpha=0.2))
    # D2_2
    model.add(Conv2DTranspose(64, (3, 3), padding='same', strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(rate=dropout_rate))
    model.add(LeakyReLU(alpha=0.2))
    # Result x160

    # Layer D1
    model.add(Conv2DTranspose(1, (3, 3), padding='same', strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(rate=dropout_rate))
    model.add(LeakyReLU(alpha=0.2))
    # Result x320

    # model.add(LeakyReLU(alpha=0.2))
    model.add(Activation("sigmoid"))

    return model


def build_model(number_of_gpus=2):
    model = Sequential()
    model = encoder_model(model)
    model = decoder_model(model)

    # e_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    # d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)

    # optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    optim = SGD(lr=0.001, momentum=0.9, decay=0.00001, nesterov=True)

    run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)

    # e.compile(loss='binary_crossentropy', optimizer=e_optim, options=run_opts)
    # # d.trainable = True
    # d.compile(loss='binary_crossentropy', optimizer=d_optim, options=run_opts)
    model = multi_gpu_model(model, gpus=number_of_gpus)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    # add to compile metrics=['accuracy']

    return model
