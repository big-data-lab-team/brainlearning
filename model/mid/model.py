from keras.engine import Layer
from keras.layers import Dense, LeakyReLU, Dropout
from keras.layers import Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.models import Sequential
from keras.optimizers import SGD
import tensorflow as tf


def model_name():
    return 'mid'


def encoder_model(model):
    # Input
    model.add(Layer(input_shape=(320, 320, 1)))

    # Layer E1
    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(Activation('tanh'))
    model.add(Dropout(rate=0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Result 82x82x64

    # Layer E2
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.5))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Result 41x41x128

    # Layer E3
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Result 20x20x512

    # Layer E4
    model.add(Flatten())
    model.add(Dense(2048))
    model.add(Activation("sigmoid"))

    return model


def decoder_model(model):
    model.add(Layer(input_shape=(2048,)))

    # Layer D1
    model.add(Dense(32000))
    model.add(Activation("sigmoid"))
    model.add(Reshape((20, 20, -1)))

    # Layer D2
    model.add(Conv2DTranspose(64, (3, 3), padding='same', strides=(2, 2)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.5))

    # Layer D3
    model.add(Conv2DTranspose(32, (3, 3), padding='same', strides=(2, 2)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.5))

    # Layer D4
    model.add(Conv2DTranspose(16, (3, 3), padding='same', strides=(2, 2)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.5))

    # Layer D5
    model.add(Conv2DTranspose(1, (3, 3), padding='same', strides=(2, 2)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.5))

    model.add(Activation("sigmoid"))

    return model


def build_model():
    model = Sequential()
    model = encoder_model(model)
    model = decoder_model(model)

    # e_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    # d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)

    optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)

    run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)

    # e.compile(loss='binary_crossentropy', optimizer=e_optim, options=run_opts)
    # # d.trainable = True
    # d.compile(loss='binary_crossentropy', optimizer=d_optim, options=run_opts)

    model.compile(loss='binary_crossentropy', optimizer='adam')
    # add to compile metrics=['accuracy']

    return model
