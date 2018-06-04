from keras.engine import Layer
from keras.layers import Dense, LeakyReLU, Dropout
from keras.layers import Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.models import Sequential
from keras.optimizers import SGD


def model_name():
    return 'small'


def encoder_model():
    model = Sequential()
    # Layer E1
    model.add(Layer(input_shape=(320, 320, 1)))
    model.add(Conv2D(20, (11, 11), padding='same', strides=(2, 2)))
    # model.add(LeakyReLU(alpha=0.2))
    model.add(Activation('tanh'))
    model.add(Dropout(rate=0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Result 82x82x64

    # Layer E2
    model.add(Conv2D(40, (5, 5), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Result 41x41x128

    # Layer E3
    model.add(Conv2D(80, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Result 20x20x512

    # Layer E4
    model.add(Flatten())
    model.add(Dense(2048))
    model.add(Activation("sigmoid"))

    return model


# def decoder_model():
#     model = Sequential()
#
#     # Layer D1
#     model.add(Layer(input_shape=(2048,)))
#     model.add(Dense(32000))
#     model.add(Activation("sigmoid"))
#     model.add(Reshape((20, 20, -1)))
#
#     # Layer D2
#     model.add(Conv2DTranspose(80, (3, 3), padding='same'))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(Dropout(rate=0.5))
#     model.add(UpSampling2D(size=(2, 2)))
#
#     # Layer D3
#     model.add(Conv2DTranspose(40, (5, 5), padding='same'))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(Dropout(rate=0.5))
#     model.add(UpSampling2D(size=(2, 2)))
#
#     # Layer D4
#     model.add(Conv2DTranspose(20, (11, 11), padding='same'))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(Dropout(rate=0.5))
#     model.add(UpSampling2D(size=(2, 2)))
#
#     # Layer D5
#     model.add(Conv2DTranspose(1, (5, 5), padding='same'))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(Dropout(rate=0.5))
#     model.add(UpSampling2D(size=(2, 2)))
#
#     # model.add(Softmax())
#     # model.add(LeakyReLU(alpha=0.2))
#     model.add(Activation("sigmoid"))
#
#     return model


def decoder_model():
    model = Sequential()

    # Layer D1
    model.add(Layer(input_shape=(2048,)))
    model.add(Dense(32000))
    model.add(Activation("sigmoid"))
    model.add(Reshape((20, 20, -1)))

    # Layer D2
    model.add(Conv2DTranspose(80, (3, 3), padding='same', strides=(2, 2)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.5))

    # Layer D3
    model.add(Conv2DTranspose(40, (5, 5), padding='same', strides=(2, 2)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.5))

    # Layer D4
    model.add(Conv2DTranspose(20, (11, 11), padding='same', strides=(2, 2)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.5))

    # Layer D5
    model.add(Conv2DTranspose(1, (5, 5), padding='same', strides=(2, 2)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.5))

    # model.add(Softmax())

    # model.add(LeakyReLU(alpha=0.2))
    model.add(Activation("sigmoid"))

    return model


def combine_models(e, d):
    model = Sequential()
    model.add(e)
    model.add(d)
    return model


def build_model():
    e = encoder_model()
    d = decoder_model()
    e_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)

    e.compile(loss='binary_crossentropy', optimizer=e_optim)
    print("Small Encoder Model")
    print(e.summary())

    # d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    print("Small Decoder Model")
    print(d.summary())

    full_model = combine_models(e, d)
    print("Full Small Model")
    print(full_model.summary())

    return full_model
