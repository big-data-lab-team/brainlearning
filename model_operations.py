import argparse
import datetime as dt
import importlib
import os
import time

import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model
from keras.optimizers import SGD

import pre_process

timestamp_format = '%Y-%m-%d-%H:%M:%S.%f'
round_accuracy = 4


def sorensen_dice_distance(img_1, img_2):
    numerator = np.sum(np.multiply(img_1, img_2))
    denominator = np.sum(img_1) + np.sum(img_2)
    return (2 * numerator) / denominator


def dice_loss(y_true, y_pred):
    result = 1 - sorensen_dice_distance(y_true, y_pred)
    print(result)
    return result


def experiments():
    m = np.array([[[1, 2], [3, 4]], [[21, 22], [23, 24]]], int)
    print(m.shape)
    print(m[0, :, :])

    n = np.rot90(m, axes=(0, 1))
    print(n.shape)
    print(n[0, :, :])

    x = np.rot90(m, axes=(0, 2))
    print(x.shape)
    print(x[0, :, :])

    new_array = np.concatenate((m, n, x))
    print(new_array.shape)
    print(new_array[0, :, :])

    maxs = np.amax(new_array, axis=(1, 2))
    print(maxs.shape)

    normallized = np.divide(new_array, maxs[:, None, None])
    print(normallized.shape)


def train(model_imported=None, data_generator_instance=None, continue_training=False, model_dir=None,
          model_file='model.hdf5', epochs=10):
    if continue_training:
        print("Continue Train")
        model_to_train = load_model('./model/' + model_dir + model_file, custom_objects={'dice_loss': dice_loss})
        print(model_to_train.summary())
        save_path = './model/' + model_dir
        graph_path = './graph/' + model_dir
    else:
        print("Train")
        model_to_train = model_imported.build_model()
        optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        model_to_train.compile(loss='binary_crossentropy', optimizer=optim)
        save_path = './model/' + model_imported.model_name() + '/'
        graph_path = './graph/' + model_imported.model_name() + '/'

    # x, y = pre_process.pre_process()
    #
    # x_train = x[0:900, ]
    # y_train = y[0:900, ]
    # x_train = x_train.reshape((900, 320, 320, 1))
    # y_train = y_train.reshape((900, 320, 320, 1))
    #
    # x_test = x[900:, ]
    # y_test = y[900:, ]
    # x_test = x_test.reshape((60, 320, 320, 1))
    # y_test = y_test.reshape((60, 320, 320, 1))

    if not os.path.exists(graph_path):
        os.makedirs(graph_path)

    tb_call_back = TensorBoard(log_dir=graph_path, histogram_freq=0, write_graph=True, write_images=True)

    if not os.path.exists(save_path + 'epoch/'):
        os.makedirs(save_path + 'epoch/')

    file_path = save_path + 'epoch/' + dt.datetime.fromtimestamp(time.time()).strftime(
        timestamp_format) + '-weights-improvement-{epoch:02d}.hdf5'

    checkpoint = ModelCheckpoint(file_path, monitor='loss', verbose=1, save_best_only=True, mode='min')

    # model_to_train.fit(x_train,
    #                    y_train,
    #                    validation_split=20,
    #                    verbose=1,
    #                    epochs=epochs,
    #                    batch_size=batch_size,
    #                    callbacks=[tb_call_back, checkpoint])

    model_to_train.fit_generator(generator=data_generator_instance,
                                 steps_per_epoch=1,
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=[tb_call_back, checkpoint],
                                 validation_data=data_generator_instance,
                                 validation_steps=1,
                                 shuffle=False
                                 )

    print('Evaluation:')
    # acc = model_to_train.evaluate(x_test, y_test, batch_size=10)
    acc = model_to_train.evaluate_generator(
        generator=data_generator_instance,
        batch_size=10)
    print('Evaluation result:', acc)

    save_model(model_to_train, acc, save_path)


def save_model(model_to_save, accuracy, model_dir):
    # Save Model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    acc_rounded = round(accuracy, round_accuracy)
    timestamp = dt.datetime.fromtimestamp(time.time()).strftime(timestamp_format)

    # serialise model, weights and biases, optimizer state.
    model_to_save.save(model_dir + 'model.hdf5')
    model_to_save.save(model_dir + timestamp + '-acc_' + str(acc_rounded) + '-model.hdf5')

    # serialize model to JSON
    model_json = model_to_save.to_json()
    with open(model_dir + 'model.json', 'w') as json_file:
        json_file.write(model_json)

    with open(model_dir + timestamp + '-acc_' + str(acc_rounded) + '-model.json', 'w') as json_file:
        json_file.write(model_json)

    # serialize weights to .h5
    model_to_save.save_weights(model_dir + timestamp + '-acc_' + str(acc_rounded) + '-model.h5')
    print("Saved model to disk")


def predict(model_dir, model_file='model.hdf5'):
    print("Predict")
    model_to_predict = load_model('./model/' + str(model_dir) + str(model_file),
                                  custom_objects={'dice_loss': dice_loss})
    print(model_to_predict.summary())

    x, y = pre_process.pre_process()
    x_predict = x[128]
    x_predict = x_predict.reshape((1, 320, 320, 1))

    print('Predict:')
    pred = model_to_predict.predict(x_predict, verbose=1)

    timestamp = dt.datetime.fromtimestamp(time.time()).strftime(timestamp_format)
    result_dir = './model/' + model_dir + 'result/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    print('Predict result:', pred)
    np.savetxt(result_dir + timestamp + '-result.txt', pred.reshape((320, 320)), fmt='%.2f')
    np.savetxt(result_dir + timestamp + '-mri.txt', x[128], fmt='%.2f')
    np.savetxt(result_dir + timestamp + '-brain-mask.txt', y[128], fmt='%d')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--graph_dir", type=str, default='./graph/')
    parser.add_argument("--model_dir", type=str, default='./model/')
    parser.add_argument("--model_file", type=str, default='model.hdf5')
    parser.add_argument("--result_dir", type=str, default='./result/')
    # parser.add_argument("--nice", dest="nice", action="store_true")
    # parser.set_defaults(nice=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        model = importlib.import_module('model.' + args.model + '.model')
        data_generator = importlib.import_module('model.' + args.model + '.data_generator')
        data_generator_inst = data_generator.DataGenerator(
            dir_path='../ml-bet/',
            file_pattern='*.nii.gz',
            distinguish_pattern='_brain',
            batch_size=args.batch_size,
            dim=320,
            n_channels=1,
            shuffle=False)
        train(model,
              data_generator_inst,
              continue_training=False,
              epochs=args.epochs,
              model_dir=args.model_dir)
    elif args.mode == 'continue':
        data_generator = importlib.import_module('model.' + args.model + '.data_generator')
        data_generator_inst = data_generator.DataGenerator(
            dir_path='../ml-bet/',
            file_pattern='*.nii.gz',
            distinguish_pattern='_brain',
            batch_size=args.batch_size,
            dim=320,
            n_channels=1,
            shuffle=False)
        train(data_generator,
              continue_training=True,
              epochs=args.epochs,
              model_dir=args.model_dir,
              model_file=args.model_file)
    elif args.mode == "generate":
        predict(model_dir=args.model_dir,
                model_file=args.model_file)

    elif args.mode == "pre_process":
        data_generator = importlib.import_module('model.' + args.model + '.data_generator')
        data_generator_inst = data_generator.DataGenerator(
            dir_path='../ml-bet/',
            file_pattern='*.nii.gz',
            distinguish_pattern='_brain',
            batch_size=1,
            dim=320,
            n_channels=1,
            shuffle=False)
        processed = data_generator_inst.__getitem__(1)
        print(processed)
        # file_pairs = pre_process.get_file_pairs('../ml-bet/')
        # processed = pre_process.process_pair(file_pairs[0])

    # build_small_model()
    # build_model()
    # experiments()
    # train_small()
    # pre_process()

    # if args.mode == "train":
    #     train(BATCH_SIZE=args.batch_size)
    # elif args.mode == "generate":
    # generate(BATCH_SIZE=args.batch_size, nice=args.nice)
