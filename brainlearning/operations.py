import argparse
import datetime as dt
import importlib
import os
import time

import numpy as np
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.losses import binary_crossentropy
from keras.models import load_model

timestamp_format = '%Y-%m-%d-%H:%M:%S.%f'
round_accuracy = 4


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)


def train(model_imported=None,
          data_generator_instance=None,
          continue_training=False,
          model_dir=None,
          model_file='model.hdf5',
          epochs=10,
          steps_per_epoch=1,
          validation_steps=1,
          verbose=1,
          save_each_epochs=1,
          save_each_epochs_dir=None):

    K.clear_session()

    if continue_training:
        print("Continue Train")
        model_to_train = load_model('./model/' + model_dir + model_file,
                                    custom_objects={'dice_coef': dice_coef,
                                                    'bce_dice_loss': bce_dice_loss,
                                                    'dice_coef_loss': dice_coef_loss})
        save_path = './model/' + model_dir
        graph_path = './graph/' + model_dir
    else:
        print("Train")
        model_to_train = model_imported.build_model()
        save_path = './model/' + model_imported.model_name() + '/'
        graph_path = './graph/' + model_imported.model_name() + '/'

    print('\n')
    print('Model ', model_imported.model_name(), ' summary:')
    print(model_to_train.summary())

    if not os.path.exists(graph_path):
        os.makedirs(graph_path)
    tb_call_back = TensorBoard(log_dir=graph_path, histogram_freq=0, write_graph=True, write_images=True)

    if save_each_epochs_dir is None:
        save_each_epochs_dir = save_path + 'epoch/'
    if not os.path.exists(save_each_epochs_dir):
        os.makedirs(save_each_epochs_dir)
    file_path = save_each_epochs_dir + dt.datetime.fromtimestamp(time.time()).strftime(
        timestamp_format) + '-weights-improvement-{epoch:02d}.hdf5'

    checkpoint = ModelCheckpoint(file_path,
                                 monitor='loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',
                                 period=save_each_epochs)

    model_to_train.fit_generator(generator=data_generator_instance,
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=epochs,
                                 verbose=verbose,
                                 callbacks=[tb_call_back, checkpoint],
                                 validation_data=data_generator_instance,
                                 validation_steps=validation_steps,
                                 shuffle=False
                                 )

    print('\n')
    print('==================================================================================================')
    print('Evaluation of ', model_imported.model_name(), ':')
    acc = model_to_train.evaluate_generator(generator=data_generator_instance)
    print('Evaluation result:', acc)

    print('\n')
    print('==================================================================================================')
    print('Saving final ', model_imported.model_name(), ' Model')
    save_model(model_to_train, acc, save_path)
    print('Model Saved Successfully')


def save_model(model_to_save, model_dir):
    # Save Model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # acc_rounded = round(accuracy[-1], round_accuracy)
    timestamp = dt.datetime.fromtimestamp(time.time()).strftime(timestamp_format)

    # serialise model, weights and biases, optimizer state.
    model_to_save.save(model_dir + 'model.hdf5')
    # model_to_save.save(model_dir + timestamp + '-acc_' + str(acc_rounded) + '-model.hdf5')
    model_to_save.save(model_dir + timestamp + '-model.hdf5')

    # serialize model to JSON
    model_json = model_to_save.to_json()
    with open(model_dir + 'model.json', 'w') as json_file:
        json_file.write(model_json)

    # with open(model_dir + timestamp + '-acc_' + str(acc_rounded) + '-model.json', 'w') as json_file:
    #     json_file.write(model_json)
    #
    # # serialize weights to .h5
    # model_to_save.save_weights(model_dir + timestamp + '-acc_' + str(acc_rounded) + '-model.h5')
    print("Saved model to disk")


def generate(
        model_dir,
        model_file='model.hdf5',
        data_generator_instance=None,
        verbose=1):
    print("Predict")
    model_to_predict = load_model('./model/' + str(model_dir) + str(model_file),
                                  custom_objects={'bce_dice_loss': bce_dice_loss,
                                                  'dice_coef': dice_coef,
                                                  'dice_coef_loss': dice_coef_loss})
    print(model_to_predict.summary())

    x, y, z, file_name = data_generator_instance.get_file()

    print('Predict:')
    pred_x = model_to_predict.predict(x, verbose=verbose)
    pred_y = model_to_predict.predict(y, verbose=verbose)
    pred_z = model_to_predict.predict(z, verbose=verbose)

    pred_x = np.reshape(pred_x, (320, 320, 320))
    pred_x = np.rot90(pred_x, k=-1, axes=(0, 2))
    pred_x = pred_x[32:288, :, :]

    pred_y = np.reshape(pred_y, (320, 320, 320))
    pred_y = np.rot90(pred_y, k=-1, axes=(0, 1))
    pred_y = pred_y[32:288, :, :]

    pred_z = np.reshape(pred_z, (320, 320, 320))
    pred_z = pred_z[32:288, :, :]

    timestamp = dt.datetime.fromtimestamp(time.time()).strftime(timestamp_format)
    result_dir = './model/' + model_dir + 'result/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Save as is
    data_generator_instance.save_to_file(pred_x, file_name,
                                         result_dir + file_name.split('/')[-1] + '-' + timestamp + '-result-x.nii.gz')
    data_generator_instance.save_to_file(pred_y, file_name,
                                         result_dir + file_name.split('/')[-1] + '-' + timestamp + '-result-y.nii.gz')
    data_generator_instance.save_to_file(pred_z, file_name,
                                         result_dir + file_name.split('/')[-1] + '-' + timestamp + '-result-z.nii.gz')
    # Save average
    data_generator_instance.save_to_file(((pred_x + pred_y + pred_z) / 3), file_name,
                                         result_dir + file_name.split('/')[-1] + '-' + timestamp + '-result-xyz.nii.gz')


def generate_3(
        model_x,
        model_y,
        model_z,
        result_dir,
        file_to_process=None,
        data_generator_instance=None,
        verbose=1):
    print('\n')
    print('==================================================================================================')
    print("Predict")
    model_to_predict_x = load_model(str(model_x),
                                    custom_objects={'dice_coef': dice_coef,
                                                    'bce_dice_loss': bce_dice_loss,
                                                    'dice_coef_loss': dice_coef_loss})
    model_to_predict_y = load_model(str(model_y),
                                    custom_objects={'dice_coef': dice_coef,
                                                    'bce_dice_loss': bce_dice_loss,
                                                    'dice_coef_loss': dice_coef_loss})
    model_to_predict_z = load_model(str(model_z),
                                    custom_objects={'dice_coef': dice_coef,
                                                    'bce_dice_loss': bce_dice_loss,
                                                    'dice_coef_loss': dice_coef_loss})

    x, y, z, mri_file_name, padding = data_generator_instance.get_file(file_to_process)

    print('Predict:')
    pred_x = model_to_predict_x.predict(x, verbose=verbose)
    pred_y = model_to_predict_y.predict(y, verbose=verbose)
    pred_z = model_to_predict_z.predict(z, verbose=verbose)

    print('Process results:')
    pred_x = data_generator_instance.process_x(pred_x, padding)
    print('pred_x shape = ', pred_x.shape)
    pred_y = data_generator_instance.process_y(pred_y, padding)
    print('pred_y shape = ', pred_y.shape)
    pred_z = data_generator_instance.process_z(pred_z, padding)
    print('pred_z shape = ', pred_z.shape)

    timestamp = dt.datetime.fromtimestamp(time.time()).strftime(timestamp_format)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Save as is
    data_generator_instance.save_to_file(pred_x,
                                         mri_file_name,
                                         result_dir + mri_file_name.split('/')[
                                             -1] + '-' + timestamp + '-result-x.nii.gz')

    data_generator_instance.save_to_file(pred_y,
                                         mri_file_name,
                                         result_dir + mri_file_name.split('/')[
                                             -1] + '-' + timestamp + '-result-y.nii.gz')

    data_generator_instance.save_to_file(pred_z,
                                         mri_file_name,
                                         result_dir + mri_file_name.split('/')[
                                             -1] + '-' + timestamp + '-result-z.nii.gz')

    # Save average
    data_generator_instance.save_to_file(((pred_x + pred_y + pred_z) / 3),
                                         mri_file_name,
                                         result_dir + mri_file_name.split('/')[
                                             -1] + '-' + timestamp + '-result-xyz.nii.gz')


def get_args():
    parser = argparse.ArgumentParser(description="Tensorflow models operations")
    parser.add_argument("--mode", type=str, required=True,
                        help="Mode of the program.",
                        choices=["train", "continue", "generate", "generate_3"])

    parser.add_argument("--model", type=str, required=True, help="Model Name.")

    parser.add_argument("--verbose", type=int, default=1, help="Verbosity of logging.")
    parser.add_argument("--graph_dir", type=str, default='./graph/', help="Directory to store Tensorflow Graph info.")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of Epochs.")
    parser.add_argument("--save_each_epochs", type=int, default=1, help="Intermediate model Save after # of epochs.")
    parser.add_argument("--save_each_epochs_dir", type=str, default=None, help="Directory to store intermediate model.")
    parser.add_argument("--steps_per_epoch", type=int, default=1, help="Number of data draws per epochs.")
    parser.add_argument("--validation_steps", type=int, default=1, help="Number of data draws on validation.")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size.")
    parser.add_argument("--n_channels", type=int, default=1, help="Number of channels/layers on input.")
    parser.add_argument("--images_dir_path", type=str, default='../ml-bet/',
                        help="Path to Train and Validation file directories.")

    parser.add_argument("--model_dir", type=str, default='./model/', help="Directory of the model.")
    parser.add_argument("--model_file", type=str, default='model.hdf5', help="The model file name.")

    # Generation specific parameters
    parser.add_argument("--result_dir", type=str, default='./result/', help="Directory to store result.")

    # Generation 3 specific parameters
    parser.add_argument("--model_x", type=str, help="Model X file *.hdf5")
    parser.add_argument("--model_y", type=str, help="Model Y file *.hdf5")
    parser.add_argument("--model_z", type=str, help="Model Z file *.hdf5")
    parser.add_argument("--file_to_process", type=str, required=True, help="File nii or nii.gz to generate mask.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.mode == "train":
        data_generator = importlib.import_module('model.' + args.model + '.data_generator')
        data_generator_inst = data_generator.DataGenerator(
            dir_path=args.images_dir_path,
            file_pattern='*.nii.gz',
            distinguish_pattern='_brain',
            batch_size=args.batch_size,
            dim=320,
            n_channels=args.n_channels,
            shuffle=False)
        model = importlib.import_module('model.' + args.model + '.model')
        train(model,
              data_generator_inst,
              continue_training=False,
              model_dir=args.model_dir,
              model_file=args.model_file,
              epochs=args.epochs,
              steps_per_epoch=args.steps_per_epoch,
              validation_steps=args.validation_steps,
              verbose=args.verbose,
              save_each_epochs=args.save_each_epochs,
              save_each_epochs_dir=args.save_each_epochs_dir)
    elif args.mode == 'continue':
        data_generator = importlib.import_module('model.' + args.model + '.data_generator')
        data_generator_inst = data_generator.DataGenerator(
            dir_path=args.images_dir_path,
            file_pattern='*.nii.gz',
            distinguish_pattern='_brain',
            batch_size=args.batch_size,
            dim=320,
            n_channels=args.n_channels,
            shuffle=False)
        model = importlib.import_module('model.' + args.model + '.model')
        train(model,
              data_generator_instance=data_generator_inst,
              continue_training=True,
              model_dir=args.model_dir,
              model_file=args.model_file,
              epochs=args.epochs,
              steps_per_epoch=args.steps_per_epoch,
              validation_steps=args.validation_steps,
              verbose=args.verbose,
              save_each_epochs=args.save_each_epochs,
              save_each_epochs_dir=args.save_each_epochs_dir)
    elif args.mode == "generate":
        data_generator = importlib.import_module('model.' + args.model + '.data_generator')
        data_generator_inst = data_generator.DataGenerator(
            dir_path=args.images_dir_path,
            file_pattern='*.nii.gz',
            distinguish_pattern='_brain',
            batch_size=args.batch_size,
            dim=320,
            n_channels=args.n_channels,
            shuffle=False)
        generate(model_dir=args.model_dir,
                 model_file=args.model_file,
                 data_generator_instance=data_generator_inst,
                 verbose=args.verbose)
    elif args.mode == "generate_3":
        data_generator = importlib.import_module('generate_3_data_generator')
        data_generator_inst = data_generator.DataGenerator(
            file_pattern='*.nii.gz',
            distinguish_pattern='_brain',
            dim=320)
        generate_3(model_x=args.model_x,
                   model_y=args.model_y,
                   model_z=args.model_z,
                   result_dir=args.result_dir,
                   file_to_process=args.file_to_process,
                   data_generator_instance=data_generator_inst,
                   verbose=args.verbose)
