# Brainlearning
Course project dedicated to an attempt to use Deep Convolutional Neural Networks to produce a brain cropping mask.

## Prerequisites
 - Python 3
 - Tensorflow
 - Keras
 - nibabel

Recommended to have a GPU to significantly decrease time of training and inseption.

## Running the operations.py script
### Run syntax
```
usage: operations.py [-h] 
                      --mode {train,continue,generate,generate_3} 
                      --model MODEL 
                     [--verbose VERBOSE] 
                     [--graph_dir GRAPH_DIR]
                     [--epochs EPOCHS] 
                     [--save_each_epochs SAVE_EACH_EPOCHS]
                     [--save_each_epochs_dir SAVE_EACH_EPOCHS_DIR]
                     [--steps_per_epoch STEPS_PER_EPOCH]
                     [--validation_steps VALIDATION_STEPS]
                     [--batch_size BATCH_SIZE] 
                     [--images_dir_path IMAGES_DIR_PATH]
                     [--model_dir MODEL_DIR] 
                     [--model_file MODEL_FILE]
                     [--result_dir RESULT_DIR] 
                     [--model_x MODEL_X]
                     [--model_y MODEL_Y] 
                     [--model_z MODEL_Z] 
                     --file_to_process FILE_TO_PROCESS
```
### Parameters
| mode                      | parameter             | Vales / example                           | description |
|:-------------------------:|-----------------------|-------------------------------------------|-------------|
| all                       |--mode                 | (train, continue, generate, generate_3)   | Mode of the program. |
| all                       |--model                | "f_1i_mal_x"                              | Model Name.|
| all                       |--verbose              | Default 1. (0, 1, 2)                      | Verbosity of logging.|
| train, continue           |--graph_dir            | Default "graph/"                          | Directory to store Tensorflow Graph info.|
| train, continue           |--epochs               | Default 10                                | Number of Epochs.|
| train, continue           |--save_each_epochs     | Default 1                                 | Intermediate model Save after # of epochs if accuracy improved.|
| train, continue           |--save_each_epochs_dir | Default "epoch/"                          | Directory to store intermediate model.|
| train, continue           |--steps_per_epoch      | Default 1                                 | Number of data draws per epochs.|
| train, continue           |--validation_steps     | Default 1                                 | Number of data draws on validation.|
| train, continue           |--batch_size           | Default 10                                | Batch size.|
| train, continue, generate |--images_dir_path      | Default "../ml-bet/"                      | Path to Train and Validation file directories.|
| train, continue, generate |--model_dir            | Default "./model/"                        | Directory of the model.|
| train, continue, generate |--model_file           | Default "model.hdf"                       | The model file name.|
| generate                  |--result_dir           | Default "./result/"                       | Directory to store result.|
| generate_3                |--model_x              | "/model/f_1i_mal_x/model.hdf5"            | Model X file *.hdf5|
| generate_3                |--model_y              | "/model/f_1i_mal_y/model.hdf5"            | Model Y file *.hdf5|
| generate_3                |--model_z              | "/model/f_1i_mal_z/model.hdf5"            | Model Z file *.hdf5|
| generate_3                |--file_to_process      | "/training-set/106016_3T_T1w_MPR1.nii.gz" | File nii or nii.gz to generate mask.|

## Run Examples
### Training
```
model=f_1i_mal_x
python operations.py \ 
    --mode train \
    --model $model \
    --batch_size 4 \
    --steps_per_epoch 1 \
    --epochs 250 \
    --save_each_epochs 20 \
    --save_each_epochs_dir $HOME/scratch/model/$model/ \
    --images_dir_path ../project/ml-bet/
```
### Continue
```
model=f_1i_mal_x
python operations.py \
    --mode continue 
    --model $model 
    --model_dir $model/ 
    --batch_size 2 
    --steps_per_epoch 1 
    --epochs 200 
    --save_each_epochs 20 
    --save_each_epochs_dir $HOME/scratch/model/$model/ 
    --images_dir_path ../project/ml-bet/ 
```
### Generate
```
model=f_1i_mal_x
python operations.py \
    --mode generate 
    --model $model 
    --model_dir $model/ 
    --images_dir_path ../project/ml-bet/
```
### Generate 3
Generate a mask using 3 models for X, Y, Z axis
```
python operations.py \ 
    --mode generate_3 \
    --model_x model/f_1i_mal_x/model.hdf5 \
    --model_y model/f_1i_mal_y/model.hdf5 \
    --model_z model/f_1i_mal_z/model.hdf5 \
    --file_to_process ../training-set/106016_3T_T1w_MPR1.nii.gz \
    --result_dir model/result_3/ \

```