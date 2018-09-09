#!/bin/bash
#SBATCH --gres=gpu:2        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=127518M       # memory per node
#SBATCH --time=0-12:00      # time (DD-HH:MM)
#SBATCH --output=%j-%N.out  # %N for node name, %j for jobID

model=f_1i_mal_y

module load cuda cudnn python/3.6.3
echo "Present working directory is $PWD"
source $HOME/tensorflow/bin/activate

for i in {1..4}
do
    echo "------------------------------------------------------------------------------"
    echo "$model 500-1 - $i"
    python $HOME/brainlearning/brainlearning/operations.py --mode continue --model $model --model_dir $model/ --batch_size 2 --n_channels 1 --steps_per_epoch 1 --epochs 500 --save_each_epochs 20 --save_each_epochs_dir $HOME/scratch/model/$model/ --images_dir_path ../project/ml-bet/ --verbose 2
done

for i in {1..4}
do
    echo "------------------------------------------------------------------------------"
    echo "$model 200-1 - $i"
    python $HOME/brainlearning/brainlearning/operations.py --mode continue --model $model --model_dir $model/ --batch_size 2 --n_channels 1 --steps_per_epoch 1 --epochs 200 --save_each_epochs 20 --save_each_epochs_dir $HOME/scratch/model/$model/ --images_dir_path ../project/ml-bet/ --verbose 2
done

for i in {1..4}
do
    echo "------------------------------------------------------------------------------"
    echo "$model 100 - $i"
    python $HOME/brainlearning/brainlearning/operations.py --mode continue --model $model --model_dir $model/ --batch_size 2 --n_channels 1 --steps_per_epoch 1 --epochs 100 --save_each_epochs 20 --save_each_epochs_dir $HOME/scratch/model/$model/ --images_dir_path ../project/ml-bet/ --verbose 2
done

for i in {1..2}
do
    echo "------------------------------------------------------------------------------"
    echo "$model 500-2 - $i"
    python $HOME/brainlearning/brainlearning/operations.py --mode continue --model $model --model_dir $model/ --batch_size 2 --n_channels 1 --steps_per_epoch 1 --epochs 500 --save_each_epochs 20 --save_each_epochs_dir $HOME/scratch/model/$model/ --images_dir_path ../project/ml-bet/ --verbose 2
done

for i in {1..2}
do
    echo "------------------------------------------------------------------------------"
    echo "$model 200-1 - $i"
    python $HOME/brainlearning/brainlearning/operations.py --mode continue --model $model --model_dir $model/ --batch_size 2 --n_channels 1 --steps_per_epoch 1 --epochs 200 --save_each_epochs 20 --save_each_epochs_dir $HOME/scratch/model/$model/ --images_dir_path ../project/ml-bet/ --verbose 2
done

for i in {1..6}
do
    echo "------------------------------------------------------------------------------"
    echo "$model 100 - $i"
    python $HOME/brainlearning/brainlearning/operations.py --mode continue --model $model --model_dir $model/ --batch_size 2 --n_channels 1 --steps_per_epoch 1 --epochs 100 --save_each_epochs 20 --save_each_epochs_dir $HOME/scratch/model/$model/ --images_dir_path ../project/ml-bet/ --verbose 2
done
