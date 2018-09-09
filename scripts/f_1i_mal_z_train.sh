#!/bin/bash
#SBATCH --gres=gpu:2        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=127518M       # memory per node
#SBATCH --time=0-02:00      # time (DD-HH:MM)
#SBATCH --output=%j-%N.out  # %N for node name, %j for jobID

model=f_1i_mal_z

module load cuda cudnn python/3.6.3
echo "Present working directory is $PWD"
source $HOME/tensorflow/bin/activate
python $HOME/brainlearning/brainlearning/operations.py --mode train --model $model --batch_size 4 --n_channels 1 --steps_per_epoch 1 --epochs 250 --save_each_epochs 20 --save_each_epochs_dir $HOME/scratch/model/$model/ --images_dir_path ../project/ml-bet/ --verbose 2
