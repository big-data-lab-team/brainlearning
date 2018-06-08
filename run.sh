#!/bin/bash
#SBATCH --gres=gpu:2        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=0-03:00      # time (DD-HH:MM)
#SBATCH --output=%j-%N.out  # %N for node name, %j for jobID

module load cuda cudnn python/3.6.3
echo "Present working directory is $PWD"
source $HOME/tensorflow/bin/activate
python $HOME/brainlearning/model_operations.py --mode train --model small --batch_size 1 --steps_per_epoch 1 --epochs 30 --images_dir_path ../project/ml-bet/
