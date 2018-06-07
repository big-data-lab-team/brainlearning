#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=0-03:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID

module load cuda cudnn python/3.6.3
source tensorflow/bin/activate
python ./model_operations.py --mode train --model small --batch_size 1 --steps_per_epoch 1 --epochs 30 -- ./project/ml-bet/