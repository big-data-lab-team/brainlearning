#!/bin/bash
#SBATCH --gres=gpu:2        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=127518M       # memory per node
#SBATCH --time=0-00:30      # time (DD-HH:MM)
#SBATCH --output=%j-%N.out  # %N for node name, %j for jobID

model=f_1i_mal_x

module load cuda cudnn python/3.6.3
echo "Present working directory is $PWD"
source $HOME/tensorflow/bin/activate
python $HOME/brainlearning/brainlearning/operations.py --mode generate --model $model --model_dir $model/ --images_dir_path ../project/ml-bet/
