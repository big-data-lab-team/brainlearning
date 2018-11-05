#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=127518M       # memory per node
#SBATCH --time=0-01:00      # time (DD-HH:MM)
#SBATCH --output=%j-%N.out  # %N for node name, %j for jobID

module load cuda cudnn python/3.6.3
echo "Present working directory is $PWD"
source $HOME/tensorflow/bin/activate

# ml-bet
python $HOME/brainlearning/brainlearning/operations.py \
                                            --mode generate_3 \
                                            --model_x $HOME/brainlearning/model/f_1i_mal_x/model.hdf5 \
                                            --model_y $HOME/brainlearning/model/f_1i_mal_y/model.hdf5 \
                                            --model_z $HOME/brainlearning/model/f_1i_mal_z/model.hdf5 \
                                            --file_to_process $HOME/project/ml-bet/training-set/103515_3T_T1w_MPR1.nii.gz \
                                            --result_dir $HOME/brainlearning/model/result_3/ \
   \
python $HOME/brainlearning/brainlearning/operations.py \
                                            --mode generate_3 \
                                            --model_x $HOME/brainlearning/model/f_1i_mal_x/model.hdf5 \
                                            --model_y $HOME/brainlearning/model/f_1i_mal_y/model.hdf5 \
                                            --model_z $HOME/brainlearning/model/f_1i_mal_z/model.hdf5 \
                                            --file_to_process $HOME/project/ml-bet/training-set/103818_3T_T1w_MPR1.nii.gz \
                                            --result_dir $HOME/brainlearning/model/result_3/ \
  \
python $HOME/brainlearning/brainlearning/operations.py \
                                            --mode generate_3 \
                                            --model_x $HOME/brainlearning/model/f_1i_mal_x/model.hdf5 \
                                            --model_y $HOME/brainlearning/model/f_1i_mal_y/model.hdf5 \
                                            --model_z $HOME/brainlearning/model/f_1i_mal_z/model.hdf5 \
                                            --file_to_process $HOME/project/ml-bet/training-set/106016_3T_T1w_MPR1.nii.gz \
                                            --result_dir $HOME/brainlearning/model/result_3/ \

python $HOME/brainlearning/brainlearning/operations.py \
                                            --mode generate_3 \
                                            --model_x $HOME/brainlearning/model/f_1i_mal_x/model.hdf5 \
                                            --model_y $HOME/brainlearning/model/f_1i_mal_y/model.hdf5 \
                                            --model_z $HOME/brainlearning/model/f_1i_mal_z/model.hdf5 \
                                            --file_to_process $HOME/project/ml-bet/training-set/106319_3T_T1w_MPR1.nii.gz \
                                            --result_dir $HOME/brainlearning/model/result_3/ \
\
python $HOME/brainlearning/brainlearning/operations.py \
                                            --mode generate_3 \
                                            --model_x $HOME/brainlearning/model/f_1i_mal_x/model.hdf5 \
                                            --model_y $HOME/brainlearning/model/f_1i_mal_y/model.hdf5 \
                                            --model_z $HOME/brainlearning/model/f_1i_mal_z/model.hdf5 \
                                            --file_to_process $HOME/project/ml-bet/training-set/106521_3T_T1w_MPR1.nii.gz \
                                            --result_dir $HOME/brainlearning/model/result_3/ \
 \