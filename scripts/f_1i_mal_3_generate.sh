#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=127518M       # memory per node
#SBATCH --time=0-01:00      # time (DD-HH:MM)
#SBATCH --output=%j-%N.out  # %N for node name, %j for jobID

module load cuda cudnn python/3.6.3
echo "Present working directory is $PWD"
source $HOME/tensorflow/bin/activate

# BMB_1
python $HOME/brainlearning/brainlearning/operations.py \
                                            --mode generate_3 \
                                            --model_x $HOME/brainlearning/model/f_1i_mal_x/model.hdf5 \
                                            --model_y $HOME/brainlearning/model/f_1i_mal_y/model.hdf5 \
                                            --model_z $HOME/brainlearning/model/f_1i_mal_z/model.hdf5 \
                                            --file_to_process $HOME/scratch/test-set/datasets.datalad.org/corr/RawDataBIDS/BMB_1/sub-0003002/ses-1/anat/sub-0003002_ses-1_run-1_T1w.nii.gz \
                                            --result_dir $HOME/brainlearning/model/result_3/ \
                                            --images_dir_path ../project/ml-bet/

python $HOME/brainlearning/brainlearning/operations.py \
                                            --mode generate_3 \
                                            --model_x $HOME/brainlearning/model/f_1i_mal_x/model.hdf5 \
                                            --model_y $HOME/brainlearning/model/f_1i_mal_y/model.hdf5 \
                                            --model_z $HOME/brainlearning/model/f_1i_mal_z/model.hdf5 \
                                            --file_to_process $HOME/scratch/test-set/datasets.datalad.org/corr/RawDataBIDS/BMB_1/sub-0003004/ses-1/anat/sub-0003004_ses-1_run-1_T1w.nii.gz \
                                            --result_dir $HOME/brainlearning/model/result_3/ \
                                            --images_dir_path ../project/ml-bet/

# BNU_1
python $HOME/brainlearning/brainlearning/operations.py \
                                            --mode generate_3 \
                                            --model_x $HOME/brainlearning/model/f_1i_mal_x/model.hdf5 \
                                            --model_y $HOME/brainlearning/model/f_1i_mal_y/model.hdf5 \
                                            --model_z $HOME/brainlearning/model/f_1i_mal_z/model.hdf5 \
                                            --file_to_process $HOME/scratch/test-set/datasets.datalad.org/corr/RawDataBIDS/BNU_1/sub-0025864/ses-1/anat/sub-0025864_ses-1_run-1_T1w.nii.gz \
                                            --result_dir $HOME/brainlearning/model/result_3/ \
                                            --images_dir_path ../project/ml-bet/

python $HOME/brainlearning/brainlearning/operations.py \
                                            --mode generate_3 \
                                            --model_x $HOME/brainlearning/model/f_1i_mal_x/model.hdf5 \
                                            --model_y $HOME/brainlearning/model/f_1i_mal_y/model.hdf5 \
                                            --model_z $HOME/brainlearning/model/f_1i_mal_z/model.hdf5 \
                                            --file_to_process $HOME/scratch/test-set/datasets.datalad.org/corr/RawDataBIDS/BNU_1/sub-0025865/ses-1/anat/sub-0025865_ses-1_run-1_T1w.nii.gz \
                                            --result_dir $HOME/brainlearning/model/result_3/ \
                                            --images_dir_path ../project/ml-bet/

python $HOME/brainlearning/brainlearning/operations.py \
                                            --mode generate_3 \
                                            --model_x $HOME/brainlearning/model/f_1i_mal_x/model.hdf5 \
                                            --model_y $HOME/brainlearning/model/f_1i_mal_y/model.hdf5 \
                                            --model_z $HOME/brainlearning/model/f_1i_mal_z/model.hdf5 \
                                            --file_to_process $HOME/scratch/test-set/datasets.datalad.org/corr/RawDataBIDS/BNU_1/sub-0025866/ses-1/anat/sub-0025866_ses-1_run-1_T1w.nii.gz \
                                            --result_dir $HOME/brainlearning/model/result_3/ \
                                            --images_dir_path ../project/ml-bet/

# DC_1
python $HOME/brainlearning/brainlearning/operations.py \
                                            --mode generate_3 \
                                            --model_x $HOME/brainlearning/model/f_1i_mal_x/model.hdf5 \
                                            --model_y $HOME/brainlearning/model/f_1i_mal_y/model.hdf5 \
                                            --model_z $HOME/brainlearning/model/f_1i_mal_z/model.hdf5 \
                                            --file_to_process $HOME/scratch/test-set/datasets.datalad.org/corr/RawDataBIDS/DC_1/sub-0027306/ses-1/anat/sub-0027306_ses-1_run-1_T1w.nii.gz \
                                            --result_dir $HOME/brainlearning/model/result_3/ \
                                            --images_dir_path ../project/ml-bet/

python $HOME/brainlearning/brainlearning/operations.py \
                                            --mode generate_3 \
                                            --model_x $HOME/brainlearning/model/f_1i_mal_x/model.hdf5 \
                                            --model_y $HOME/brainlearning/model/f_1i_mal_y/model.hdf5 \
                                            --model_z $HOME/brainlearning/model/f_1i_mal_z/model.hdf5 \
                                            --file_to_process $HOME/scratch/test-set/datasets.datalad.org/corr/RawDataBIDS/DC_1/sub-0027307/ses-1/anat/sub-0027307_ses-1_run-1_T1w.nii.gz \
                                            --result_dir $HOME/brainlearning/model/result_3/ \
                                            --images_dir_path ../project/ml-bet/

python $HOME/brainlearning/brainlearning/operations.py \
                                            --mode generate_3 \
                                            --model_x $HOME/brainlearning/model/f_1i_mal_x/model.hdf5 \
                                            --model_y $HOME/brainlearning/model/f_1i_mal_y/model.hdf5 \
                                            --model_z $HOME/brainlearning/model/f_1i_mal_z/model.hdf5 \
                                            --file_to_process $HOME/scratch/test-set/datasets.datalad.org/corr/RawDataBIDS/DC_1/sub-0027308/ses-1/anat/sub-0027308_ses-1_run-1_T1w.nii.gz \
                                            --result_dir $HOME/brainlearning/model/result_3/ \
                                            --images_dir_path ../project/ml-bet/
