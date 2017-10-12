#!/bin/bash -l
# Name your job (used in the PBS output file names)
#PBS -N GPUTEST
# Specify the gpuq queue
#PBS -q gpuq
# Specify the number of gpus
#PBS -l nodes=1:ppn=1:gpus=1
# Specify the gpu feature
#PBS -l feature=gpu
# Specify your resource account (use qr command to determine)
#PBS -A Physics
# Specify how much time you think the job will run
#PBS -l walltime=00:15:00
# Have the job send you email when the job ends or aborts
#PBS -M john.doe@dartmouth.edu
#PBS -m ea
# Join error and standard output into one file
#PBS -j oe 
# Change to the directory that the job was submitted from
cd $PBS_O_WORKDIR
# Parse the PBS_GPUFILE to determine which GPU you have been assigned
# and unset CUDA_VISIBLE_DEVICES
gpuNum=`cat $PBS_GPUFILE | sed -e 's/.*-gpu//g'`
unset CUDA_VISIBLE_DEVICES
# if using PyCUDA set the CUDA_DEVICE environment variable
export CUDA_DEVICE=$gpuNum
# Pass the GPU number as an argument to your program
program_name $gpuNum
exit 0