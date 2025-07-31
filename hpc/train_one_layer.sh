#!/bin/sh
### General options
### -- set the job Name --
#SBATCH -J train_one_layer
### -- set the job array --
#SBATCH --array=1-15
### -- ask for number of cores (default: 1) --
#SBATCH -n 1
### -- set walltime limit: j-h:m:s
#SBATCH --time 1:0:0
### -- Specify the output and error file. %A_%a is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#SBATCH -o /beegfs/lfontain/unsupervised-pcn/outputs/logs/train_one_layer_%A_%a.out
#SBATCH -e /beegfs/lfontain/unsupervised-pcn/outputs/logs/train_one_layer_%A_%a.err
# -- end of Slurm options --


# Load modules
module load build/conda/4.10
module load compiler/cuda/12.3

conda activate torch_env

N_hidden=(10 20 30 40 50 60 70 80 90 100 110 120 130 140 150)
n_hidden=${N_hidden[$SLURM_ARRAY_TASK_ID - 1]}
srun python /beegfs/lfontain/unsupervised-pcn/src/pcn/train_one_layer.py --n_hidden=$n_hidden