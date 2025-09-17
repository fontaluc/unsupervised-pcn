#!/bin/sh
### General options
### -- set the job Name --
#SBATCH -J tune_first_layer
### -- set the job array --
#SBATCH --array=1-12
### -- ask for number of cores (default: 1) --
#SBATCH -n 1
### -- set walltime limit: j-h:m:s
#SBATCH --time 10:0:0
### -- Specify the output and error file. %A_%a is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#SBATCH -o /beegfs/lfontain/unsupervised-pcn/outputs/logs/tune_first_layer_%A_%a.out
#SBATCH -e /beegfs/lfontain/unsupervised-pcn/outputs/logs/tune_first_layer_%A_%a.err
# -- end of Slurm options --


# Load modules
module load build/conda/4.10
module load compiler/cuda/12.3

conda activate torch_env

N_hidden=(600 550 500 450 400 350 300 250 200 150 100 50)
n_hidden=${N_hidden[$SLURM_ARRAY_TASK_ID - 1]}
srun python /beegfs/lfontain/unsupervised-pcn/src/pcn/eval_one_layer.py --n_hidden=$n_hidden