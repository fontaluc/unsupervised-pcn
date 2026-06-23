#!/bin/bash
### General options
### -- set the job Name --
#SBATCH -J tune_first_layer
### -- set the job array --
#SBATCH --array=1-2
### -- ask for number of cores (default: 1) --
#SBATCH -n 1
### -- set walltime limit: j-h:m:s
#SBATCH --time 10:0:0
### -- Specify the output and error file. %A_%a is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#SBATCH -o /beegfs/lfontain/unsupervised-pcn/outputs/logs/tune_first_layer_%A_%a.out
#SBATCH -e /beegfs/lfontain/unsupervised-pcn/outputs/logs/tune_first_layer_%A_%a.err
# -- end of Slurm options --

# Initialize environment modules (module command is not loaded by default in some nodes)
source /etc/profile

# Load modules
module load build/conda/4.10
module load compiler/cuda/12.3

conda activate torch_env

N_vc=(800 700)
n_vc=${N_vc[$((SLURM_ARRAY_TASK_ID - 1))]}

srun python /beegfs/lfontain/unsupervised-pcn/src/pcn/eval_one_layer.py --dataset="mnist" --n_vc=$n_vc --normalize