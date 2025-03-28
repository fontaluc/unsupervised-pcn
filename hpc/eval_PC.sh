#!/bin/sh
### General options
### –- specify queue --
#SBATCH -C sirocco
### -- set the job Name --
#SBATCH -J eval_PC
### -- set the job array --
#SBATCH --array=1-9
### -- ask for number of cores (default: 1) --
#SBATCH -n 1
### -- set walltime limit: j-h:m:s
#SBATCH --time 1:0:0
### -- Specify the output and error file. %A_%a is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#SBATCH -o /home/lfontain/unsupervised-pcn/outputs/logs/eval_PC_%A_%a.out
#SBATCH -e /home/lfontain/unsupervised-pcn/outputs/logs/eval_PC_%A_%a.err
# -- end of Slurm options --


# Load modules
module load build/conda/4.10
module load compiler/cuda/12.3

conda activate torch_env

N=(64 1280 2560 3840 5120 6400 7680 8960 10097) 
n=${N[$SLURM_ARRAY_TASK_ID - 1]}
srun python $HOME/unsupervised-pcn/src/pcn/eval_PC.py --N=$n