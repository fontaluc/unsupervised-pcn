#!/bin/sh
### General options
### –- specify queue --
#SBATCH -C a100
### -- set the job Name --
#SBATCH -J train_PCModule
### -- set the job array --
#SBATCH --array=1-2
### -- ask for number of cores (default: 1) --
#SBATCH -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#SBATCH --exclusive
### -- set walltime limit: j-h:m:s
#SBATCH --time 20:0:0
### -- Specify the output and error file. %A_%a is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#SBATCH -o /home/lfontain/unsupervised-pcn/outputs/logs/train_PCModule_%A_%a.out
#SBATCH -e /home/lfontain/unsupervised-pcn/outputs/logs/train_PCModule_%A_%a.err
# -- end of Slurm options --


# Load modules
module load build/conda/4.10
module load compiler/cuda/12.3

conda activate torch_env

N=(64 10097)
n=${N[$SLURM_ARRAY_TASK_ID - 1]}
N_epochs=(20000 1000)
n_epochs=${N_epochs[$SLURM_ARRAY_TASK_ID - 1]}
srun python $HOME/unsupervised-pcn/src/pcn/train_PCModule.py --n_epochs=$n_epochs --N=$n