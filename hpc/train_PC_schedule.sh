#!/bin/sh
### General options
### –- specify queue --
#SBATCH -C a100
### -- set the job Name --
#SBATCH -J train_PC_schedule
### -- set the job array --
#SBATCH --array=1-2
### -- ask for number of cores (default: 1) --
#SBATCH -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#SBATCH --exclusive
### -- set walltime limit: j-h:m:s
#SBATCH --time 1-0
### -- Specify the output and error file. %A_%a is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#SBATCH -o /beegfs/lfontain/unsupervised-pcn/outputs/logs/train_PC_schedule_%A_%a.out
#SBATCH -e /beegfs/lfontain/unsupervised-pcn/outputs/logs/train_PC_schedule_%A_%a.err
# -- end of Slurm options --


# Load modules
module load build/conda/4.10
module load compiler/cuda/12.3

conda activate torch_env

N=(64 10097)
n=${N[$SLURM_ARRAY_TASK_ID - 1]}
srun python /beegfs/lfontain/unsupervised-pcn/src/pcn/train_PC.py --n_epochs=20000 --N=$n --schedule=true