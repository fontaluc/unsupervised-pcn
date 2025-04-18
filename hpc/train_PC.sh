#!/bin/sh
### General options
### –- specify queue --
#SBATCH -C sirocco
### -- set the job Name --
#SBATCH -J train_PC
### -- set the job array --
#SBATCH --array=1-2
### -- ask for number of cores (default: 1) --
#SBATCH -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#SBATCH --exclusive
### -- set walltime limit: j-h:m:s
#SBATCH --time 10:0:0
### -- Specify the output and error file. %A_%a is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#SBATCH -o /beegfs/lfontain/unsupervised-pcn/outputs/logs/train_PC_%A_%a.out
#SBATCH -e /beegfs/lfontain/unsupervised-pcn/outputs/logs/train_PC_%A_%a.err
###SBATCH -o /beegfs/lfontain/unsupervised-pcn/outputs/logs/train_PC_%J.out
###SBATCH -e /beegfs/lfontain/unsupervised-pcn/outputs/logs/train_PC_%J.err
# -- end of Slurm options --


# Load modules
module load build/conda/4.10
module load compiler/cuda/12.3

conda activate torch_env

N=(512 640)
n=${N[$SLURM_ARRAY_TASK_ID - 1]}
N_epochs=(9000 8000)
n_epochs=${N_epochs[$SLURM_ARRAY_TASK_ID - 1]}
srun python /beegfs/lfontain/unsupervised-pcn/src/pcn/train_PC.py --n_epochs=$n_epochs --N=$n