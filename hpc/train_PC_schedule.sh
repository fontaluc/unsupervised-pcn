#!/bin/sh
### General options
### –- specify queue --
#SBATCH -C sirocco
### -- set the job Name --
#SBATCH -J train_PC_schedule
### -- ask for number of cores (default: 1) --
#SBATCH -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#SBATCH --exclusive
### -- set walltime limit: j-h:m:s
#SBATCH --time 10:0:0
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#SBATCH -o /beegfs/lfontain/unsupervised-pcn/outputs/logs/train_PC_schedule_%J.out
#SBATCH -e /beegfs/lfontain/unsupervised-pcn/outputs/logs/train_PC_schedule_%J.err
# -- end of Slurm options --


# Load modules
module load build/conda/4.10
module load compiler/cuda/12.3

conda activate torch_env

srun python /beegfs/lfontain/unsupervised-pcn/src/pcn/train_PC.py --n_epochs=10000 --N=10097 --schedule=true