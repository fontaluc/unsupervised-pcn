#!/bin/sh
### General options
### –- specify queue --
#SBATCH -C a100
### -- set the job Name --
#SBATCH -J train_PC_decay
### -- ask for number of cores (default: 1) --
#SBATCH -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#SBATCH --exclusive
### -- set walltime limit: j-h:m:s
#SBATCH --time 1-0:0:0
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#SBATCH -o /beegfs/lfontain/unsupervised-pcn/outputs/logs/train_PC_decay_%J.out
#SBATCH -e /beegfs/lfontain/unsupervised-pcn/outputs/logs/train_PC_decay_%J.err
# -- end of Slurm options --


# Load modules
module load build/conda/4.10
module load compiler/cuda/12.3

conda activate torch_env

srun python /beegfs/lfontain/unsupervised-pcn/src/pcn/train_PC.py --lr=1e-5 --n_vc=400 --n_ec=10 --decay=true