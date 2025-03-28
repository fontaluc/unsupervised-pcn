#!/bin/sh
### General options
### –- specify queue --
#SBATCH -C sirocco
### -- set the job Name --
#SBATCH -J train_PC_extra
### -- ask for number of cores (default: 1) --
#SBATCH -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#SBATCH --exclusive
### -- set walltime limit: j-h:m:s
#SBATCH --time 10:0:0
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#SBATCH -o /home/lfontain/unsupervised-pcn/outputs/logs/train_PC_extra_%J.out
#SBATCH -e /home/lfontain/unsupervised-pcn/outputs/logs/train_PC_extra_%J.err
# -- end of Slurm options --


# Load modules
module load build/conda/4.10
module load compiler/cuda/12.3

conda activate torch_env

srun python $HOME/unsupervised-pcn/src/pcn/train_PC.py --n_epochs=1942 --N=8960