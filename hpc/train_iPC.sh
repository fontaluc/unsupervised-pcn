#!/bin/sh
### General options
### –- specify queue --
#SBATCH -C a100
### -- set the job Name --
#SBATCH -J train_iPC
### -- ask for number of cores (default: 1) --
#SBATCH -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#SBATCH --exclusive
### -- set walltime limit: j-h:m:s
#SBATCH --time 3-0
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#SBATCH -o /home/lfontain/unsupervised-pcn/outputs/logs/train_iPC_%J.out
#SBATCH -e /home/lfontain/unsupervised-pcn/outputs/logs/train_iPC_%J.err
# -- end of Slurm options --


# Load modules
module load build/conda/4.10
module load compiler/cuda/12.3

conda activate torch_env

srun python $HOME/unsupervised-pcn/src/pcn/make_mnist.py
srun python $HOME/unsupervised-pcn/src/pcn/train_model.py --N=10097