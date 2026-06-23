#!/bin/sh
### General options
### –- specify queue --
#SBATCH -C sirocco
### -- set the job Name --
#SBATCH -J train_two_layers
### -- set the job array --
#SBATCH --array=1-8
### -- ask for number of cores (default: 1) --
#SBATCH -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#SBATCH --exclusive
### -- set walltime limit: j-h:m:s
#SBATCH --time 10:0:0
### -- Specify the output and error file. %A_%a is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#SBATCH -o /beegfs/lfontain/unsupervised-pcn/outputs/logs/train_two_layers_%A_%a.out
#SBATCH -e /beegfs/lfontain/unsupervised-pcn/outputs/logs/train_two_layers_%A_%a.err
# -- end of Slurm options --

# Initialize environment modules (module command is not loaded by default in some nodes)
source /etc/profile

# Load modules
module load build/conda/4.10
module load compiler/cuda/12.3

conda activate torch_env

N_ec=(500 400 300 250 200 150 100 50)
n_ec=${N_ec[$((SLURM_ARRAY_TASK_ID - 1))]}

srun python /beegfs/lfontain/unsupervised-pcn/src/pcn/train_PC.py --dataset="cifar10" --n_ec=$n_ec --n_vc=2000 --seed=1 --scheduler