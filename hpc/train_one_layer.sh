#!/bin/bash
### General options
### –- specify queue --
#SBATCH -C sirocco
### -- set the job Name --
#SBATCH -J train_one_layer
### -- set the job array --
#SBATCH --array=1-6
### -- ask for number of cores (default: 1) --
#SBATCH -n 1
### -- ask for number of cores (default: 1) --
### -- set walltime limit: j-h:m:s
#SBATCH --time 10:0:0
### -- Specify the output and error file. %A_%a is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#SBATCH -o /beegfs/lfontain/unsupervised-pcn/outputs/logs/train_one_layer_%A_%a.out
#SBATCH -e /beegfs/lfontain/unsupervised-pcn/outputs/logs/train_one_layer_%A_%a.err
# -- end of Slurm options --


# Load modules
module load build/conda/4.10
module load compiler/cuda/12.3

conda activate torch_env

seeds=(1 2)
N_vc=(2000 750 550)
datasets=(cifar10 fmnist mnist)
N_epochs=(200 100 50)
N=${#N_vc[@]}

# compute seed index (tasks 1-3 -> 0, 4-6 -> 1, 7-9 -> 2)
seed_index=$(( (SLURM_ARRAY_TASK_ID - 1) / $N ))
seed=${seeds[$seed_index]}

n_index=$(( (SLURM_ARRAY_TASK_ID - 1) % $N ))
n_vc=${N_vc[$n_index]}
dataset=${datasets[$n_index]}
n_epochs=${N_epochs[$n_index]}

srun python /beegfs/lfontain/unsupervised-pcn/src/pcn/train_one_layer.py --dataset="$dataset" --n_vc=$n_vc --seed=$seed --n_epochs=$n_epochs --scheduler