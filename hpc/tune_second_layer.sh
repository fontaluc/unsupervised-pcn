#!/bin/sh
### General options
### -- set the job Name --
#SBATCH -J tune_second_layer
### -- set the job array --
#SBATCH --array=1-24
### -- ask for number of cores (default: 1) --
#SBATCH -n 1
### -- set walltime limit: j-h:m:s
#SBATCH --time 1-0:0:0
### -- Specify the output and error file. %A_%a is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#SBATCH -o /beegfs/lfontain/unsupervised-pcn/outputs/logs/tune_second_layer_%A_%a.out
#SBATCH -e /beegfs/lfontain/unsupervised-pcn/outputs/logs/tune_second_layer_%A_%a.err
# -- end of Slurm options --

# Initialize environment modules (module command is not loaded by default in some nodes)
source /etc/profile

# Load modules
module load build/conda/4.10
module load compiler/cuda/12.3

conda activate torch_env

datasets=(cifar10 fmnist mnist)
N_ec=(400 350 300 250 200 150 100 50)
N=${#N_ec[@]}

# compute seed index (tasks 1-3 -> 0, 4-6 -> 1, 7-9 -> 2)
dataset_index=$(( (SLURM_ARRAY_TASK_ID - 1) / $N ))
dataset=${datasets[$dataset_index]}

n_index=$(( (SLURM_ARRAY_TASK_ID - 1) % $N ))
n_ec=${N_ec[$n_index]}

srun python /beegfs/lfontain/unsupervised-pcn/src/pcn/eval_accuracy.py --dataset="$dataset" --n_ec=$n_ec --seed=0