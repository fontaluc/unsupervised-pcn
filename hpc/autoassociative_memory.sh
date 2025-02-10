#!/bin/sh
### General options
### –- specify queue --
#SBATCH -C v100
### -- set the job Name AND the job array --
#BSUB -J train_AM
### -- ask for number of cores (default: 1) --
#SBATCH -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#SBATCH --exclusive
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#SBATCH -time 10:0:0
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##SBATCH --mail-user=your_email_address
### -- send notification at start and completion --
#SBATCH --mail-type=BEGIN,END
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#SBATCH -o /home/lfontain/unsupervised-pc/outputs/logs/train_AM_%J.out
#SBATCH -e /home/lfontain/unsupervised-pc/outputs/logs/train_AM_%J.err
# -- end of Slurm options --

unset PYTHONHOME
unset PYTHONPATH
source $HOME/miniconda3/bin/activate

nvidia-smi
# Load the cuda module
module swap cuda/11.6

/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

python3 src/models/train_model.py