# Run experiments on PLaFRIM

## Steps 
* Create an account on PLaFRIM
* Connect to PLaFRIM using Git Bash
```
ssh username@plafrim
```
* Load conda
```
module load build/conda/4.10
```
* Create conda environment
```
conda create --name torch_env python=3.11
```
* Activate environment
```
conda activate torch_env
```
* Clone Github repo
```
module load tools/git/2.36.0
git clone https://github.com/fontaluc/unsupervised-pcn.git
```
* Install project as package
```
pip install unsupervised-pcn
```
* Launch job on PLaFRIM
```
sbatch train_AM.sh
```

## Useful command lines on PLaFRIM
* Check available modules
```
module avail
```
* Check CUDA version
```
nvidia-smi
```