# Run experiments on PLaFRIM

Prerequisite: create an account on wandb.ai and retrieve your API key (in User Settings)

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
* Install pytorch using the official Pytorch index and not using PyPI with requirements.txt because CUDA-specific versions of CUDA are not listed in PyPI
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
* Install project as package
```
pip install unsupervised-pcn
```
* Set the WANDB_API_KEY environment variable to your API key in .bashrc
```
export WANDB_API_KEY=<your_api_key>
```
* Launch job on PLaFRIM
```
sbatch unsupervised-pcn/hpc/train_AM.sh
```

## Useful tips for PLaFRIM
* Check available modules
```
module avail
```
* Check CUDA version
```
nvidia-smi
```
We use A100 nodes in PLaFRIM which have CUDA 12.3. NVIDIA drivers are backward compatible with older CUDA versions, so we can install Pytorch built for CUDA 11.8 (latest Pytorch version is not built for 12.3). 
* Load recurrent modules automatically after login, by adding in $HOME/.bashrc
```
module load modulename
```
where modulename is tools/git/2.36.0 for example. 