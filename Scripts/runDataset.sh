#!/usr/bin/env bash
#
#SBATCH --job-name vldb_data
#SBATCH --ntasks=1
#SBATCH --time=12-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=CPU
#SBATCH --mem=4500M
#SBATCH --exclude=worker-a

# debug info
hostname
which python3
nvidia-smi

env

# venv
source ./venv/bin/activate
pip3 install -r requirements.txt
python3 -c "import torch; print(torch.cuda.device_count())"

# train
python3 ./datahandler.py