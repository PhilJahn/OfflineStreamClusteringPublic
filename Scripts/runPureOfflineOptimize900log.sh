#!/usr/bin/env bash
#
#SBATCH --job-name vldb_poffline
#SBATCH --ntasks=1
#SBATCH --time=12-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=CPU
#SBATCH --mem=8500M
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
python3 ./optimize_pure_offline.py --ds $1 --offline $2 --batch_budget 900_logdim --batch 1000 --smac_seed $3 --memory_limit 8000 --trial_limit 1000