#!/usr/bin/env bash
#
#SBATCH --job-name vldb_offline
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
python3 ./optimize_offline.py --ds $1 --online $2 --offline $3 --online_batch_budget 300_logdim --offline_batch_budget 600_logdim --batch 1000 --smac_seed $4 --memory_limit 8000 --trial_limit 1000 --batch_id $5