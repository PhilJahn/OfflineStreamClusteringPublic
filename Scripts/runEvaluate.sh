#!/usr/bin/env bash
#
#SBATCH --job-name vldb_eval
#SBATCH --ntasks=1
#SBATCH --time=12-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=CPU
#SBATCH --mem=4500M
#SBATCH --exclude=worker-a,worker-b

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
python3 ./result_evaluator.py --ds $1 --online $2 --offline $3 --batch_budget_online $4 --batch_budget_offline $5 --batch 1000 --overwrite 1