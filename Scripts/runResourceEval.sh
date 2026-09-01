#!/usr/bin/env bash
#
#SBATCH --job-name vldb_resource
#SBATCH --ntasks=1
#SBATCH --time=12-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=CPU
#SBATCH --mem=8500M
#SBATCH --exclude=worker-a,worker-c

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
python3 ./online_resource_eval.py --online $1 --n $2 --d $3 --k $4 --data_seed 0 --method_seed $5
python3 ./online_resource_eval.py --online $1 --n $2 --d $3 --k $4 --data_seed 1 --method_seed $5
python3 ./online_resource_eval.py --online $1 --n $2 --d $3 --k $4 --data_seed 2 --method_seed $5
python3 ./online_resource_eval.py --online $1 --n $2 --d $3 --k $4 --data_seed 3 --method_seed $5
python3 ./online_resource_eval.py --online $1 --n $2 --d $3 --k $4 --data_seed 4 --method_seed $5