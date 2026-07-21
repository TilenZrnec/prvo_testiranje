#!/bin/bash
#SBATCH --job-name=tfm-subset
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-2
#SBATCH --output=logs/subset-%A_%a.out
# Po potrebi izpolni iz: sacctmgr show assoc user=$USER
##SBATCH --account=
##SBATCH --reservation=

set -euo pipefail

# TABPFN_TOKEN za headless uporabo TabPFN v3
source ~/.tabpfn_token

export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p logs results/per_dataset

$HOME/bin/micromamba run -p $HOME/envs/tabular \
    python -m src.run_one_dataset --index $SLURM_ARRAY_TASK_ID --ids-file scripts/subset_ids.json
