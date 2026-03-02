#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH -J finetune-ministral3b
#SBATCH -o /home/khamlichi/Projekt_NMDL_2/logs/%x-%j.out
#SBATCH -e /home/khamlichi/Projekt_NMDL_2/logs/%x-%j.err

set -euo pipefail
cd /home/khamlichi/Projekt_NMDL_2
source /home/khamlichi/Projekt_NMDL_2/.venv/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
python -u finetune.py