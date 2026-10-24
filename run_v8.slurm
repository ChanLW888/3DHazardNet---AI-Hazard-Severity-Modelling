#!/bin/bash
#SBATCH --job-name=v8_foreground
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=v8_slurm_%j.out
#SBATCH --error=v8_slurm_%j.err

echo "===== JOB INFO ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

# Load Python environment
module load miniforge3
conda activate mapanything2

echo "===== PYTHON ====="
which python
python --version

echo "===== GPU ====="
nvidia-smi

# Go to MapAnything directory
cd /home/lcha0115/bt60_scratch/lcha_data/map-anything

echo "===== WORKING DIRECTORY ====="
pwd

echo "===== STARTING V8 ====="

python -u /home/lcha0115/bt60_scratch/lcha_data/map-anything/scripts/v9_trainer.py \
    --epochs 50 \
    --batch_size 4 \
    --workers 8

echo "===== FINISHED ====="

echo "End: $(date)"