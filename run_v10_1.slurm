#!/bin/bash
#SBATCH --job-name=v10_1_foreground
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=v10_1_slurm_%j.out
#SBATCH --error=v10_1_slurm_%j.err

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

python -u /home/lcha0115/bt60_scratch/lcha_data/map-anything/scripts/v10_1_trainer.py \
    --epochs 50 \
    --batch_size 2 \
    --workers 8 \
    --resume

echo "===== FINISHED ====="

echo "End: $(date)"