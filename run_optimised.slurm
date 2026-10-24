#!/bin/bash
#SBATCH --job-name=v10_1_hazard
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=v10_1_slurm_%j.out
#SBATCH --error=v10_1_slurm_%j.err

echo "=================================================="
echo "V10.1 HAZARD TRAINING"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo

# --------------------------------------------------
# Python environment
# --------------------------------------------------

module load miniforge3
conda activate mapanything2

echo "===== PYTHON ====="
which python
python --version
echo

# --------------------------------------------------
# GPU
# --------------------------------------------------

echo "===== GPU ====="
nvidia-smi
echo

# --------------------------------------------------
# Working directory
# --------------------------------------------------

cd /home/lcha0115/bt60_scratch/lcha_data/map-anything

echo "===== WORKING DIRECTORY ====="
pwd
echo

# --------------------------------------------------
# Training script
# --------------------------------------------------

TRAIN_SCRIPT="/home/lcha0115/bt60_scratch/lcha_data/map-anything/scripts/optimised_trainer.py"

echo "===== TRAINING SCRIPT ====="
echo "$TRAIN_SCRIPT"
echo

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "[ERROR] Training script not found:"
    echo "$TRAIN_SCRIPT"
    exit 1
fi

# --------------------------------------------------
# Start V10.1
# --------------------------------------------------

echo "=================================================="
echo "STARTING V10.1 HAZARD TRAINING"
echo "=================================================="
echo "Latent grid: 40 x 8 x 20"
echo "Occupancy: full-grid empty supervision"
echo "GPU: L40S"
echo "=================================================="
echo

python -u "$TRAIN_SCRIPT"

EXIT_CODE=$?

echo
echo "=================================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "V10.1 TRAINING FINISHED SUCCESSFULLY"
else
    echo "V10.1 TRAINING FAILED"
    echo "Exit code: $EXIT_CODE"
fi

echo "End: $(date)"
echo "=================================================="

exit $EXIT_CODE