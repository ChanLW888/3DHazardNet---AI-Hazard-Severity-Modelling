#!/bin/bash
#SBATCH --job-name=v10_1_pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=v10_1_pipeline_%j.out
#SBATCH --error=v10_1_pipeline_%j.err

echo "============================================================"
echo "V10.1 FULL PIPELINE"
echo "============================================================"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Start:   $(date)"
echo

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

module load miniforge3
conda activate mapanything2

echo "===== PYTHON ====="
which python
python --version
echo

# ------------------------------------------------------------
# GPU
# ------------------------------------------------------------

echo "===== GPU ====="
nvidia-smi
echo

# ------------------------------------------------------------
# Working directory
# ------------------------------------------------------------

cd /home/lcha0115/bt60_scratch/lcha_data/map-anything

echo "===== WORKING DIRECTORY ====="
pwd
echo

# ------------------------------------------------------------
# Script directory
# ------------------------------------------------------------

SCRIPT_DIR="/home/lcha0115/bt60_scratch/lcha_data/map-anything/scripts"

echo "===== SCRIPTS ====="
echo "Generator : $SCRIPT_DIR/new_fine_dataset.py"
echo "GPS filter: $SCRIPT_DIR/filter_gps_npzs.py"
echo "Trainer   : $SCRIPT_DIR/v10_1_trainer.py"
echo

# ============================================================
# STEP 1 — GENERATE 5 CM VOXELS
# ============================================================

echo "============================================================"
echo "STEP 1: 5 CM VOXEL GENERATION"
echo "============================================================"
echo "Voxel size : 0.05 m"
echo "Bounds     : X=-8..8 m"
echo "             Y=-1..1 m"
echo "             Z=0.5..5 m"
echo "Grid       : 320 x 40 x 90"
echo "Start:     $(date)"
echo

python -u "$SCRIPT_DIR/new_fine_dataset.py"

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "STEP 1 FAILED"
    echo "Exit code: $EXIT_CODE"
    echo "Pipeline stopped."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    exit $EXIT_CODE
fi

echo
echo "===== STEP 1 COMPLETE ====="
echo "Finished: $(date)"
echo


# ============================================================
# STEP 2 — FILTER GPS NPZS
# ============================================================

echo "============================================================"
echo "STEP 2: GPS NPZ FILTERING"
echo "============================================================"
echo "Start: $(date)"
echo

python -u "$SCRIPT_DIR/filter_gps_npzs.py"

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "STEP 2 FAILED"
    echo "Exit code: $EXIT_CODE"
    echo "Pipeline stopped."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    exit $EXIT_CODE
fi

echo
echo "===== STEP 2 COMPLETE ====="
echo "Finished: $(date)"
echo


# ============================================================
# STEP 3 — V10.1 TRAINING
# ============================================================

echo "============================================================"
echo "STEP 3: V10.1 TRAINING"
echo "============================================================"
echo "Start: $(date)"
echo

python -u "$SCRIPT_DIR/v10_1_trainer.py"

EXIT_CODE=$?

echo
echo "============================================================"
echo "V10.1 TRAINING FINISHED"
echo "============================================================"
echo "Exit code: $EXIT_CODE"
echo "End: $(date)"

exit $EXIT_CODE