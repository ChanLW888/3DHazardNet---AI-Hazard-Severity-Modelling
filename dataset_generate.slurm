#!/bin/bash
#SBATCH --job-name=5cm_generator
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=v8_5cm_%j.out
#SBATCH --error=v8_5cm_%j.err

echo "============================================================"
echo " 5 CM VOXEL GENERATION"
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
# Generator configuration
# ------------------------------------------------------------

echo "===== V8 GEOMETRY ====="
echo "Voxel size : 0.05 m"
echo "Bounds     : X=-8..8 m"
echo "             Y=-1..1 m"
echo "             Z=0.5..5 m"
echo "Grid       : 320 x 40 x 90"
echo

# ------------------------------------------------------------
# Run generator
# ------------------------------------------------------------

echo "===== STARTING GENERATION ====="
echo

python /home/lcha0115/bt60_scratch/lcha_data/map-anything/scripts/new_fine_dataset.py

EXIT_CODE=$?

echo
echo "============================================================"
echo "GENERATION FINISHED"
echo "============================================================"
echo "Exit code: $EXIT_CODE"
echo "End: $(date)"

exit $EXIT_CODE