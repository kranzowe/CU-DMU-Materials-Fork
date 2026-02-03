#!/bin/bash
#SBATCH --job-name=hw2_vi
#SBATCH --output=hw2_%j.out
#SBATCH --error=hw2_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nice=100

echo "==================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "==================================="

# Check current node memory before starting
free -h

# Key fixes for threading stability:
# 1. Set thread pinning
export JULIA_EXCLUSIVE=1

# 2. Limit OpenBLAS threads (prevents oversubscription)
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 3. Increase stack size for threads
ulimit -s unlimited

# ============================================
# Run Julia
# ============================================
julia --threads=$SLURM_CPUS_PER_TASK \
      --heap-size-hint=50G \
      --check-bounds=yes \
      hw2_solutions.jl

exit_code=$?

echo ""
echo "==================================="
echo "Job finished: $(date)"
echo "Exit code: $exit_code"
free -h
echo "==================================="

exit $exit_code