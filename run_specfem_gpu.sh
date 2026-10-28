#!/bin/bash
#SBATCH --job-name=specfem_gpu
#SBATCH --account=nn4680k
#SBATCH --partition=accel
#SBATCH --output=logs/specfem_gpu_%j.out
#SBATCH --error=logs/specfem_gpu_%j.err

# Pour un run complet ~19 h :
#SBATCH --time=24:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G

set -euo pipefail

ROOT="$HOME/specfem-waves"

cd "$ROOT"
mkdir -p logs

module --force purge
module load NRIS/GPU
module load HDF5/1.14.5-gompi-2024a
module load CUDA/12.8.0

# CPU / OpenMP
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# OpenMPI recommandé pour mono-noeud GPU sur Olivia
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=sm,smcuda,self

echo "========================================"
echo " SPECFEM-WAVES GPU run"
echo "========================================"

echo
echo "=== Job info ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Host=$(hostname)"
echo "Architecture=$(uname -m)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

echo
echo "=== GPU ==="
nvidia-smi

echo
echo "=== Modules ==="
module list

echo
echo "=== SPECFEM executable ==="
ls -lh "$ROOT/build_gpu/specfem_wave"

echo
echo "=== Configuration ==="
echo "$ROOT/config/config.json"

echo
echo "=== Run ==="

srun \
  --mpi=pmix \
  -n "${SLURM_NTASKS}" \
  "$ROOT/build_gpu/specfem_wave" \
  "$ROOT/config/config.json"
