#!/bin/bash
#SBATCH --job-name=specfem_gpu
#SBATCH --account=nn4680k
#SBATCH --partition=accel
#SBATCH --output=/cluster/home/leay/specfem-waves/logs/specfem_gpu_%j.out
#SBATCH --error=/cluster/home/leay/specfem-waves/logs/specfem_gpu_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G

set -euo pipefail

ROOT="$HOME/specfem-waves"
CONFIG_SRC="$ROOT/config"

# Per-job fast scratch on Olivia GPU nodes.
RUN_DIR="$SCRATCH/specfem"
RUN_CONFIG="$RUN_DIR/config"

# Temporary staging area. Copy it back to the local PC, then delete it.
STAGE_BASE="/cluster/work/projects/nn4680k/leay/specfem/results"
STAGE_DIR="$STAGE_BASE/$SLURM_JOB_ID"

mkdir -p "$ROOT/logs"
mkdir -p "$RUN_CONFIG"
mkdir -p "$STAGE_DIR"

echo "========================================"
echo " SPECFEM-WAVES GPU run"
echo "========================================"
echo
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Host=$(hostname)"
echo "Architecture=$(uname -m)"
echo "SCRATCH=$SCRATCH"
echo "RUN_DIR=$RUN_DIR"
echo "STAGE_DIR=$STAGE_DIR"

module --force purge
module load NRIS/GPU
module load HDF5/1.14.5-gompi-2024a
module load CUDA/12.8.0

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=sm,smcuda,self

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
echo "=== Stage inputs to fast scratch ==="

# Copy the complete configuration directory but never copy previous outputs.
# -aL dereferences symlinks, so Isfjorden.e is copied as a real file to scratch.
rsync -aL \
    --exclude='output_test/' \
    --exclude='output_test' \
    "$CONFIG_SRC/" \
    "$RUN_CONFIG/"

# Make sure a fresh output directory exists in scratch.
rm -rf "$RUN_CONFIG/output_test"
mkdir -p "$RUN_CONFIG/output_test"

echo "Scratch configuration:"
du -sh "$RUN_CONFIG" || true
ls -lh "$RUN_CONFIG/Isfjorden.e" || true

# Register result recovery BEFORE the computation starts.
# Sigma2 runs this command when the job exits, including normal program failures.
cleanup "mkdir -p '$STAGE_DIR' && \
rsync -a '$RUN_CONFIG/output_test/' '$STAGE_DIR/output_test/' && \
cp '$RUN_CONFIG/config.json' '$STAGE_DIR/config_used.json' && \
test ! -f '$RUN_CONFIG/materials_generated.json' || cp '$RUN_CONFIG/materials_generated.json' '$STAGE_DIR/materials_generated.json' && \
test ! -f '$RUN_CONFIG/receivers_generated.json' || cp '$RUN_CONFIG/receivers_generated.json' '$STAGE_DIR/receivers_generated.json'"

echo
echo "=== Configuration ==="
echo "$RUN_CONFIG/config.json"

# Run from the copied configuration directory so relative paths such as
# "output_test" resolve inside $SCRATCH instead of $HOME or /cluster/work.
cd "$RUN_CONFIG"

echo
echo "=== Run ==="

srun \
  --mpi=pmix \
  -n "${SLURM_NTASKS}" \
  "$ROOT/build_gpu/specfem_wave" \
  "$RUN_CONFIG/config.json"

echo
echo "=== SPECFEM completed ==="
echo "Scratch output size:"
du -sh "$RUN_CONFIG/output_test" || true
echo
echo "Results will be staged by cleanup to:"
echo "$STAGE_DIR"
