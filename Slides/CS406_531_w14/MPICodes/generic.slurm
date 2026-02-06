#!/bin/bash
#SBATCH --job-name=mpi_hello_4nodes
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4          # 1 MPI rank per node (classic "4-node" hello)
#SBATCH --time=00:02:00
#SBATCH --output=mpi_hello_%j.out
#SBATCH --error=mpi_hello_%j.err

# Optional: load your MPI module (adjust to your site)
# module purge
# module load gcc
# module load openmpi

echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_NODELIST"

# Path to your compiled MPI binary
MPI_EXE=./hello_mpi

# Recommended with Slurm-managed MPI:
srun --mpi=pmix $MPI_EXE