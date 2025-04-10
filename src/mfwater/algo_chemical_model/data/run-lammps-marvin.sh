#!/bin/bash
#SBATCH --partition=intelsr_short
#SBATCH --account=ag_mctc_kirchner
#SBATCH --ntasks=N_CPU
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --job-name=JOB_NAME

export OMP_NUM_THREADS=1

module load LAMMPS/23Jun2022-foss-2022a-kokkos
mpirun lmp -i input.lmp
