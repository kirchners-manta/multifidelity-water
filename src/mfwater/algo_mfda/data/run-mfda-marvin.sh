#!/bin/bash
# Template for an MFDA run on Marvin. Edit the settings below, or replace the
# placeholders with sed. Submit with `sbatch run-mfda-marvin.sh`.
#
# Number of tasks: chains x calc_cpus(N_1). Get it with
#   mfwater -a mfda-ncpu --molecules <N_1> ... --chains <chains>
# and put it in --ntasks (N_CPU). All tasks must fit on one node (--nodes=1),
# because each chain starts its own `mpirun` from the same allocation.
#
# RESTART: resubmit this very script (same --workdir and same arguments).
# Finished chains are skipped, unfinished chains are replayed from the start with
# the same seed and every cached evaluation returns instantly. Arguments are
# compared with <workdir>/manifest.json; a mismatch aborts with an error.
#
# SMOKE TEST (run once before a long job, one real evaluation, no MCMC):
#   mfwater -a mfda-smoke --forward-model md --molecules 32 --workdir ./smoke_run
#
#SBATCH --partition=intelsr_long
#SBATCH --account=ag_mctc_kirchner
#SBATCH --ntasks=N_CPU
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --job-name=JOB_NAME

export OMP_NUM_THREADS=1

module load LAMMPS/23Jun2022-foss-2022a-kokkos
# fftool, packmol, TRAVIS (serial) and msdiff must be in PATH as well.

# Several chains call mpirun concurrently. Open MPI would bind each mpirun to
# the same first cores, hence --bind-to none. Alternative (check on Marvin):
#   LAMMPS_CMD="srun --exact -n {ncpu} lmp -i {input}"
LAMMPS_CMD="mpirun --bind-to none -np {ncpu} lmp -i {input}"

mfwater -a markov-chain \
    --forward-model md \
    --workdir ./mfda_run \
    --chains N_CHAINS \
    --models 3 \
    --molecules 1000 500 100 \
    --mcchainlength 1000 \
    --mcsubchainlength 10 \
    --mcburnin 100 \
    --params lj \
    --seed 1 \
    --lammps-cmd "${LAMMPS_CMD}"
