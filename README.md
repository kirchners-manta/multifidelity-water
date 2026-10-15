# mfwater
---

[![CI](https://github.com/kirchners-manta/multifidelity-water/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/kirchners-manta/multifidelity-water/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/kirchners-manta/multifidelity-water/branch/main/graph/badge.svg)](https://codecov.io/gh/kirchners-manta/multifidelity-water)
[![Paper I](https://img.shields.io/badge/Paper%20I-J.%20Chem.%20Phys.%202026-informational.svg)](https://doi.org/10.1063/5.0308381)
[![Paper II](https://img.shields.io/badge/Paper%20II-in%20preparation-lightgrey.svg)](#reference)
![Python versions](https://img.shields.io/badge/python-3.12%20|%203.13%20|%203.14-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-2a6db2.svg)](https://mypy-lang.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

A program to perform multifidelity Monte Carlo (MFMC) molecular dynamics (MD) simulations of the diffusion coefficient of water.
The MFMC scheme is based on the work of [Peherstorfer *et al.*](https://doi.org/10.1137/15M1046472)

## Installation
The tool can be installed using `pip`:
```bash
git clone git@github.com:kirchners-manta/multifidelity-water.git
cd multifidelity-water
pip install .
```

## Usage
The tool can be used from the command line:
```bash
mfwater -h
```
will show the help message including the available commands and options.
The main option of the program is the algorithm to be executed, which can be selected using the `-a` option.
As of now, the available algorithms are:
- `build`: Build an `.hdf5` file containing the models and their parameters as requested by the user through the command line (using default values if not specified).
- `chemmod-prep`: Preparations for the chemical model (i.e., generating input files for [LAMMPS](https://www.lammps.org/), currently only for the [OPC3](https://doi.org/10.1063/1.4960175) force field).
- `chemmod-post`: Postprocessing of the chemical model (i.e., calculating the MSD and the diffusion coefficient)
- `mfmc-prep`: Calculating the correlations between the models.
- `model-select`: Selects optimal models for the MFMC algorithm based on their correlations and costs.
- `eval-estimator`: Estimates the optimal number of evaluations for the selected models.
*Note*: After the `eval-estimator` step, the `chemmodel-prep` and `chemmod-post` steps need to be repeated with the selected models.
After that, the `mfmc` step can be executed.
- `mfmc`: Computes the MFMC estimator.
- `markov-chain`: Multifidelity delayed acceptance MCMC (MFDA) for calibrating force field parameters, see [markov-chain](#markov-chain).
- `mfda-smoke`: Runs a single forward evaluation (no MCMC), to test the external programs.
- `mfda-ncpu`: Prints the number of CPUs needed for the chosen number of chains.

For a detailed description of the algorithms, see the [Algorithms](#algorithms) section below.

As the number of options is large, all options can also be given in a [TOML](https://toml.io/en/) configuration file, which is read with `-c`/`--config`:
```bash
mfwater -c my_config.toml
```
The keys are the long option names without the leading dashes (`molecules`, `mcchainlength`, `forward-model`, ...; `-` and `_` are interchangeable), or the short options' names `algorithm`, `input` and `output`. Lists are TOML arrays, flags are booleans:
```toml
algorithm     = "markov-chain"
forward-model = "md"
models        = 3
molecules     = [1000, 500, 100]
mcchainlength = 1000
mcburnin      = 100
params        = "lj"
orthoboxy     = false
seed          = 1
```
**Note:** Options in the configuration file override options given on the command line (and the defaults). Values are checked like on the command line, and unknown keys are rejected. A `false` flag does not switch off a flag given on the command line.
The file overrides the command line on purpose: it is the record of what was run (a copy is stored in the `--workdir` of `markov-chain`).

The program operates on HDF5 files containing information on the models, which can for example be inspected using the `h5dump` command, e.g., like:
```bash
h5dump -n 1 somefile.hdf5
```

## Algorithms

### build
This algorithm builds the input file for the entire procedure.
The number of models, molecules and evaluations can be specified using the `--models`, `--molecules` and `--evals` options, respectively, where the latter two expect a list of integers of the same length as the number of models.
`-o` specifies the output file name (suffix `.hdf5` is recommended).
```bash
mfwater -a build --models 3 --molecules 128 64 32 --evals 50 50 50 -o blub.hdf5
```
will produce a file `blub.hdf5` with 3 models, 128 molecules for the first model, 64 for the second and 32 for the third, each with 50 evaluations and the following output:
```
Input file 'blub.hdf5' created with the following settings:
Model        Mols         Evals
-------------------------------
model_1       128            50
model_2        64            50
model_3        32            50
```
After the `build` step, the input file will have the following structure (inspecting it with `h5dump -n 1 blub.hdf5`):
```
HDF5 "blub.hdf5" {
FILE_CONTENTS {
 group      /
 attribute  /last_algo
 group      /models
 attribute  /models/n_models
 group      /models/model_1
 attribute  /models/model_1/n_evals
 attribute  /models/model_1/n_molecules
 group      /models/model_2
 attribute  /models/model_2/n_evals
 attribute  /models/model_2/n_molecules
 group      /models/model_3
 attribute  /models/model_3/n_evals
 attribute  /models/model_3/n_molecules
 }
}
```
where `models` and `model_{n}` are groups and `n_models` is an attribute of the `models` group.
Every `model_{n}` group has attributes `n_molecules` and `n_evals` (the number of molecules and evaluations for the model).
Additionally, the `last_algo` attribute is added to the root group, which stores the name of the last executed algorithm and checks whether a new algorithm can be executed on the input file.

The `build` algorithm can be executed several times. If the output file already exists, the user is asked whether to overwrite it, but only in an interactive session; in a batch job, a pipe or any other session without a terminal, `build` refuses and leaves the file untouched.

### chemmod-prep
This algorithm generates the directories and input files for the MD simualtion using LAMMPS for each model and evaluation.
To do so, it uses template input files available [here](./src//mfwater//algo_chemical_model/data/), modifies them accordingly, and uses [fftool](https://github.com/paduagroup/fftool) and [packmol](https://m3g.github.io/packmol/) to create the LAMMPS data files.
Specifically, the program draws random Lennard-Jones (LJ) parameters from a Gaussian distribution centered at the original values of the OPC3 force field with a standard deviation of 1/300 of the original value for $\epsilon$ and $\sigma$ (`--params lj`), or draws the oxygen partial charge with a standard deviation of 1/50 of its value and sets $q_\text{H} = -q_\text{O}/2$ (`--params q`).
The option `--params lj-q` perturbs both; paper I never perturbs both at once.
Additionally, random seeds are generated for the LAMMPS velocity command and the PACKMOL input file.
The random number generator is seeded with `--seed` (a random seed is drawn, printed and stored if the option is omitted); the seed is stored as attribute `seed` of the `models` group, so the sampled parameters and seeds can be reproduced.

It is important to note that the program will draw $n_\text{max}$ random LJ parameters where $n_\text{max}$ is the maximum number of evaluations of all models.
These parameters (and the random seeds) are stored as datasets of the `models` group in the input file.
Models with $n < n_\text{max}$ evaluations will use the first $n$ parameters for their evaluations.

The algorithm can be executed with the following command:
```bash
mfwater -a chemmodel-prep -i blub.hdf5
```
where `blub.hdf5` is the input file created in the `build` step.
The file structure will be modified to include the LJ parameters and the random seeds as datasets in the `models` group.
```
HDF5 "blub.hdf5" {
FILE_CONTENTS {
 group      /
 attribute  /last_algo
 group      /models
 attribute  /models/n_models
 dataset    /models/lj_params
 group      /models/model_1
 attribute  /models/model_1/n_evals
 attribute  /models/model_1/n_molecules
 group      /models/model_2
 ...
 dataset    /models/seeds
 }
}
```
A directory `models` will be created in the current working directory, containing subdirectories for each model and evaluation.
In each evaluattion subdirectory, a simulation input folder `siminp` will be created, containing the LAMMPS input files and a runscript for the [Marvin cluster](https://www.hpc.uni-bonn.de/en/systems/marvin) at University of Bonn.

Optionally, instead of cubic simulation boxes (which is the default), the user can request tetragonal boxes in the [OrthoBoXY](https://doi.org/10.1021/acs.jpcb.3c04492) style by specifying the `--orthoboxy` option.

After executing the `chemmod-prep` algorithm, the user can run the simulations in parallel on the cluster (not included in this code).
When completed, the mean squared displacement (MSD) and the diffusion coefficient can be calculated using [TRAVIS](http://www.travis-analyzer.de/) and [msdiff](https://github.com/kirchners-manta/msdiff) (not included in this code).
Please note that when specifying the `--orthoboxy` option, the TRAVIS and msdiff anlyses have to be adjusted accordingly, as the simulation box is not cubic anymore.

The evaluations are created below `--models-dir` (default `models`).
The `chemmodel-prep` algorithm cannot be executed several times on the same input file, as it would draw new random parameters.
It also refuses to run if the `siminp`, `simout` or `msd` directory of one of its evaluations already exists (for example from the calibration phase), and nothing is deleted or overwritten. Use a new `--models-dir` for every phase, and pass the same directory to `chemmodel-post`. `fftool` and `packmol` are called without a shell.

### chemmod-post
This algorithm postprocesses the results of the MD simulations.
```bash
mfwater -a chemmodel-post -i big_oms.hdf5 --models-dir models
```
It looks for the output files of LAMMPS and msdiff in the `simout/log.lammps` and `msd/msdiff_out.csv` folders/files of the models and evaluations directories, to extract the computation time and the diffusion coefficient, respectively.
For each model, the dataset `diffusion_coeff` is added to the model group, containing the diffusion coefficient (in $10^{-12}\,\text{m}^2\,\text{s}^{-1}$) for each evaluation.
Additionally, each model will be given an attibute `computation_time` with the averaged cost of the evaluations in core-seconds: the LAMMPS `Total wall time` multiplied by the number of MPI ranks, read from the log of that evaluation only. An unfinished log (no `Total wall time` line) is an error.
The LJ parameters and partial charges printed in each LAMMPS log must agree with the stored ones, otherwise the algorithm stops with the model and evaluation number. (Files in the old format without partial charges skip the charge check with a warning.)
The expected file structure after executing the `chemmod-post` algorithm is:
```
HDF5 "blub.hdf5" {
FILE_CONTENTS {
 group      /
 attribute  /last_algo
 group      /models
 attribute  /models/n_models
 dataset    /models/lj_params
 group      /models/model_1
 attribute  /models/model_1/computation_time
 attribute  /models/model_1/n_evals
 attribute  /models/model_1/n_molecules
 dataset    /models/model_1/diffusion_coeff
 group      /models/model_2
 ...
 dataset    /models/seeds
 }
}
```
The `chemmodel-post` algorithm can be executed several times on the same input file, but it will overwrite the existing datasets and attributes.

### mfmc-prep
This algorithm calculates the mean and standard deviation of the diffusion coefficients for each model and stores them as attributes `mean` and `std` in the respective model group.
Based on that, the correlations between the models are calculated and stored as attribute `correlation` in the respective model group.

The models are ordered by decreasing number of molecules (the first one is the high-fidelity model), never by the alphabetical order of the HDF5 group names, so ten or more models work.
The statistics are computed by `mfwater.mfmc_core` (standard deviation and correlation with `ddof = 0`).

The algorithm can be executed with the following command (using a difrerent input file with six models):
```bash
mfwater -a mfmc-prep -i big.hdf5
```
and will yield something like this:
```
MFMC preparation:
Model        Mols         Evals          Mean           Std   Correlation
-------------------------------------------------------------------------
model_1      1024          1000   2281.568604    666.583191      1.000000
model_2       512          1000   2217.837891    648.927795      0.995251
model_3       256          1000   2133.094727    624.330750      0.992927
model_4       128          1000   2036.662231    601.401062      0.986987
model_5        64          1000   1900.939941    575.116638      0.981722
model_6        32          1000   1772.538452    507.623077      0.964130
```
The file structure after executing the `mfmc-prep` algorithm will be:
```
HDF5 "big.hdf5" {
FILE_CONTENTS {
 group      /
 attribute  /last_algo
 group      /models
 attribute  /models/n_models
 dataset    /models/lj_params
 group      /models/model_1
 attribute  /models/model_1/computation_time
 attribute  /models/model_1/correlation
 attribute  /models/model_1/mean
 attribute  /models/model_1/n_evals
 attribute  /models/model_1/n_molecules
 attribute  /models/model_1/std
 dataset    /models/model_1/diffusion_coeff
 group      /models/model_2
 ...
 dataset    /models/seeds
 }
}
```
The `mfmc-prep` algorithm can be executed several times on the same input file, but it will overwrite the existing attributes.

### model-select
This algorithm selects the optimal models for the MFMC algorithm based on their correlations and costs (computation time).
It will create a new input file with the selected models and their parameters (remaining models will be removed).
For example,
```bash
mfwater -a model-select -i big.hdf5 -o selected_models.hdf5
```
could yield the following output:
```
Optimal models selected and saved to 'big_test.hdf5':
Model        Mols         Evals          Mean           Std
-----------------------------------------------------------
model_1      1024          1000   2281.568604    666.583191
model_2        64          1000   1900.939941    575.116638
model_3        32          1000   1772.538452    507.623077
```
saying that of the previous six models, three (numbers 1, 5, and 6) were selected for the MFMC algorithm, with the latter two being renamed to `model_2` and `model_3`, respectively.

The selected models are renamed `model_1`, `model_2`, ... in the order of decreasing squared correlation, which is the order the following algorithms use.
The output file must not exist: `model-select` refuses to overwrite an existing file (choose another one with `-o`). The input file is not changed.

### eval-estimator
This algorithm estimates the optimal number of evaluations for the selected models.
Based on a computational budget in s, specified by the `--budget` option, it will update the number of evaluations as an attribute `n_evals` to the model group and prepare the input file to be operated by the `chemmod-prep` algorithm.
Therefore, several all old attributes and datasets will be renamed with an `_initial` suffix.
```bash
mfwater -a eval-estimator -i selected_models.hdf5 --budget 1000000
```
could yield the following output:
```
Estimated optimal number of evaluations:
Model        Mols  Evals(init.)   Mean(init.)    Std(init.)   Evals(opt.)
-------------------------------------------------------------------------
model_1      1024          1000   2281.568604    666.583191             7
model_2        64          1000   1900.939941    575.116638            36
model_3        32          1000   1772.538452    507.623077           278
```
The file structure after executing the `eval-estimator` algorithm will be:
```
HDF5 "selected_models.hdf5" {
FILE_CONTENTS {
 group      /
 attribute  /last_algo
 group      /models
 attribute  /models/budget
 attribute  /models/n_models
 dataset    /models/lj_params_initial
 group      /models/model_1
 attribute  /models/model_1/alpha
 attribute  /models/model_1/computation_time_initial
 attribute  /models/model_1/correlation
 attribute  /models/model_1/mean_initial
 attribute  /models/model_1/n_evals
 attribute  /models/model_1/n_evals_initial
 attribute  /models/model_1/n_molecules
 attribute  /models/model_1/std_initial
 dataset    /models/model_1/diffusion_coeff_initial
 group      /models/model_2
 ...
 dataset    /models/seeds_initial
 }
}
```
The `eval-estimator` algorithm cannot be executed several times on the same input file, because some attributes and datasets were renamed with an `_initial` suffix.

After completing the `eval-estimator` step, the `chemmodel-prep` and `chemmod-post` steps need to be repeated with the selected models.

### mfmc
This algorithm computes the MFMC estimator based on the selected models and their results obtained with optimal number of evaluations.
The MFMC estimator, its error and `gamma` are added as attributes `mfmc_estimator`, `mfmc_error` and `gamma` to the input file.
```bash
mfwater -a mfmc -i selected_models.hdf5
```
An exemplary output could be:
```
Calculation of the MFMC estimator:
Model        Mols  Evals(init.)   Mean(init.)    Std(init.)   Evals(opt.)    Mean(opt.)
---------------------------------------------------------------------------------------
model_1      1024          1000   2281.568604    666.583191             7   2325.117920
model_2        64          1000   1900.939941    575.116638            36   1736.157715
model_3        32          1000   1772.538452    507.623077           278   1653.460938
---------------------------------------------------------------------------------------
MFMC Estimator                                                              2194.473635
```
where (`init`) and (`opt`) refer to the initial and optimal values, respectively.
The file structure after executing the `mfmc` algorithm will be:
```
HDF5 "selected_models.hdf5" {
FILE_CONTENTS {
 group      /
 attribute  /last_algo
 group      /models
 attribute  /models/budget
 attribute  /models/gamma
 attribute  /models/mfmc_error
 attribute  /models/mfmc_error_performance_cost
 attribute  /models/mfmc_estimator
 attribute  /models/n_models
 attribute  /models/rmse_cost
 dataset    /models/lj_params
 dataset    /models/lj_params_initial
 group      /models/model_1
 attribute  /models/model_1/alpha
 attribute  /models/model_1/computation_time_initial
 attribute  /models/model_1/correlation
 attribute  /models/model_1/mean
 attribute  /models/model_1/mean_initial
 attribute  /models/model_1/n_evals
 attribute  /models/model_1/n_evals_initial
 attribute  /models/model_1/n_molecules
 attribute  /models/model_1/std_initial
 dataset    /models/model_1/diffusion_coeff
 dataset    /models/model_1/diffusion_coeff_initial
 group      /models/model_2
 ...
 dataset    /models/seeds
 dataset    /models/seeds_initial
 }
}
```
The RMSE `mfmc_error` (Eq. 20 of paper I) and the ratio `gamma` (Eq. 22; MSE of the MFMC estimator relative to the high-fidelity Monte Carlo estimator, `gamma < 1` means a saving) are computed with the costs selected by `--rmse-cost`:
- `performance` (default): the costs `computation_time` measured in the evaluation run; this reproduces the `mfmc_error` stored in the paper-I files.
- `calibration`: the costs `computation_time_initial` of the calibration run, i.e. the costs the allocation was based on.

`mfmc_error_performance_cost` and `mfmc_error_calibration_cost` hold the RMSE of each basis (if its costs are stored in the file), `gamma` belongs to the selected basis, and `rmse_cost` records which one was selected.

The `mfmc` algorithm can be executed several times on the same input file, but it will overwrite the existing attributes.

### markov-chain
Multifidelity delayed acceptance MCMC (MFDA, after [Lykkegaard et al.](https://arxiv.org/abs/2202.03876)) for the parameters `(epsilon_OO, sigma_OO, q_O)` of OPC3.
Level 1 is the finest (largest `N`), the last level the coarsest.
```bash
mfwater -a markov-chain --forward-model md --workdir ./mfda_run --chains 4 \
    --molecules 1000 500 100 --mcchainlength 1000 --mcsubchainlength 10 \
    --mcburnin 100 --params lj --seed 1
```
Arguments (defaults in brackets):
- `--forward-model {dummy,md}` [`dummy`]: cheap stand-in or the MD pipeline (fftool, packmol, LAMMPS, TRAVIS, msdiff).
- `--workdir` [`./mfda_run`]: run directory, see below.
- `--chains` [1]: independent chains, run concurrently as separate processes. Chain `c` uses `np.random.default_rng([seed, c])` and its own cache.
- `--molecules`: molecules per level (strictly descending); the number of levels is `len(--molecules)`. `--models` is ignored by `markov-chain`.
- `--mcchainlength` [1000]: steps `M_1` on the finest level that are kept. `--mcburnin` [100]: burn-in steps `B`, run in addition (`B + M_1` fine steps in total); everything produced on all levels during the first `B` steps is discarded, so the finest level keeps exactly `M_1` samples. `B = 0` is allowed but prints a warning.
- `--mcsubchainlength` [10]: subchain lengths for levels `2..eta`; one value is used for all levels, otherwise `len(--molecules) - 1` values.
- `--params {lj,q,lj-q}` [`lj`]: calibrated parameters, the others stay at the OPC3 value.
- `--seed` [random, printed and stored in the manifest], `--orthoboxy` (tetragonal OrthoBoXY boxes, see below), `--lammps-cmd` [`mpirun -np {ncpu} lmp -i {input}`], `--print-level {0,1}` [0] (0 = silent, 1 = debug output of the MFDA chains; also settable as `print_level = 1` in the TOML file).
- `-o` [`default.hdf5`]: name of the combined summary file inside `--workdir` (an absolute path is used as is).

Run directory:
```
manifest.json                 arguments, seed, algorithm and package version, git hash
cache/chain_000/N{n}/{key_hash}/result.json
cache/chain_000/N{n}/{key_hash}/attempt_{k}/{siminp,simout,msd}/
chain_000.hdf5, ...           samples/proposals per level (post-burn-in), estimator; attributes
                              n_burnin (B), chain_length (M_1), n_steps_total (B + M_1)
default.hdf5                  per-chain estimators, mean, between-chain standard deviation
config.toml                   copy of the `-c` file (if used); changed content on a restart is
                              kept as config_1.toml, ... (nothing is overwritten)
```
Nothing is overwritten or deleted. Existing result files are refused.

**Restart.** Rerun the same command (on Marvin: resubmit the same job script).
Chains with an existing `chain_{c:03d}.hdf5` are skipped; the others are replayed from the start with the same seed, and every evaluation with a `result.json` in the cache returns instantly (the number of cached/computed evaluations is printed per chain).
The arguments and an internal `algorithm_version` (bumped when a code change alters the results of a replay) are compared with `manifest.json`; on a mismatch the run aborts (use a new `--workdir`). A manifest without `algorithm_version` counts as version 1 and is refused.
A different git hash, uncommitted changes (`git_dirty`) or a different package version only print a `Warning:`; you are responsible for the change not affecting results. `--lammps-cmd` and the print level are stored but not compared (e.g. `mpirun` -> `srun` between restarts).
An interrupted MD evaluation leaves its `attempt_{k}` directory in place (never modified). The MFDA LAMMPS input `src/mfwater/md/data/input_mfda.lmp` is the paper-I protocol plus `restart 100000 restart.*` (a restart file every 100 ps, named by timestep, so none is overwritten) and `run ... upto` (absolute timesteps 250 000 and 1 250 000: 0.25 ns equilibration and 1 ns production at 1 fs).
If the newest attempt of an evaluation has no finished analysis but has restart files, the next attempt `attempt_{k+1}` resumes from the newest one instead of starting over: it reads `siminp/input_resume.lmp` (generated; `read_restart`, then the settings a restart file does not store, same `fTHERMO` fix ID), and records the source in `attempt_{k+1}/resume_from.txt`.
Several interruptions in a row are followed back through `resume_from.txt`. A restart file is used only if its size equals that of the next-older one (all restart files of an evaluation have the same size); a smaller, probably truncated newest file (e.g. job killed while writing) is skipped with a `Warning:` and never modified or deleted, and without any valid restart file the evaluation starts fresh in a new attempt. This size check is a heuristic (see `select_restart_file`). Without any restart file the evaluation starts fresh in a new attempt, as before.
For a resumed evaluation the production trajectory is spread over several attempts; TRAVIS reads the new file `msd/prod_combined.lammpstrj` (frames of the earlier attempts below the restart step plus all frames of the resumed run, 1001 frames, spacing 1000 steps; the build fails if there is a gap).
`result.json` lists all attempts used in `attempt_dirs` (`attempt_dir` is the last one).
A resumed trajectory is not bitwise identical to an uninterrupted one (neighbour lists, PPPM, MPI layout); this is acceptable because every (N, theta) is evaluated once.
A run whose summary file exists is complete and is refused.

**Smoke test.** `mfwater -a mfda-smoke --forward-model md --molecules 32 --workdir ./smoke_run` runs one MD evaluation at the OPC3 parameters and prints D.

**Cluster job.** Put all run settings in a run TOML (`run.toml`) and generate the job script:
```
mfwater -a mfda-jobscript --cluster marvin -c run.toml [--jobscript run.sh]
```
The settings are validated (as in `markov-chain`), `--ntasks = chains * calc_cpus(N_1)` is checked against `cores_per_node` (all chains share one node), and the script calls `mfwater -a markov-chain -c <absolute path of run.toml>`.
`--lammps-cmd` is taken from the cluster profile unless `run.toml` sets `lammps_cmd` (the run TOML wins). The job name is the basename of `--workdir`.
The default output is `run-mfda-<cluster>.sh` in the current directory; an existing file is never overwritten. Restart = resubmit the script.
`mfwater -a mfda-ncpu --molecules 1000 500 100 --chains 4` prints the number of tasks only.

**Adding a cluster.** Add one TOML file (copy `src/mfwater/algo_mfda/data/clusters/marvin.toml`), or pass it with `--cluster-file my_cluster.toml`.
It defines `scheduler` (currently only `slurm`; the script comes from `data/jobscript_<scheduler>.sh.template`), `cores_per_node`, the scheduler table (`[slurm]`: `partition`, `account`, `time`, `extra`), `[environment] setup` (shell snippet; it should set `MFWATER` to the `mfwater` executable) and `[lammps] cmd` (with `{ncpu}` and `{input}`).
The Marvin profile needs fftool `d89546c` or newer (`--lmp`).
TRAVIS and msdiff run serially inside each chain.

**OrthoBoXY and msdiff.** With `--orthoboxy`, the MD forward model uses tetragonal OrthoBoXY boxes and the TRAVIS input `travis_input_msd_orthoboxy.txt` (two MSD observations, x-y and z).
msdiff is called as in paper I, so that MFDA values are comparable to paper-I data: `msdiff -f msd_H2O_#2.csv --from-travis --hummer 298.15 0.89e-3 0.0` (cubic) or `msdiff -f msd_H2O_#2_XY.csv --from-travis --hummer 298.15 0.89e-3 0.0 --orthoboxy msd_H2O_#2_Z.csv` (OrthoBoXY); the box lengths are read from `travis.log`.
The returned D is the `D_0` column (Hummer term K not added), converted from 1e-12 to 1e-9 m^2/s.
The cache stores the box shape and refuses to mix cubic and OrthoBoXY results; use a separate `--workdir` per box shape.
Several chains start `mpirun` concurrently in one allocation; use `--bind-to none` (default of the script) or `srun --exact -n {ncpu} lmp -i {input}` via `--lammps-cmd`. This is untested on Marvin.

## Notes
The generation of input files for LAMMPS involves external software, namely [fftool](https://github.com/paduagroup/fftool) and [packmol](https://m3g.github.io/packmol/).
After completion of the MD simulation, the diffusion coefficient of water is calculated externally, using [TRAVIS](http://www.travis-analyzer.de/) and [msdiff](https://github.com/kirchners-manta/msdiff).
For the sake of simplicity, the execution of these programs is not included in the code.


## Reference

This software accompanies two papers.
The MFMC forward problem is described in paper I, the multifidelity inverse problem (MFDA) in paper II.

1. T. Frömbgen, A. Kuhn, J. Dölz, B. Kirchner, [A Multifidelity Monte Carlo Approach for Simulating the Diffusion Coefficient of Water. I. Forward Problem](https://doi.org/10.1063/5.0308381). *J. Chem. Phys.* **2026**, 164, 024504.
2. T. Frömbgen, A. Kuhn, J. Dölz, B. Kirchner, A Multifidelity Monte Carlo Approach for Simulating the Diffusion Coefficient of Water. II. Inverse Problem. *In preparation.*
