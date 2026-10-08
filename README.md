# mfwater
---

![Python versions](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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

The `build` algorithm can be executed several times, but the output file will be overwritten if it already exists (question will be asked to the user).

### chemmod-prep
This algorithm generates the directories and input files for the MD simualtion using LAMMPS for each model and evaluation.
To do so, it uses template input files available [here](./src//mfwater//algo_chemical_model/data/), modifies them accordingly, and uses [fftool](https://github.com/paduagroup/fftool) and [packmol](https://m3g.github.io/packmol/) to create the LAMMPS data files.
Specifically, the program draws random Lennard-Jones (LJ) parameters from a Gaussian distribution centered at the original values of the OPC3 force field with a standard deviation of 1/120 of the original value for $\epsilon$ and $\sigma$.
Additionally, random seeds are generated for the LAMMPS velocity command and the PACKMOL input file.

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

The `chemmodel-prep` algorithm cannot be executed several times on the same input file, as it will overwrite the existing files and draw new random parameters.

### chemmod-post
This algorithm postprocesses the results of the MD simulations.
```bash
mfwater -a chemmodel-post -i big_oms.hdf5
```
It looks for the output files of LAMMPS and msdiff in the `simout/log.lammps` and `msd/misdiff_out.csv` folders/files of the models and evaluations directories, to extract the computation time and the diffusion coefficient, respectively.
For each model, the dataset `diffusion_coeff` is added to the model group, containing the diffusion coefficient (in $10^{-12}\,\text{m}^2\,\text{s}^{-1}$) for each evaluation.
Additionally, each model will be given an attibute `computation_time` with the averaged computation time (in s) of the evaluations.
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

The `model-select` algorithm can be executed several times on the same input file, but it will overwrite the existing models and parameters.

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
The MFMC estimator and its error are added as attributes `mfmc_estimator` `mfmc_error` to the input file.
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
 attribute  /models/mfmc_error
 attribute  /models/mfmc_estimator
 attribute  /models/n_models
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
The `mfmc` algorithm can be executed several times on the same input file, but it will overwrite the existing attributes.

### markov-chain
Multifidelity delayed acceptance MCMC (MFDA, after [Lykkegaard et al.](https://arxiv.org/abs/2202.03876)) for the parameters `(epsilon_OO, sigma_OO, q_O)` of OPC3.
Level 1 is the finest (largest `N`), the last level the coarsest.
```bash
mfwater -a markov-chain --forward-model md --workdir ./mfda_run --chains 4 \
    --models 3 --molecules 1000 500 100 --mcchainlength 1000 --mcsubchainlength 10 \
    --mcburnin 100 --params lj --seed 1
```
Arguments (defaults in brackets):
- `--forward-model {dummy,md}` [`dummy`]: cheap stand-in or the MD pipeline (fftool, packmol, LAMMPS, TRAVIS, msdiff).
- `--workdir` [`./mfda_run`]: run directory, see below.
- `--chains` [1]: independent chains, run concurrently as separate processes. Chain `c` uses `np.random.default_rng([seed, c])` and its own cache.
- `--models`, `--molecules`: number of levels and molecules per level (strictly descending).
- `--mcchainlength` [1000]: steps `M_1` on the finest level. `--mcburnin` [100]: discarded fine steps (all levels).
- `--mcsubchainlength` [10]: subchain lengths for levels `2..n_models`; one value is used for all levels, otherwise `n_models - 1` values.
- `--params {lj,q,lj-q}` [`lj`]: calibrated parameters, the others stay at the OPC3 value.
- `--seed` [random, printed and stored in the manifest], `--orthoboxy` (needs OrthoBoXY support of the MD model), `--lammps-cmd` [`mpirun -np {ncpu} lmp -i {input}`].
- `-o` [`default.hdf5`]: name of the combined summary file inside `--workdir` (an absolute path is used as is).

Run directory:
```
manifest.json                 arguments, seed, package version, git hash
cache/chain_000/N{n}/{key_hash}/result.json
cache/chain_000/N{n}/{key_hash}/attempt_{k}/{siminp,simout,msd}/
chain_000.hdf5, ...           samples/proposals per level, estimator
default.hdf5                  per-chain estimators, mean, between-chain standard deviation
```
Nothing is overwritten or deleted. Existing result files are refused.

**Restart.** Rerun the same command (on Marvin: resubmit the same job script).
Chains with an existing `chain_{c:03d}.hdf5` are skipped; the others are replayed from the start with the same seed, and every evaluation with a `result.json` in the cache returns instantly (the number of cached/computed evaluations is printed per chain).
The arguments and the package version/git hash are compared with `manifest.json`; on a mismatch the run aborts (use a new `--workdir`).
An interrupted MD evaluation leaves its `attempt_{k}` directory in place and is recomputed in a new attempt.
A run whose summary file exists is complete and is refused.

**Smoke test.** `mfwater -a mfda-smoke --forward-model md --molecules 32 --workdir ./smoke_run` runs one MD evaluation at the OPC3 parameters and prints D.

**Cluster job.** `src/mfwater/algo_mfda/data/run-mfda-marvin.sh` is a template (partition `intelsr_long`, 7 days).
`mfwater -a mfda-ncpu --molecules 1000 500 100 --chains 4` prints `chains * calc_cpus(N_1)` for `--ntasks`.
TRAVIS and msdiff run serially inside each chain.
Several chains start `mpirun` concurrently in one allocation; use `--bind-to none` (default of the script) or `srun --exact -n {ncpu} lmp -i {input}` via `--lammps-cmd`. This is untested on Marvin.

## Notes
The generation of input files for LAMMPS involves external software, namely [fftool](https://github.com/paduagroup/fftool) and [packmol](https://m3g.github.io/packmol/).
After completion of the MD simulation, the diffusion coefficient of water is calculated externally, using [TRAVIS](http://www.travis-analyzer.de/) and [msdiff](https://github.com/kirchners-manta/msdiff).
For the sake of simplicity, the execution of these programs is not included in the code.


## Reference

T. Frömbgen, A. Kuhn, J. Dölz, B. Kirchner, [A Multifidelity Monte Carlo Approach for Simulating the Diffusion Coefficient of Water. I. Forward Problem](https://doi.org/10.1063/5.0308381). *J. Chem. Phys.* **2026**, 164, 024504.
