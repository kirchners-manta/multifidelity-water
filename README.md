# mfwater
---

![Python versions](https://img.shields.io/badge/python-3.10%20|%203.11-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


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
- `chemmod-prep`: Preparations for the chemical model (i.e., generating input files for LAMMPS)
- `chemmod-post`: Postprocessing of the chemical model (i.e., calculating the MSD and the diffusion coefficient)

The program operates on an HDF5 file containing information on the models.
The current file structure is (omitting attributes):
```
models
    model_1
    model_1/lj_params
    model_2
    model_2/lj_params
...
```
where `models` and `model_{n}` are groups and `lj_params` is a dataset containing the Lennard-Jones parameters for the model.
The `models` group has the attribute `n_models` and for each `model_{n}` group there are attributes `n_molecules` and `n_evals` (the number of molecules and evaluations for the model).
The `lj_params` dataset is of dimensions `2, n_evals` and contains the Lennard-Jones parameters, $\sigma$ and $\epsilon$, for the model and each evaluation.

`lj_params` is obtained by adding a Gaussian noise to the original values, taken from the [OPC3](https://doi.org/10.1063/1.4960175) force field.
For $\epsilon$, the Gaussian was chosen to be centered at the original values with a standard deviation of 1/30 of the original value, to make sure that 99.7% of the values are in the range of $\pm 10 \%$ of the original value.
For $\sigma$, a tighter Gaussian was chosen ($\pm 5 \%$ of the original value), because a small $\sigma$ leads to a large force and lets the simulation explode.

When executing the `chemmod-prep` algorithm, the program will generate a default .hdf5 input file if not given any input file.
It will always create input files for LAMMPS for each model and evaluation.

Upon executing the `chemmod-post` algorithm, the program will look for the output files of LAMMPS and msdiff in the respective folders of the models and evaluations.
For each model, it adds a dataset `diffusion_coeff` to the model group, containing the diffusion coefficient (in $10^{-12}\,\text{m}^2\,\text{s}^{-1}$) for each evaluation.
Additionally, each model will be given an attibute `computation_time` with the averaged computation time (in s) of the evaluations.
Afterwards, the input file will have the following structure (omitting attributes):
```
models
    model_1
    model_1/lj_params
    model_1/diffusion_coeff
    model_2
    model_2/lj_params
    model_2/diffusion_coeff
...
```
The program also checks whether the LJ parameters given in the input file are identical to the ones used in the LAMMPS simulations (and throws an error if not).

## Implementation hints
@CodingAllan, to implement your algorithms, I recommend following the structure that I used [here](./src/mfwater/algo_chemical_model/).
You can just create your own folder and add code to it.
Other options to the argparser can be added [here](./src/mfwater/argparser/argparser.py)
Please comment all your code and add docstrings to all your functions.
If you find something in my code that you don't understand, please let me know and I will improve it.


## Notes
The generation of input files for LAMMPS involves external software, namely [fftool](https://github.com/paduagroup/fftool) and [packmol](https://m3g.github.io/packmol/).
After completion of the MD simulation, the diffusion coefficient of water is calculated externally, using [TRAVIS](http://www.travis-analyzer.de/) and [msdiff](https://github.com/kirchners-manta/msdiff).
For the sake of simplicity, the execution of these programs is not included in the code.
