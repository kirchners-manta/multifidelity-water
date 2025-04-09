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
- `chemmod-prep`: Preparations for the chemical model (i.e., generating input files for LAMMPS)
- `chemmod-post`: Postprocessing of the chemical model (i.e., calculating the MSD and the diffusion coefficient)

The program operates on an HDF5 file containing information on the models.
The current file structure is
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

**Attention:** `lj_params` is obtained by adding a Gaussian noise to the original values, taken from the [OPC3](https://doi.org/10.1063/1.4960175) force field. 
The Gaussian was chosen to be centered at the original values with a standard deviation of 1/6 of the original value, to make sure that 99.7% of the values are in the range of 0.5 to 1.5 times the original value. 

## Implementation hints 
@CodingAllan, to implement your algorithms, I recommend following the structure that I used [here](./src/mfwater/algo_chemical_model/). 
You can just create your own folder and add code to it.
Other options to the argparser can be added [here](./src/mfwater/argparser/argparser.py)
Please comment all your code and add docstrings to all your functions.
If you find something in my code that you don't understand, please let me know and I will improve it.


## Notes
The generation of input files for LAMMPS involves external software, namely [fftool](https://github.com/paduagroup/fftool) and [packmol](https://m3g.github.io/packmol/).