import h5py
import numpy as np
from typing import Tuple
from algo_input import build_default_input
from algo_chemical_model import chemical_model_post, chemical_model_prep
import argparse
from pathlib import Path

def setup_params(args: argparse.Namespace)-> str | Path:
    """
    Prepares input files and runs chemical model preparation/postprocessing steps.

    Args:
        args (argparse.Namespace): Parsed command-line arguments with keys like
            'output', 'n_models', 'n_molecules', and model evaluations 'm'.

    Returns:
        str | Path: Path to the generated input file.
    """
    args_forhdf5 = argparse.Namespace(output= args.output , n_models = args.n_models, n_molecules= args.n_molecules ,n_evals = args.m)
    build_default_input(args=args_forhdf5)
    args_forCM = argparse.Namespace(input=args_forhdf5.output )
    chemical_model_prep(args_forCM)
    chemical_model_post(args_forCM)
    
    return args_forCM.input 

def multifidelity_preparation(path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    Args:
        path (str | Path): Path to hdf5 file with the multifidelity hierarchy

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: returns the variances, correlation coefficients and weights. 
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input file '{path}' does not exist.")

    with h5py.File(path , "r") as f:
        models = f["models"]
        variances = np.empty(models.attrs["n_models"], dtype=np.float64)
        correlation_coeffs = np.empty(models.attrs["n_models"], dtype=np.float64)
        weights = np.empty(models.attrs["n_models"], dtype=np.float64)
        
        highfidelity_model = list(models.values())[0]
        highfid_diffs = np.array(highfidelity_model["diffusion_coeff"][:])
        n_eval = highfidelity_model.attrs["n_eval"]
        if n_eval != len(highfid_diffs):
            raise ValueError("FILE CORRUPTED: the attribute 'n_eval' does not correspond with the actual number of evaluations.")
        highfid_mc_estim = np.mean(highfid_diffs)
        for k, mod in enumerate(models):
            if n_eval != models[mod].attrs["n_eval"]:
                raise ValueError("INVALID EVALUATIONS: In the MFMC preparation stage, every model gets evaluated the same number of times!")
            weights[k] = models[mod].attrs["computation_time"]
            if k>0:
                diffs =  np.array(models[mod]["diffusion_coeff"][:])
                if n_eval != len(diffs):
                    raise ValueError("FILE CORRUPTED: the attribute 'n_eval' does not correspond with the actual number of evaluations.")
                mc_estim = np.mean(diffs)
                variances[k] = np.mean((diffs - mc_estim) ** 2)**0.5
                corr = np.mean((highfid_diffs - highfid_mc_estim) * (diffs - mc_estim))
                corr /= variances[0] * variances[k]
                correlation_coeffs[k] = corr
            else:
                variances[0] = np.mean((highfid_diffs - highfid_mc_estim) ** 2)**0.5
                correlation_coeffs[0] = 1.0
        return variances , correlation_coeffs, weights
            
            
            
