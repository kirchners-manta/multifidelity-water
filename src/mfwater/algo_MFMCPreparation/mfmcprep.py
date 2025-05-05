import h5py
import numpy as np
from typing import Tuple
from algo_input import build_default_input
from algo_chemical_model import chemical_model_post, chemical_model_prep
import argparse
from pathlib import Path
import math

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

def multifidelity_preparation(path: str | Path) -> int:
    """

    Args:
        path (str | Path): Path to hdf5 file with the multifidelity hierarchy

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: returns the variances, correlation coefficients and weights. 
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input file '{path}' does not exist.")

    with h5py.File(path , "r+") as f:
        models = f["models"]
        
        highfidelity_model = list(models.values())[0]
        highfid_diffs = np.array(highfidelity_model["diffusion_coeff"][:])
        n_eval = highfidelity_model.attrs["n_eval"]
        if n_eval != len(highfid_diffs):
            raise ValueError("FILE CORRUPTED: the attribute 'n_eval' does not correspond with the actual number of evaluations.")
        highfid_mc_estim = np.mean(highfid_diffs)
        for k, mod in enumerate(models):
            if n_eval != models[mod].attrs["n_eval"]:
                raise ValueError("INVALID EVALUATIONS: In the MFMC preparation stage, every model gets evaluated the same number of times!")
            if k>0:
                diffs =  np.array(models[mod]["diffusion_coeff"][:])
                if n_eval != len(diffs):
                    raise ValueError("FILE CORRUPTED: the attribute 'n_eval' does not correspond with the actual number of evaluations.")
                mc_estim = np.mean(diffs)
                f["models"][mod].attrs["variance"] = np.mean((diffs - mc_estim) ** 2)**0.5
                corr = np.mean((highfid_diffs - highfid_mc_estim) * (diffs - mc_estim))
                corr /= models[mod].attrs["variance"] * models[highfidelity_model].attrs["variance"]
                f["models"][mod].attrs["correlation"] = corr
            else:
                f["models"][mod].attrs["variance"] = np.mean((highfid_diffs - highfid_mc_estim) ** 2)**0.5
                f["models"][mod].attrs["correlation"] = 1.0
    return 0

def OptimalModel_select(path: str | Path) -> int:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input file '{path}' does not exist.")
    
    with h5py.File(path, "r") as f:
        models = f["models"]
        to_order = models.attrs["n_models"] -1
        model_items = [(name, group) for name, group in models.items() if isinstance(group, h5py.Group)]
        for _, group in model_items:
            if "correlation" not in group.attrs or "computation_time" not in group.attrs or "variance" not in group.attrs:
                raise KeyError(f"Missing required attributes ('correlation', 'computation_time', or 'variance') in model {name}")
        
        sorted_models = sorted(model_items, key=lambda x: x[1].attrs["correlation"], reverse=True)
        permutation = [name for name, _ in sorted_models]
        if permutation[0] != model_items[0][0]:
            raise ValueError("\rho_{1,1} is always the largest. Something is wrong with the correlation coefficients!")
        highfidelity_model = models[model_items[0][0]]
        v_star = highfidelity_model.attrs["variance"]**2 
        M_star = [model_items[0][0]] #List of optimal model names
        for z in range(to_order+1): 
            #Go over all subsets of size 'z'
            order_index = [to_order-z+j for j in range(1,z+1)] #Start with largest possible index
            c= [ 0 for k in range(z)]
            for j in range(math.comb(to_order,z)): #Iterate over the set ${(i_1,...,i_z)\in {1,...,to_order}^z: i_1 <i_2<...<i_2}$
                for k in range(z):
                    if c[k]>= math.comb(order_index[k]-1,k):
                        if order_index[k] >k+1:
                            order_index[k] -= 1
                            order_index[:k] = [order_index[k]-k+j for j in range(k)]
                            c[:k+1]= [0 for j in range(k+1)]
                c = [a+1 for a in c ]
                cur_models  = [model_items[0][0]] + [permutation[order_index[k]] for k in range(z)] #current models that satisfy the condition on the correlations
                weights = np.array([models[cur_models[i]].attrs["computation_time"] for i in range(z+1)])
                correlation = [models[cur_models[i]].attrs["correlation"] for i in range(z+1)] + [0]
                differences = np.array([(correlation[q]**2 - correlation[q+1]**2) for q in range(z+1)])
                skip_model = False
                for i in range(1,z+1):
                    if weights[i-1]*differences[i]<= weights[i]* differences[i-1]:
                        skip_model = True #This model does not satisfy the assumptions of the Theorem. Skip this selection
                        break
                if skip_model:
                    continue
                current_mse = (highfidelity_model.attrs["variance"]**2/weights[0])*np.dot(np.sqrt(weights),np.sqrt(differences))**2 #Compute MSE given optimal coefficients
                if current_mse < v_star:
                    M_star = cur_models
                    v_star = current_mse
        with h5py.File("optimal_models.hdf5", 'a') as g:
            if "models" not in g:
                g.create_group("models")
            g["models"].attrs["n_models"]= models.attrs["n_models"]
            for k,(name, mod) in enumerate(models.items()):
                if name not in M_star:
                    continue
                if name in g["models"]:
                    del g["models"][name]
                f.copy(mod,g["models"],name=name )
    return 0
            
            
