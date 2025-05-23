from fenics import *
import numpy as np
from typing import Literal, Any

def filter_Mus(Mus_wanted:list[float], Mus_all:list[float], 
               convergence_flags:list[int], converged_only:bool):
    """
    Filters values from Mus_wanted that are present in Mus_all and 
    fulfill the criterion 'converged_only'.

    Parameters
    ----------
    Mus_wanted : list[float]
        Values to find in 'Mus_all'
    Mus_all : list[float]
        List of values to be filtered
    convergence_flags : list[int]
        List of convergene flags corresponding to Mus_all
    converged_only : bool
        If True, only return μ that have converged

    Returns
    -------
    ids_found : list of int
        Indices for μ of 'Mus_wanted' in 'Mus_all'
    mus_found : list of float
        The μ values of 'Mus_wanted' that are in 'Mus_all' and satisfy 'converged_only'.
    convs_found : list of int
        convergence flags corresponding to 'mus_found'
    """

    ids_found = [] 
    mus_found = [] 
    convsflags_found = []

    for id, mu in enumerate(Mus_all):
        if not float_is_in(mu, Mus_wanted):
            continue

        converged = int(convergence_flags[id])
        if converged_only and not converged:
            continue
        
        ids_found.append(id)
        mus_found.append(mu)
        convsflags_found.append(converged)
    
    return ids_found, mus_found, convsflags_found

def str2pathstr(string:str):
    """
    modifies given string s.t. it becomes suitable for a path-string

    This function removes/replaces characters that are problematic in paths
    """
    s = f"{string}"\
        .replace(" ","").replace("'","").replace("{","").replace(":","")\
        .replace("}","").replace(".","").replace("-","m")
    return s

def w2file( filename:str, text:Any, mode:Literal["x","a","w"] = "a", 
           float_format = None):
    
    """
    Write given text/DataFrame/dictionary to a file
    """
    import pandas as pd
    import os

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if isinstance(text, pd.DataFrame):
        file_exists = os.path.exists(filename)
        if mode == "a":
            text.to_csv(filename, mode='a', header=not file_exists, sep=' ', 
                        index=False, float_format=float_format)
        else:
            text.to_csv(filename, mode=mode, header=True, sep=' ', 
                        index=False, float_format=float_format)

    elif isinstance(text, dict):
        df = pd.DataFrame(text)
        file_exists = os.path.exists(filename)
        if mode == "a":
            df.to_csv(filename, mode='a', header=not file_exists, sep=' ', 
                      index=False, float_format=float_format)
        else:
            df.to_csv(filename, mode=mode, header=True, sep=' ', 
                      index=False, float_format=float_format)

    else:
        with open(filename, mode, encoding="utf-8") as log_file:
            print(text, file=log_file)

def appendDict(dict1:dict, dict2:dict):
    """
    Appends the values of 'dict2' to corresponding arrays in 'dict1'

    Parameters
    ----------
    dict1 : dict
        dictionary with numpy arrays as values (will be modified)
    dict2 : dict
        dictionary with values to append to dict1 arrays
    """
    if not dict1:
        for key, val in dict2.items():
            dict1[key] = np.append([], dict2[key])
    else:
        for key, val in dict1.items():
            dict1[key] = np.append(val, dict2[key])

def indeces_of(val:float, lst:list[float],tol:float=1e-9):
    """
    Find indeces of given value 'val' ind given list 'lst'.
    

    Parameters
    ----------
    val : float
        Value to search for
    lst : list[float]
        list of values to search within
    tol : float
        tolerance for comparing values

    Returns
    -------
    list
        list of indices
    """
    ids = []
    for i, x in enumerate(lst):
        if abs(x-val) < tol:
            ids.append(i)
    
    return ids 

def float_is_in(val:float, lst:list[float], tol:float=1e-9):
    for x in lst:
        if abs(x-val) < tol:
            return True
    return False
