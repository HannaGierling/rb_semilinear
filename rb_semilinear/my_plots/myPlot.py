import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def _float_is_in(val:float, lst:list[float], tol:float=1e-9):
    for x in lst:
        if abs(x-val) < tol:
            return True
    return False

def _filter_Mus(Mus_wanted:list[float], Mus_all:list[float], 
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
        if not _float_is_in(mu, Mus_wanted):
            continue

        converged = int(convergence_flags[id])
        if converged_only and not converged:
            continue
        
        ids_found.append(id)
        mus_found.append(mu)
        convsflags_found.append(converged)
    
    return ids_found, mus_found, convsflags_found

def myPlot(path:str, Ns_2plot:list[int], Mus_2plot:list[float], 
           converged_only:bool, 
           plotfilename=None, plot=False,
           csvfile = None):
    """
    Plots solutions for selected N and μ values in 'Ns_2plot' and 'Mus_2plot'. 

    Parameters
    ----------
    path : str
        _description_
    Ns_2plot : list[int]
        _description_
    Mus_2plot : list[float]
        _description_
    converged_only : bool
        _description_
    plotfilename : _type_, optional
        _description_, by default None

    """

    if not os.path.exists(f"{path}/mapping.log"):
        raise FileNotFoundError(f"'mapping.log' not found in path {path}")

    df = pd.read_csv("%s/mapping.log"%(path), sep=r'\s+', engine='python')

    for N in Ns_2plot:
        dfN = df[df['N'] == N]
        if dfN.empty:
            continue

        Mus_all = dfN['μ'].unique().tolist()

        conv_all = dfN['converged'].tolist()
        ids_MusFound, MusFound, convs_MusFound = _filter_Mus(Mus_2plot,Mus_all,
                                                            conv_all,
                                                            converged_only)

        x = np.loadtxt(f"{path}/x_{N}.csv", delimiter=',')

        S = np.loadtxt(f"{path}/S_{N}.csv", delimiter=',')
        if S.ndim == 1 and len(MusFound) == 1:
            plt.plot(x, S, label=f"μ={MusFound[0]:.3e}, conv={convs_MusFound[0]}")
            plt.title("Solutions"); plt.xlabel("x"); plt.ylabel("u(μ)")
            plt.legend()
        else:
            S = S[:, ids_MusFound]
            [plt.plot(x, S[:, i], label=f"μ={MusFound[i]:.3e}," +\
                    f"conv={convs_MusFound[i]}") for i in range(len(MusFound))]
            plt.title("Solutions"); plt.xlabel("x"); plt.ylabel("u(μ)")
            plt.legend()
    
    if csvfile is not None:
        os.makedirs(os.path.dirname(csvfile), exist_ok=True)
        header = {f"mu={MusFound[i]:.2e}":S[::10,i] for i in range(len(MusFound))}
        header.update({"x":x[::10]})
        df = pd.DataFrame(header)
        df.to_csv(csvfile, mode="w", header=True, sep=',', 
                    index=False, float_format='%.12e')

    if plotfilename is not None:
        plt.savefig(f"{path}/{plotfilename}")
    if plot == True:
        plt.show()
    else: 
        plt.close()



    
 