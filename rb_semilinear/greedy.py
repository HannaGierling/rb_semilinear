from fenics import * 
from utils import *

import os 
import numpy as np

#def proj_error(V_h,u_dof:np.ndarray, V:np.ndarray, X:np.ndarray):
def _proj_error(u_dof:np.ndarray, V:np.ndarray, X:np.ndarray):
    err_dof = u_dof - V @ V.T @ X @ u_dof
    return np.sqrt(err_dof.T @ X @ err_dof)


def _gram_schmidt(V:np.ndarray, u_dof:np.ndarray, X:np.ndarray):
    """Orthonormalisiere u gegen V_list bzgl. Skalarprodukt mit M-Matrix"""
    """vgl Quateroni"""
    if V.size == 0:
        v = u_dof
    else:
        proj = V@ (V.T @ (X @ u_dof))
        v = u_dof - proj
    norm_v = np.sqrt(np.dot(v, X @ v))
    v /= norm_v
    return v

def comp_greedyRB(S: np.ndarray,
           M: np.ndarray,
           P_train:np.ndarray, mu_0:float|None=None,
           N_max: int = 10,
           greedy_tol: float = 1e-6,
           folder:str = None, 
           ret_Pused:bool = False, ret_errors:bool = False
           ):
    """
    Constructs a reduced basis (RB) using Greedy algorithm.
    (vgl. Alg. 7.1. in Quateroni)

    Parameters
    ----------
    S : np.ndarray
        Snapshot matrix (snapshot per column corresponding to μ in P_train)
    M : np.ndarray
        mass matrix 
    P_train : np.ndarray
        Training parameter set
    N_max : int, optional
        Maximum Number of basis vectors, by default 10.
    greedy_tol : float, optional
        Tolerance for projection error, by default 1e-6
    folder : str, optional
        If not None, saves RB and parameter info to this folder, by default None
    ret_Pused: bool, optional
        If True, returns also selected parameter values, by default False
    ret_errors : bool, optional
        If True, projection errors of remaining parameter values are returned.

    Returns
    -------
    V : np.ndarray
        Matrix with reduced basis vectors in columns
    P_trained : np.ndarray
        Selected parameter values (only if ret_Ptrained == True)
    """
    # --- check whether RB already exists --- #
    if os.path.exists(f"{folder}/RB_Greedy.csv") and os.path.exists(f"{folder}/P_used.csv"):
        V = np.loadtxt(f"{folder}/RB_Greedy.csv", delimiter=',')
        P_used = np.loadtxt(f"{folder}/P_used.csv", delimiter=',')
        if ret_Pused == True:
            return V, P_used
        else:
            return V

    V           = []                # reduced basis
    P_used      = []                # parameter values selected by Greedy
    P_ids_used  = []                # indeces of selected param.vals in P_train
    eps         = greedy_tol + 1    # initial error value

    Ps = P_train.copy()

    # --- select first parameter ---#
    if mu_0 is None:
        id_muN = 0
        muN    = Ps[id_muN]
    else:
        id_muN = np.where(np.isclose(P_train, mu_0))[0][0]
        muN    = mu_0
        
    # --- Greedy iterations --- #
    N         = 0
    N_max_bar = len(P_train)
    while N < min(N_max, N_max_bar) and eps > greedy_tol:

        # --- report --- #
        print(f"[{N}] μ = {muN:.4e}, error = {eps:.2e}")

        # --- orthonormalize snapshot corresp. to 'mu' and add to RB 'V' --- #
        u_mu_dof = S[:, id_muN]
        V.append(_gram_schmidt(np.array(V).T, u_mu_dof, M))
        P_used.append(muN)

        # --- delete trained 'mu' and corresponding snapshot --- #
        S = np.delete(S, id_muN, axis=1)
        Ps = np.delete(Ps, id_muN)
        #P_ids_ntrained = np.delete(P_ids_ntrained, id_mu)
        P_ids_used.append(id_muN) 

        # --- select μ with max. projection error --- #
        errors = [_proj_error(S[:,id], np.array(V).T,M) for id in range(len(Ps))]
        id_muN, eps = max(enumerate(errors), key=lambda x: x[1])
        muN = Ps[id_muN]

        N += 1
    
    # --- report --- #
    print(f"max. error = {eps:.2e} (for μ = {muN:.4e})")

    V = np.stack([v for v in V], axis=1)
    # --- flags indicating whether corresp. param. in P_train was selected 
    P_used_flags = np.zeros_like(P_train)
    P_used_flags[P_ids_used] = 1 
    
    if folder is not None:
        os.makedirs(folder, exist_ok=True)
        np.savetxt(f"{folder}/RB_Greedy.csv", V, delimiter=',', fmt='%.12e')       
        np.savetxt(f"{folder}/P_used.csv", P_used, delimiter=',', fmt='%.12e')
        np.savetxt(f"{folder}/P_used_flags.csv",
                   np.stack((P_train,P_used_flags), axis=1),delimiter=',')
    
    if ret_Pused == True:
        if ret_errors == True:
            return V, P_used, errors
        else :
            return V, P_used
    else:
        if ret_errors == True:
            return V, errors
        else :
            return V




