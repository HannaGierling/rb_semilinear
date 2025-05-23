from fenics import *
from utils import str2pathstr

import numpy as np

def comp_POD(S:np.array, N_max:int, tol:float, 
        M:np.array = None, 
        folder:str = None, plot:bool = False, ret_sigma:bool = False):
    """
    Perform Proper Orthogonal Decompositon (POD) on given snapshot matrix 'S'.

    Computes a reduced basis using singular value decompositon (SVD) of
    snapshot matrix 'S', optionally w.r.t. mass matrix 'M'

    Parameters
    ----------
    S : np.array
        Snapshot matrix with one snapshot per column
    N_max : int
        maximum number of basis vectors
    tol : float
        Tolerance
    M : np.array, optional
        mass matrix
    folder : str, optional
        folder in which to save singular value plot and csv-file
    plot : bool, optional
        If True, displays the singular value plot, by default False
    ret_sigma : bool, optional
        If True, returns singular values in addition to reduced basis, by default False
    
    Returns
    -------
    V : np.array
        Matrix of POD modes 
    sigma : np.array, optional
        singular values - only returned if 'ret_sigma' is True.
    """

    # --- compute singular values 'sigma' and reduced basis 'V' --- #
    if M is not None:
        EV, sigma_sq, Z = np.linalg.svd(S.T @ M @ S)           #entspr. sigma, V = LA.eigh(S.T@M@S)
        sigma = np.sqrt(sigma_sq)
        factor = 1/sigma
        SEV = S @ EV
        V = np.stack([factor[i]*SEV[:,i] for i in range(len(factor))],axis=1)
    else:
        V, sigma, Z = np.linalg.svd(S)           

    # --- determine N --- #
    if tol > 0.0:
        ratio = np.cumsum(sigma**2)/np.sum(sigma**2)
        for id, x in enumerate(ratio):
            N = id
            if 1.0-x <= tol**2:
                break
    else : 
        N = min([N_max,len(sigma)])

    # --- save singular values in csv file and plot --- #
    if folder is not None: 
        import matplotlib.pyplot as plt
        import os
        import pandas as pd
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # --- csv-file --- #
        df = pd.DataFrame({"sigma":sigma}); df['i'] = df.axes[0]
        df.to_csv(f"{folder}/sigma_Nmax_{N_max}_tol_"+str2pathstr(f"{tol:.2e}")+".csv", 
                  sep='\t', encoding='utf-8',index=False, header=True)

        # --- plot --- #
        plt.semilogy(range(len(sigma)), sigma)
        plt.xlabel("i"); plt.ylabel(u'${\sigma_i}$')
        #plt.title(f"Singular values of snapshot matrix \n N={N} for tol={tol:.3e}, uperrbound={uperrbound:.3e}")
        plt.title(f"Singular values of snapshot matrix \n N={N} for tol={tol:.3e}")
        plt.savefig(f"{folder}/SingVals.png")
        plt.show() if plot==True else plt.close()

    if ret_sigma == True:
        return V[:, 0:N], sigma[0:N]
    return V[:, 0:N]
