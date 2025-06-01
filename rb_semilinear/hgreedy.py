from fenics import * 
from .utils import float_is_in, w2file

from rb_semilinear.sl_problems import MySemilinearProblem
from rb_semilinear.greedy import comp_greedyRB
from rb_semilinear.parameter import   get_P
from rb_semilinear.snapshots import comp_S

import os 
import shutil
import numpy as np
from typing import Literal
import pandas as pd
import matplotlib.pyplot as plt

def _greedy(hf_problem:MySemilinearProblem,
            P_train:np.ndarray, mu_0:float, 
            N_max:int, greedy_tol:float, 
            Bl_folder:str):

    # --- compute and save snapshot matrix --- #
    S, _, P_conv = comp_S(hf_problem, P_train)

    os.makedirs(Bl_folder, exist_ok=True)

    if len(P_conv) <  len(P_train):
        #raise Exception("hf_problem did not converge for all μ in P_train!")
        print("WARNING: hf_problem did not converge for all μ in P_train!")
    np.savetxt(f"{Bl_folder}/S.csv", S, delimiter=',', fmt='%.12e')       

    # --- mass matrix --- #
    m = inner(TrialFunction(hf_problem.V_h),TestFunction(hf_problem.V_h))*dx
    X = assemble(m).array()

    V, M, errors = comp_greedyRB(S, X, P_conv, mu_0, N_max, greedy_tol, 
                                 Bl_folder, ret_Pused=True, ret_errors=True) 
    
    return V, M, errors

    
# --- proximity function --- #
def _proximity(amu_Bl:float, mu:float, opt:Literal["lin", "log"]):

    # --- measure distance of Exponents for Basis 10  --- #
    if opt == "log":
        exp_amu_Bl = np.log10(amu_Bl)
        exp_mu = np.log10(mu) 
        d = abs(exp_amu_Bl-exp_mu)

    # ---  measure eukl. distance of Values --- #
    elif opt == "lin":
        d = abs(mu-amu_Bl) 

    else:
        raise Exception(f"opt = {opt} not implemented!")

    return d

def _get_amu(Bl:str, folder:str):
    """
    Find value of anchor parameteter μ_Bl corresponding to boolean vector 'Bl' 
    from file 'amus.log' in directory 'folder'.

    Parameters
    ----------
    Bl : str
        Boolean vector 

    folder : str
        Directory with file 'amus.log' to search through.

    Returns
    -------
    amu : float
    """

    df = pd.read_csv(f"{folder}/amus.log", sep=r'\s+', engine='python')

    Bls = df['Bl'].tolist()
    if type(Bls[0]) == str:
        tmp = df[df['Bl']==Bl]
    else:
        tmp = df[df['Bl']==int(Bl)]
    
    if tmp.empty:
        raise Exception(f"No anchor parameter found for boolean vector {Bl}\n"+\
         f"if you want to build a new RB, you might have to delete {folder}")

    return tmp['amu'].iloc[0]


def _find_Bl(mu:float, Bl:str, folder:str, proximity_opt:Literal["lin", "log"]):
    """
    Find boolean vector 'Bl' corresponding to given parameter value 'mu'
    by traversing a binary tree of anchor values.
    This is a recursive function, checking whether the given boolean vector 'Bl'
    corresponds to an anchor parameter that matches best to the given parameter
    value 'mu'. If this is the case, this given boolean vector 'Bl' is returned.
    Otherwise this function is recalled with a "child" of this boolean vector.

    Parameters
    ----------
    mu : float
        The target parameter value for which the boolean vector is searched.
    Bl : str
        current checked boolean vector.
    folder : str
        Directory path containing file 'amus.log' with anchor data.
    proximity_opt : Literal["lin", "log"]
        Distance metric to use

    Returns
    -------
    Bl : str
        Boolean vector string correspondin to best matching anchor  μ.
    """
    # --- select anchor μ of children --- #
    amu_Blp0 = _get_amu(Bl+"0",folder)
    amu_Blp1 = _get_amu(Bl+"1",folder)

    # --- termination --- #
    #if amu_Blp0 == 0 and amu_Blp1 == 0:
    if np.isnan(amu_Blp0) and np.isnan(amu_Blp1):
        return Bl

    # --- determine next "child" --- #
    if _proximity(amu_Blp0, mu, proximity_opt) < _proximity(amu_Blp1,mu, proximity_opt): 
        i_next = "0" 
    else: 
        i_next = "1"
    return _find_Bl(mu, Bl+i_next, folder, proximity_opt)

def find_Bl(mu:float, folder:str, proximity_opt:Literal["lin", "log"]):
    """
    Recursevly find a boolean vector 'Bl' corresponding to given parameter 
    value 'mu' by traversing a binary tree of anchor values.

    Parameters
    ----------
    mu : float
        The target parameter value for which the boolean vector is searched.
    folder : str
        Directory path containing file 'amus.log' with anchor data.
    proximity_opt: Literal["log", "lin"]
        Distance metric to use

    Returns
    -------
    Bl : str
        Boolean vector string correspondin to best matching anchor  μ.
    """
    Bl = "1"
    Bl_mu = _find_Bl(mu, Bl, folder, proximity_opt) 
    return Bl_mu
    

# --- h-type greedy algorithm (recursive function) --- #
def _greedy_htype(hf_problem:MySemilinearProblem,   # high-fidelity problem
                 P_Bl_train:np.array,               # current training parameter samples
                 eps_tol1:float,                    # error tolerance
                 N_bar:int,                         # max. N per leaf node
                 Bl:str,                            # current leaf node
                 folder:str,                        # folder path
                 P_discr_opt:Literal["lin", "log"], # options for discr. the Param.space
                 len_P_1    # length of initial parameter training set
                ):
    if len(P_Bl_train)<=N_bar:
        P_Bl_train = get_P([P_Bl_train[0],P_Bl_train[-1]], P_discr_opt, 2*N_bar) 
    # --- get current anchor μ_Bl --- #
    amu_1 = _get_amu(Bl, folder)
    # --- add anchor μ_Bl to current training parameter set --- #
    if not float_is_in(amu_1, P_Bl_train, tol=1e-12):
        P_Bl_train = np.append(P_Bl_train,amu_1)
        P_Bl_train = np.sort(P_Bl_train)

    # --- greedy for current subdomain P_'Bl' --- #
    V_Bl, M_Bl, errors_Bl = _greedy(hf_problem, P_Bl_train, amu_1,
                                    N_max=N_bar, greedy_tol=0.0, 
                                    Bl_folder=f"{folder}/{Bl}")
    
    # --- new children --- #
    Blp0 = Bl+"0";         Blp1 = Bl+"1"

    # --- check error tolerance --- #
    if max(errors_Bl) < eps_tol1:
        # --- safe anchor μ_(Bl,0)=μ_(Bl,1)=nan for termination criterion --- #

        w2file(f"{folder}/amus.log", {"Bl": [Blp0], "amu":[np.nan]}, mode="a")
        w2file(f"{folder}/amus.log", {"Bl": [Blp1], "amu":[np.nan]}, mode="a")
        return

    else:
        # --- delete parent domain --- #
        shutil.rmtree(f"{folder}/{Bl}", ignore_errors=True)

        # --- select and safe new anchor μs --- #
        amu_Blp0 = amu_1
        amu_Blp1 = M_Bl[1]
        w2file(f"{folder}/amus.log", {"Bl": [Blp0], "amu":[amu_Blp0]}, mode="a")
        w2file(f"{folder}/amus.log", {"Bl": [Blp1], "amu":[amu_Blp1]}, mode="a")

        # --- select training parameter sets for new leaf nodes --- #
        Ps_Blp0 = [];          Ps_Blp1= []
        Ps_Bl_tilde = get_P([P_Bl_train[0],P_Bl_train[-1]], 
                             P_discr_opt, 2*len_P_1)
        opt = P_discr_opt
        for mu in Ps_Bl_tilde:
            if _proximity(amu_Blp0, mu, opt) <= _proximity(amu_Blp1, mu, opt):
                Ps_Blp0.append(mu)
            else:
                Ps_Blp1.append(mu)

        # --- plot selected parameter values ---- #
        plt.scatter(Ps_Blp0, int(Blp0)*np.ones_like(Ps_Blp0), c='red')
        plt.scatter(amu_Blp0, int(Blp0)*1, c='lightcoral', marker='s',s=80)
        plt.scatter(Ps_Blp1, int(Blp0)*np.ones_like(Ps_Blp1), c='blue')
        plt.scatter(amu_Blp1, int(Blp0)*1, c='lightsteelblue', marker='s', s=80)
        plt.xscale(P_discr_opt if P_discr_opt=="log" else "linear"), plt.yscale('log')
        
        # --- report --- #
        print(f"Bl = [{Blp1}]")
        print("P_Bl = [",end="");[print(f"{mu:.2e}",end=" ") for mu in Ps_Blp1]; 
        print("]")
        print(f"len(P_Bl) = {len(Ps_Blp1)}")

        # --- recursive function call for right leaf node --- #
        _greedy_htype(hf_problem, Ps_Blp1, eps_tol1, N_bar, Blp1, folder, P_discr_opt,
                      len_P_1)

         # --- report --- #
        print(f"Bl = [{Blp0}]")
        print("P_Bl = [",end="");[print(f"{mu:.2e}",end=" ") for mu in Ps_Blp0]; 
        print("]")
        print(f"len(P_Bl) = {len(Ps_Blp0)}")

        # --- recursive function call for left leaf node --- #
        _greedy_htype(hf_problem, Ps_Blp0, eps_tol1, N_bar, Blp0, folder, P_discr_opt,
                      len_P_1)
             



# --- h-type greedy algorithm (recursive function) --- #
def greedy_htype(hf_problem:MySemilinearProblem,    # high-fidelity problem
                 P_train:np.ndarray,                 # training parameter samples
                 amu_1:float,                       # initial anchor parameter
                 eps_tol1:float,                    # error tolerance
                 N_bar:int,                         # max. N per leaf node
                 folder:str,                        # folder path
                 P_discr_opt:Literal["lin", "log"]  # options for discr. the Param.space
                 ):

    """
    Perform the h-type Greedy algorithm to adaptively construct a reduced basis model.

    The h-type Greedy algorithm
    selects representative parameter samples based on a given error tolerance
    and recursively refines the parameter space.

    Anchor points with corresponding Boolean Vector Bl are stored in file 'amus.log' in folder 'folder'. 

    The reduced basis and corresponding models are stored in files 'RB_Greedy.csv' 
    and 'P_used.csv' in a folder named after the corresponding Boolean Vector. 
    This folder is located in the folder 'folder'.

    Parameters
    ----------
    hf_problem : MySemilinearProblem
        The high-fidelity problem object
    
    P_train : np.ndarray
        Array of training parameter values used for the first Greedy sampling.
    
    eps_tol1 : float
        Error tolerance 
    
    N_bar : int
        Maximum number of basis vectors per leaf node
    
    folder : str
        Path to the folder where results (e.g. Models, anchor parameters) are saved.
    
    P_discr_opt : {'lin', 'log'}
        Discretization strategy for the parameter space. Can be linear or logarithmic.
    
    Examples
    --------
    >>> solver = MyNewtonSolver(1e-5,100,False,"nleqerr")
    >>> hf_problem = Fisher(1000,solver,"0.5")
    >>> folder = os.path.dirname(__file__)
    >>> P_1 = get_P([0.0001,1], "log", 50)
    >>> greedy_htype(hf_problem, P_1, 1e-4, 5, folder, "log")
    >>> Bl = find_Bl(mu=0.01, folder, "log")
    >>> V = np.loadtxt(f"{folder}/{Bl}/RB_Greedy.csv", delimiter=',')
    >>> rb_problem = MyRedNonlinearProblem(hf_problem, V, solver, "0.5")
    >>> rb_problem.proj_norm = "L2-norm"
    >>> SolInfo, NitInfos = rb_problem.solve(mu=0.01)

    Notes
    -----
    This method also generates and saves a plot visualizing the selected parameter discretization.
    """

    # --- length of initial parmaeter set --- #
    len_P_1 = len(P_train)

    # --- safe first anchor parameter μ_1 --- #
    w2file(f"{folder}/amus.log", {"Bl": ["1"], "amu":[amu_1]}, mode="w", 
           float_format = "%.12e")

    # --- plot selected parameter values ---- #
    plt.scatter(P_train, int(1)*np.ones_like(P_train), c='red')
    plt.scatter(amu_1, int(1)*1, c='lightcoral', marker='s',s=80)
    if P_discr_opt == "log": plt.xscale(P_discr_opt) 
    plt.yscale('log')

    # --- report --- #
    print(f"Bl = [1]")
    print("P_Bl = [",end="");[print(f"{mu:.2e}",end=" ") for mu in P_train]; print("]")
    print(f"len(P_Bl) = {len(P_train)}")

    # --- function call for right leaf node --- #
    _greedy_htype(hf_problem, P_train, eps_tol1, N_bar, "1", folder, 
                  P_discr_opt=P_discr_opt, len_P_1=len_P_1)

    # --- plot --- #
    plt.title("parameter Discretizations")
    plt.ylabel("depth")
    plt.xlabel("μ")
    plt.savefig(f"{folder}/paramDiscr.png")
    plt.show()


