from fenics import *

from .utils import *
from rb_semilinear.sl_problems import MySemilinearProblem

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_S(S:np.ndarray, x:np.ndarray, Mus:np.ndarray, convs:np.ndarray = None):
    """
    plots the Snapshot matrix 'S' over spatial coordinates 'x'. 

    Parameters
    ----------
    S : np.array
        Snapshot matrix, each column corresponds to a solution
    x : np.array
        spatial coordinates
    Mus : _type_
        _description_
    convs : _type_, optional
        _description_, by default None
    """
    if convs is None:
        [plt.plot(x, S[:, i], label=f"μ={Mus[i]:.3e}") for i in range(len(Mus))]
    else:
        [plt.plot(x, S[:, i], label=f"μ={Mus[i]:.3e}, conv={convs[i]}") \
            for i in range(len(Mus))]
    plt.title("Snapshots"); plt.xlabel("x"); plt.ylabel("u_h(μ)")
    #plt.legend()

def load_S(P_train:np.ndarray, N_h:int,S_folder:str, plot:bool = False):
    """
    Loads snapshot matrix for parameter μ in 'P_train' files in 'S_folder'.

    Parameters
    ----------
    P_train : np.array
        Parameter values used for snapshots generation
    S_folder : str
        directory containing files 'S.log', 'mapping.log', 'x.log'
    plot : bool, optional
        If True, snapshots are plotted. Default is False.

    Returns
    -------
    S : np.array
        Snapshot matrix (one snapshot per column)
    x : np.array
        Spatial coordinates
    Mus : np.array
        Parameter values from 'P_train' for which the hf-problem converged
    """
    # --- select parameter values/ids for which the hf-problem converged --- #
    df = pd.read_csv(f"{S_folder}/mapping.log", sep=r'\s+', engine='python')
    Mus_all = df['μ'].tolist()
    convergence_flags = df['converged'].tolist()
    ids, Mus, convflags = filter_Mus(P_train, Mus_all, convergence_flags, True)
    if len(P_train) < len(Mus):
        print(f"WARNING: hf-problem did not converge for all μ in 'P_train'")

    # --- load all snapshots and spatial coordinates --- #
    S = np.loadtxt(f"{S_folder}/S_{N_h}.csv", delimiter=',')
    x = np.loadtxt(f"{S_folder}/x_{N_h}.csv", delimiter=',')
    # --- filter snapshots --- #
    S = S[:, ids]

    if plot==True: plot_S(S,x,Mus); plt.show()

    return S, x, Mus


def comp_S(problem:MySemilinearProblem, 
           P_train:np.ndarray, 
           folder:str=None, plot:bool = False):

    S = []
    Mus_conv = []
    S_conv = []
    mapping = {}

    if folder is not None:
        import os
        os.makedirs(f"{folder}", exist_ok=True)

    if problem.solver.parameters['report']==False:
        print("computing snapshots:", end=" ")

    for idMu, mu in enumerate(P_train):
        SolInfo, NitInfo = problem.solve(mu)

        # --- save snapshot --- #
        S.append(problem.u.vector().get_local())    #auch not converged! 

        if folder is not None:
            w2file(f"{folder}/NitInfos.log",f"μ={mu}, N={problem.N_h}")
            w2file(f"{folder}/NitInfos.log",NitInfo)
    
        appendDict(mapping, SolInfo)

        if SolInfo["converged"] == 1:
            S_conv.append(problem.u.vector().get_local())
            Mus_conv.append(mu)
        else:
            print(f"WARNING: hf_problem did not converge for µ = {mu}!!")
        
        if problem.solver.parameters['report']==False:
            print("| ", end="" if idMu < len(P_train)-1 else "\n")

    S = np.array(S).T
    S_conv = np.array(S_conv).T
    x = np.reshape(problem.V_h.tabulate_dof_coordinates(), (problem.N_h+1,))

    if folder is not None:
        np.savetxt(f"{folder}/S_{problem.N_h}.csv",S, delimiter=',', fmt='%.12e')
        np.savetxt(f"{folder}/x_{problem.N_h}.csv",x, delimiter=',', fmt='%.12e')
        np.savetxt(f"{folder}/P_train.csv",P_train, delimiter=',', fmt='%.12e')
        from rb_semilinear.my_plots import myPlot, convMap
        df = pd.DataFrame(mapping, index=None); df['t'] = df.axes[0]
        df = df.astype({'nit': 'int32', 'converged': 'int32', 'N': 'int32'})
        if not isinstance(problem.solver, PETScSNESSolver):
            df = df.astype({'λ-FAIL':'int32'})
        w2file("%s/%s"%(folder, "mapping.log"), 
                df.to_string(float_format=lambda x: "{:.10e}".format(x)), "w")
        myPlot(folder, [problem.N_h], P_train[0:-1:5], 
                        False, plotfilename=f"{problem.N_h}.png")
        myPlot(folder, [problem.N_h], P_train[0:-1:5], 
                        True, plotfilename=f"{problem.N_h}_conv.png")
        convMap(folder)

    if plot==True:
        plot_S(S_conv, x, Mus_conv)
        plt.show()

    return S_conv, x, Mus_conv


def get_S(P_train:np.ndarray,
          hf_problem:MySemilinearProblem, 
          S_folder:str,
          plot:bool = False):

    N_h = hf_problem.N_h
    import os
    if not os.path.exists(S_folder) \
            or not os.path.exists(f"{S_folder}/S_{hf_problem.N_h}.csv"):
        return comp_S(hf_problem, P_train, S_folder, plot)

    Ps = np.loadtxt(f"{S_folder}/P_train.csv",delimiter=',', encoding='utf-8')
    for mu in P_train:
        if not float_is_in(mu, Ps):
            return comp_S(hf_problem, P_train, S_folder, plot)

    return load_S(P_train,hf_problem.N_h,S_folder, plot=plot)