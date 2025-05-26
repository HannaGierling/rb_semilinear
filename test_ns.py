from fenics import *
import os
import pandas as pd
import numpy as np

from rb_semilinear.utils import appendDict, str2pathstr, w2file
from rb_semilinear.nl_solver import MyNewtonSolver
from rb_semilinear.my_plots import convMap, myPlot, errPlot
from rb_semilinear.parameter import get_P
from rb_semilinear.sl_problems import Fisher, Fisher_mms, SemilinearPoisson

testproblems = {"mmsF":Fisher_mms, "F":Fisher, "slP":SemilinearPoisson}

#----- SETTINGS ---------------------------------------------------------------#

# --- selcect test problem --- #       
test_problem        =  "mmsF"              # "F","mmsF" or "slP"

# --- settings for solver --- #       
solver_type         = "nleqerr"             # damping strategy : "ord", "nleqerr", "adaptDamp", "simplDamp"

# ! for solver validation use N_hs = [50, 100, 200, 500, 1000, 2000, 3000, 5000, 6000, 7000]
#   in combination with test_problem = "mmsF"

N_hs                = [1000]                # number of intervals in spat.discr.

maxit               = 100                   # max. number of iterations for NewtonSolver
atol                = lambda N_h: 1/N_h**2  # tolerance for residual in NewtonSolver
initGuess_strategy  = "0.5"                 # initial Guess strategy : "P","0","0.5" or "LP"


# --- parameter --- #
P_range = [1.e-5,1]                         # parameter range

# --- Select parameter values μ according to Section 5 of the thesis --- #
P_discr_strategy = "thesis"
Ps = get_P(P_range, P_discr_strategy)
#------------------------------------------------------------------------------#

# --- folder path --- #
problemfolder = str2pathstr(f"{test_problem}/{P_discr_strategy}") 
solverfolder = str2pathstr(f"{initGuess_strategy}/{solver_type}")
folder       = f"{os.path.dirname(__file__)}/test_MyNewtonSolver/"+\
                f"{problemfolder}/{solverfolder}"

# --- for logging --- #
mapping ={}
#vtkfile = File(f"{folder}/vtk/sol.pvd")

for idN, N_h in enumerate(N_hs):

    S = np.zeros((N_h+1, len(Ps)))
    # --- Newtonsolver --- #
    solver = MyNewtonSolver(tol=atol(N_h), maxit=maxit, report=True, 
                            solver_type=solver_type)
    # --- high-fidelity Problem --- #
    problem = testproblems[test_problem](N_h=N_h, solver=solver,
                                         initGuess_strategy=initGuess_strategy)

    for idMU, mu in enumerate(Ps):

        # --- solve problem for mu --- #
        SolInfo, NitInfo = problem.solve(mu)

        # --- collect solutions and solution information --- #
        #vtkfile << (problem.u)
        S[:, idMU] = problem.u.vector().get_local()
        appendDict(mapping, SolInfo)

        # --- log infos of Newton iterations --- #
        w2file(f"{folder}/NitInfos.log", f"μ={mu}, N={N_h}", mode="w" if idMU==0 else "a")
        w2file(f"{folder}/NitInfos.log", pd.DataFrame(NitInfo).to_string())


    # --- save solutions and spatial coordinates --- #
    np.savetxt(f"{folder}/S_{N_h}.csv", S, delimiter=',', fmt='%.12e')
    np.savetxt(f"{folder}/x_{N_h}.csv", problem.V_h.tabulate_dof_coordinates(), 
               delimiter=',', fmt='%.12e')

# --- save convergence infos and parameter values --- #
np.savetxt(f"{folder}/P.csv", Ps, delimiter=',', fmt='%.12e')
df = pd.DataFrame(mapping); df['t'] = df.axes[0]
try:
    df = df.astype({'nit':'int32', 'converged':'int32', 'N':'int32', 
                    'λ-FAIL':'int32'})
except:
    df = df.astype({'nit':'int32', 'converged':'int32', 'N':'int32'})
df['μ'] = df['μ'].map(lambda x: f"{x:.12e}")
w2file(f"{folder}/mapping.log", df.to_string(), "w")

# --- plot convergence mapping --- #
convMap(folder, plot=True)

# --- error Plot --- #
if test_problem=="mmsF" and len(N_hs)>5:
    errPlot(folder, Ps[-2], plot = True)

# --- plot converged solutions --- #
for N in N_hs:
    myPlot(folder, Ns_2plot=[N_h], Mus_2plot=Ps, converged_only=True , 
           plotfilename=f"{N_h}.png", plot=True)

