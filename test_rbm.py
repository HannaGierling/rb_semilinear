from fenics import *
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from rb_semilinear.sl_problems import Fisher_mms, Fisher, SemilinearPoisson
from rb_semilinear.nl_solver import MyNewtonSolver
from rb_semilinear.parameter import get_P
from rb_semilinear.snapshots import get_S 
from rb_semilinear.pod import comp_POD
from rb_semilinear.greedy import comp_greedyRB
from rb_semilinear.rb_nl_problem import MyRedNonlinearProblem

from rb_semilinear.utils import appendDict, str2pathstr, w2file
from rb_semilinear.my_plots import plot_mu_trained, convMap, myPlot, errRedSolsPlot

testproblems = {"mmsF":Fisher_mms, "F":Fisher, "slP":SemilinearPoisson}

#----- SETTINGS ---------------------------------------------------------------#

# --- select test problem --- #
test_problem    = "F"             # "F","mmsF" or "slP"

#------------------#
### hf: settings ###       
#------------------#

# --- settings for solver --- #       
N_h             = 1000            # number of intervals in spat.discr.
solver_opt      = "nleqerr"       # damping strategy : "ord", "nleqerr", "adaptDamp", "simplDamp"
hf_Rtol         = 1/N_h**2        # tolerance for residual in NewtonSolver
hf_maxit        = 100             # max. number of iterations for NewtonSolver
hf_initGuess_opt = "P"            # initial Guess strategy : "P","0","0.5" or "LP"
homotopie       = False 

# --- settings for parameter training space --- #
P_train_range   = [2.e-5,1]       # parameter range
P_train_ns      = 50              # number of parameter samples
P_train_random  = False           # random parameter selction
P_train_opt     = "log"           # discr. strategy for param.space: 
                                  # log","decade_log","lin","decade","lin_n_log"

# --- get 'P_train' --- #
P_train  = get_P(P_train_range, P_train_opt, P_train_ns, P_train_random)

#------------------#
### rb: settings ###       
#------------------#

# --- settings for constr. of RB --- #
RB_opt          = "Greedy"        # RB generation : "POD_L2","POD_2","Greedy","htype_Greedy"
RB_tol          = 10/N_h**2       # tolerance for either POD or Greedy
RB_N_max        = P_train_ns      # max. number of RB-functions

# --- settings for solver --- #
rb_Rtol         = 10/N_h**2       # tolerance for residual in NewtonSolver
rb_maxit        = 100             # max. number of iterations for NewtonSolver
rb_initGuess_opt = hf_initGuess_opt # initial Guess strategy : 
                                    #   "u_h", "P","0","0.5" or "LP"

# --- settings for parameter test space --- #
P_test_range    = [2.1e-5, 0.89]  # parameter range
P_test_ns       = 50              # number of parameter samples
P_test_random   = False           # random parameter selction
P_test_opt      = "log"           # discr. strategy for param.space: 
                                  #  "log","decade_log","lin","decade","P_train"

# what to do:
only_hf = False
################################################################################
################### High fidelity problem ######################################
################################################################################

folder = str2pathstr(f"{os.path.dirname(__file__)}/"+\
         f"test_rbm/{test_problem}/Nh_{N_h}/{hf_initGuess_opt}_{solver_opt}/"+\
         f"Ptrain_{P_train_opt}{P_train_ns}\
              {'random' if P_train_random==True else ''}")

# --- define high-fidelity Problem --- #
solver     = MyNewtonSolver(tol=hf_Rtol, maxit=hf_maxit, report=True, 
                            solver_type=solver_opt)
hf_problem = testproblems[test_problem](N_h=N_h, solver=solver, 
                                        initGuess_strategy=hf_initGuess_opt)

# --- get snapshot matrix 'S' --- #
S_folder   = f"{folder}/snapshots"
S, x, P_conv    = get_S(P_train, hf_problem, S_folder, plot=True)
                              
################################################################################
################### Reduced basis method #######################################
################################################################################
if only_hf == False:
# --- define folders --- #
    rb_folder = f"{folder}/{RB_opt}/"+\
                str2pathstr(f"Nmax_{RB_N_max}_tol_{RB_tol:.2e}")
    stuff_folder    =  f"{rb_folder}/stuff"

# --- compute reduced basis --- #
    if RB_opt == "Greedy":
        m = inner(TrialFunction(hf_problem.V_h),TestFunction(hf_problem.V_h))*dx
        M = assemble(m).array()
        RB = comp_greedyRB(S, M, P_conv,
                            N_max=RB_N_max, greedy_tol=RB_tol,
                            folder=stuff_folder, ret_Pused=False )
        plot_mu_trained(stuff_folder,P_train_opt, plot=True)

    elif RB_opt == "POD_2":
        RB = comp_POD(S, tol=RB_tol, N_max=RB_N_max, 
                        folder=stuff_folder, plot=True, ret_sigma=False)

    elif RB_opt == "POD_L2":
        m = inner(TrialFunction(hf_problem.V_h),TestFunction(hf_problem.V_h))*dx
        M = assemble(m).array()
        RB = comp_POD(S, tol = RB_tol, N_max = RB_N_max, M=M,
                   folder = stuff_folder, plot = True, ret_sigma=False)
    else:
        raise Exception(f"Option '{RB_opt}' for RB generation not implemented!")

    solving_folder = f"{rb_folder}/{str2pathstr(rb_initGuess_opt)}"+\
                      f"_P_test_{P_test_opt}"+\
                      f"{P_test_ns}{'random' if P_test_random==True else ''}"

# --- define reduced problem --- #
    rb_solver = MyNewtonSolver(rb_Rtol, rb_maxit, True, "nleqerr")
    rb_problem = MyRedNonlinearProblem(hf_problem, RB, rb_solver, rb_initGuess_opt)
    rb_problem.proj_norm = "2-norm" if RB_opt == "POD_2" else "L2-norm"

# --- define test parameter values --- #
    if P_test_opt =="P_train":
        P_test = P_conv
    else:
        P_test = get_P(P_test_range, P_test_opt, P_test_ns, P_test_random)

# --- test reduced problem for µ in 'P_test' --- #
    mapping = {}
    #vtkfile_sol_N = File('%s/vtk/sol.pvd' %rb_folder)
    S_N = np.zeros((N_h+1, len(P_test)))

    for idMU, mu in enumerate(P_test):

        # ---  compute high-fidelity solution 'u_h' for error-analysis --- #
        if P_test_opt == "P_train":
            u_h = Function(hf_problem.V_h)
            u_h.vector().set_local(S[:,idMU])
        else:
            print(f"\nsolve hf-problem for µ={mu}")
            SolInfo, NitInfo = hf_problem.solve(mu)
            u_h = hf_problem.u
            if SolInfo["converged"] != 1:
                print(f"WARNING: hf_problem not conveged for µ={mu}")
                u_h = None
        
        if rb_initGuess_opt == "u_h":
            if u_h == None: continue
            rb_problem.initGuess_strategy = None
            V_h = hf_problem.V_h
            M = assemble(inner(TrialFunction(V_h),TestFunction(V_h))*dx).array()
            rb_problem.u_rbc.set_local(RB.T @ M @ u_h.vector().get_local())
            plot(rb_problem.u_N(), label="u_rbc_init")

        # --- solve rb-problem for 'mu' --- #
        print(f"\nsolve rb-problem for µ={mu}")
        SolInfo, NitInfo = rb_problem.solve(mu)

        # --- error analysis --- #
        SolInfo.update(rb_problem.errornorms(u_h))#, proj_norm="L2-norm"))

        # --- log infos of Newton iterations --- #
        w2file(f"{solving_folder}/NitInfos.log", f"µ={mu}, N={N_h}", 
                    mode="w" if idMU==0 else "a")
        w2file(f"{solving_folder}/NitInfos.log", NitInfo)

        # --- collect reduced solutions in hf-functionspace 
        #     and solution information --- #
        S_N[:, idMU] = rb_problem.u_N().vector().get_local()
        appendDict(mapping, SolInfo)

        # --- plot reduced solution (and hf-solution) --- #
        plot(rb_problem.u_N(), label=f"µ={mu:.3e}, conv={SolInfo['converged']}")
        if u_h is not None :plot(u_h, label="u_h") 
        plt.legend(); plt.ylim([-0.8,0.8])
        #plt.show()

    plt.title("Reduced Solutions"); plt.xlabel("x"); plt.ylabel(r"$u_N(\mu)$")
    plt.legend(); plt.show()


    # --- save reduced solutions in hf-function space and spatial coordinates --- #
    np.savetxt(f"{solving_folder}/S_{RB.shape[1]}.csv", 
                    S_N, delimiter=',',fmt='%.12e')
    np.savetxt(f"{solving_folder}/x_{RB.shape[1]}.csv", 
                    hf_problem.V_h.tabulate_dof_coordinates(), 
                    delimiter=',', fmt='%.12e')

    # --- save convergence infos --- #
    df = pd.DataFrame(mapping, index=None); df['t'] = df.axes[0]
    try:
        df = df.astype({'nit':'int32', 'converged':'int32', 'N':'int32', 
                        'λ-FAIL':'int32'})
    except:
        df = df.astype({'nit':'int32', 'converged':'int32', 'N':'int32'})
    df['μ'] = df['μ'].map(lambda x: f"{x:.12e}")
    w2file(f"{solving_folder}/mapping.log", df.to_string(), "w")
    print(df.to_string())


    # --- plot errors of reduced solutions --- #
    csvfile =f"{solving_folder}/err_{P_test_opt}{P_test_ns}.csv"
    errRedSolsPlot(solving_folder, csvfile,  
                    P_train_opt if P_test_opt == "P_train" else P_test_opt,
                    plot = True)

    # --- plot some converged solutions --- #
    myPlot(solving_folder, Ns_2plot=[RB.shape[1]], Mus_2plot=P_test,#[0:-1:5], 
            converged_only=True, plotfilename=f"{RB.shape[1]}.png", plot=True)

    # --- plot convergence mapping --- #
    convMap(solving_folder, plot = True)





