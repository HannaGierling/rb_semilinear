from fenics import *
from rb_semilinear.utils import appendDict, str2pathstr, w2file

from rb_semilinear.sl_problems import Fisher_mms, Fisher, SemilinearPoisson
from rb_semilinear.nl_solver import MyNewtonSolver
from rb_semilinear.parameter import get_P 
from rb_semilinear.hgreedy import greedy_htype, find_Bl
from rb_semilinear.rb_nl_problem import MyRedNonlinearProblem 
from rb_semilinear.my_plots import plot_mu_trained, convMap, myPlot, errRedSolsPlot

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

#------------------#
### rb: settings ###       
#------------------#

# --- settings for parameter training space --- #
P_train_range   = [2.e-5,1]       # parameter range

# --- settings for constr. of RB --- #
RB_tol          = 10/N_h**2
RB_N_max        = 15 
P_1_discr_opt   = "log"
P_1_ns          = 50
P_Bl_discr_opt  = "log"

# --- settings for solver --- #
rb_Rtol         = 10/N_h**2       # tolerance for residual in NewtonSolver
rb_maxit        = 100             # max. number of iterations for NewtonSolver
rb_initGuess_opt = "u_h(amu)"#hf_initGuess_opt # initial Guess strategy : 
                                    # "u_h(amu)","P","0","0.5" or "LP"

# --- settings for parameter test space --- #
P_test_range    = [2.1e-5, 0.89] # parameter range
P_test_ns       = 50              # number of parameter samples
P_test_random   = False           # random parameter selction
P_test_opt      = "log"           # discr. strategy for param.space: 
                                  #    "log","decade_log","lin","decade" 

# what to do:
only_hf         = False
only_rbtesting  = False
################################################################################
################### High fidelity problem ######################################
################################################################################

folder = str2pathstr(f"{os.path.dirname(__file__)}/"+\
         f"test_rbm/{test_problem}/Nh_{N_h}/{hf_initGuess_opt}_{solver_opt}/"+\
         f"P1_{P_1_discr_opt}{P_1_ns}")


# --- define high-fidelity Problem --- #
solver     = MyNewtonSolver(hf_Rtol, maxit = hf_maxit, report = False,
                               solver_type=solver_opt)
hf_problem = testproblems[test_problem](N_h=N_h, solver=solver, 
                                        initGuess_strategy=hf_initGuess_opt)

################################################################################
################### Reduced basis problem ######################################
################################################################################
if only_hf == False:

# --- define folders --- #
    #vtkfile_sol_N = File('%s/vtk/sol.pvd' %solving_folder)
    htype_folder    =   f"{folder}/htype_Greedy"
    redMod_folder   =   f"{htype_folder}/"+\
                        f"P_1_{P_1_discr_opt}{P_1_ns}/N{RB_N_max}_"+\
                        str2pathstr(f"eps{RB_tol:.1e}")
    stuff_folder    =   f"{redMod_folder}/stuff"
    solving_folder  =   f"{redMod_folder}/"+\
                        f"{str2pathstr(rb_initGuess_opt)}_{P_test_opt}"+\
                         f"{P_test_ns}{'random' if P_test_random==True else ''}"

# --- (compute) reduced basis --- #
    import os
    if os.path.exists(stuff_folder):
        only_rbtesting = True
    if not only_rbtesting:
        P_1_train       = get_P(P_train_range, P_1_discr_opt, P_1_ns)
        greedy_htype(hf_problem, P_1_train, RB_tol, N_bar=RB_N_max,
                            folder=stuff_folder, 
                            P_discr_opt=P_Bl_discr_opt) 

# --- define test parameter values --- #
    if False:#P_test_opt =="P_train":
        P_test = P_conv
    else:
        P_test = get_P(P_test_range, P_test_opt, P_test_ns, P_test_random)
    
# --- test reduced problem for µ in 'P_test' --- #
    mapping = {}
    S = np.zeros((N_h+1, len(P_test)))
    for idD, D in enumerate(P_test[25:35]):
        #plt.figure(figsize=(10, 16))
        #--- find corresponging V --- #
        opt         = "Val" if P_Bl_discr_opt == "lin" else "Exp"
        Bl          = find_Bl(D, stuff_folder, opt)
        V           = np.loadtxt(f"{stuff_folder}/{Bl}/RB_Greedy.csv", delimiter=',')

        # --- define corresponidng reduced problem --- #
        rb_solver = MyNewtonSolver(rb_Rtol, rb_maxit, True, "nleqerr")
        rb_problem  = MyRedNonlinearProblem(hf_problem, V, rb_solver, rb_initGuess_opt)
        rb_problem.proj_norm = "L2-norm"

        # --- compute u_h for error-analysis --- #
        if False:#P_test_opt == "P_train":
            u_h = Function(hf_problem.V_h)
            u_h.vector().set_local(S[:,idD])
        else:
            SolInfo, NitInfo = hf_problem.solve(D)#, hf_initGuess_opt)
            u_h = Function(hf_problem.V_h)
            u_h.assign(hf_problem.u)
            if SolInfo["converged"] != 1:
                print(f"WARNING: hf_problem not conveged for µ={D}")
                u_h = None
            if u_h is not None : plot(u_h, label="u_h") 
        
        # set corresponding initial guess
        if rb_initGuess_opt == "u_h(amu)":
            from rb_semilinear.hgreedy import _get_amu
            rb_problem.initGuess_strategy = None

            amu = _get_amu(Bl, stuff_folder)
            hf_problem.solve(amu)

            V_h = hf_problem.V_h
            M = assemble(inner(TrialFunction(V_h),TestFunction(V_h))*dx).array()
            rb_problem.u_rbc.set_local(V.T @ M @ hf_problem.u.vector().get_local())
            plot(rb_problem.u_N(), label="u_rbc_init")


        # --- solve rb-problem for 'mu' --- #
        SolInfo, NitInfo = rb_problem.solve(D)

        #--- error analysis ---#
        SolInfo.update(rb_problem.errornorms(u_h))

        # --- log infos of Newton iterations --- #
        w2file(f"{solving_folder}/NitInfos.log", f"µ={D}, N={N_h}", mode="w" if idD==0 else "a")
        w2file(f"{solving_folder}/NitInfos.log", NitInfo)

        # --- collect reduced solutions in hf-functionspace 
        #     and solution information --- #
        appendDict(mapping, SolInfo)
        S[:, idD] = rb_problem.u_N().vector().get_local()
        #vtkfile_sol_N << (rb_problem.u_N())

        # --- plot reduced solution (and hf-solution) --- #
        plot(rb_problem.u_N(), label=f"µ={D:.3e}, conv={SolInfo['converged']}")
        plt.title(f"errP={SolInfo['||u_h-P_N(u_h)||_L2']:.2e}, err={SolInfo['||u_h-u_N||_L2']:.2e}")
        plt.legend(); plt.show()

    plt.title("Reduced Solutions"); plt.xlabel("x"); plt.ylabel(r"$u_N(\mu)$")
    plt.legend(); plt.show()

    # --- save reduced solutions in hf-function space and spatial coordinates --- #
    np.savetxt(f"{solving_folder}/S_{RB_N_max}.csv", S, delimiter=',', fmt='%.12e')
    np.savetxt(f"{solving_folder}/x_{RB_N_max}.csv", hf_problem.V_h.tabulate_dof_coordinates(), 
               delimiter=',', fmt='%.12e')

    # --- logging/plotting --- #
    df = pd.DataFrame(mapping, index=None); df['t'] = df.axes[0]
    try:
        df = df.astype({'nit':'int32', 'converged':'int32', 'N':'int32', 'λ-FAIL':'int32'})
    except:
        df = df.astype({'nit':'int32', 'converged':'int32', 'N':'int32'})
    df['μ'] = df['μ'].map(lambda x: f"{x:.12e}")
    w2file(f"{solving_folder}/mapping.log", df.to_string(), "w")
    print(df.to_string())


    # --- plot errors of reduced solutions --- #
    csvfile =f"{solving_folder}/err_{P_test_opt}{P_test_ns}.csv"
    errRedSolsPlot(solving_folder, csvfile, 
                    P_test_opt, 
                    plot=True)

    # --- plot some converged solutions --- #
    myPlot(solving_folder, Ns_2plot=[RB_N_max], Mus_2plot=P_test,#[0:-1:5], 
            converged_only=True, plotfilename=f"{RB_N_max}.png", plot = True)

    # --- plot convergence mapping --- #
    convMap(solving_folder, plot = True)

