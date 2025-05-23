from fenics import *

class MyNewton_nleqerr(PETScSNESSolver):
    """
    Newton solver using PETSc's SNES framework with NLEQ-ERR line search.

    This class wraps PETSc's `SNES` nonlinear solver and configures it to use
    for easily configuring solver parameters 'absolute tolerance', 
    'maximum_iterations' and 'report'.

    It uses PETSc's `newtonls` method with a `nleqerr` line search strategy.

    PETSc Documentation Reference:
        https://petsc.org/main/manualpages/SNES/SNESSetFromOptions/

    Parameters
    ----------
    tol : float
        Absolute tolerance for Newton solver
    maxit : int
        Maximum number of Newton iterations
    report : bool
        If True, the solver prints informations while solving

    Attributes
    ----------
    solInfo : dict
        Dictionary with information of finial iteration
    NitInfos : dict
        Dictionary with information at each iteration
    """
    def __init__(self, tol, maxit=100, report = True):
        PETScSNESSolver.__init__(self)
        self.parameters["method"] = "newtonls"
        self.parameters["line_search"] = "nleqerr"
        self.parameters["maximum_iterations"] = maxit 
        self.parameters["absolute_tolerance"] = tol
        self.parameters["relative_tolerance"] = 1.e-12
        self.parameters["solution_tolerance"] = tol
        self.parameters["linear_solver"] = "lu"#"cg"
        #self.parameters["preconditioner"] = "amg"#"ilu"
        self.parameters["error_on_nonconvergence"] = False
        self.parameters["report"] = report 
        
        #PETScOptions.set("snes_linesearch_monitor", "")
        if report == True:
            PETScOptions.set("snes_monitor", "")
            PETScOptions.set("snes_converged_reason", "")
        #else:
            #PETScOptions.set("snes_converged_reason", "")
        self.set_from_options()        

        self.tol = tol
        self.NitInfos = {"nit":[], "||F(u)||_2":[],"atol":[]}
        self.solInfo = {"nit":0, "||F(u)||_2":100,"atol":tol,"converged":False}
        
        def my_snes_monitor(snes, its, norm):
            if its == 0:
                self.NitInfos = {"nit":[], "||F(u)||_2":[], "atol":[]}
                self.solInfo.update({"nit":0, "||F(u)||_2":norm,"converged":False})
            self.NitInfos["nit"].append(its)
            self.NitInfos["||F(u)||_2"].append(norm)
            self.NitInfos["atol"].append(self.solInfo["atol"])
            self.solInfo.update({"nit":its, "||F(u)||_2":norm,
                                 "converged":snes.is_converged})
        self.snes().setMonitor(my_snes_monitor)

