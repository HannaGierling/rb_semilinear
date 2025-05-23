from fenics import * 
           
class MyNewtonBase(NewtonSolver):
    """
    Base class for configuring a Newton solver

    This class wraps the FEniCS 'NEwtonSolver' for easily configure
    solver parameters 'absolute tolerance', 'maximum_iterations' 
    and 'report'.
    For solving the linear equation systems an LU-solver is used,
    'error_on_nonconvergence' is set to False and 'relative_tolerance' to 1.e-12.

    Parameters
    ----------
    tol : float
        Absolute tolerance for Newton solver
    maxit : int
        Maximum number of Newton iterations
    report : bool
        If True, the solver prints informations while solving
    lam_min : float
        Minimal damping factor used in case of damped Newton solvers

    Attributes
    ----------
    tol : float
        absolute convergence tolerance.
    """
    def __init__(self, tol:float, maxit:int, report:bool, 
                 lam_min=1e-8):
        NewtonSolver.__init__(self)
        self.tol = tol
        self.lam_min = lam_min
        prm=self.parameters
        prm['maximum_iterations'] = maxit
        prm['linear_solver'] = "lu"
        prm['error_on_nonconvergence'] = False
        prm['absolute_tolerance'] = tol 
        prm['relative_tolerance'] = 1.e-12
        prm['report'] = report 


