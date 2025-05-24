from fenics import * 
from . import * 
from typing import Literal

class MyNewtonSolver():
    """
    Factory class to create a NewtonSolver instance based on solver_type.

    Depending on the value of `solver_type`, 
    an instance of one of the following Newton solver classes is returned:
    - "simplDamp" : MyNewton_simplDamp
    - "adaptDamp" : MyNewton_adaptDamp (with fixed tau=0.5)
    - "ord"       : MyNewton_ord
    - "nleqerr"   : MyNewton_nleqerr

    Parameters
    ----------
    tol : float
        Absolute tolerance for Newton solver
    maxit : int
        Maximum number of Newton iterations
    report : bool
        If True, the solver prints informations while solving
    solver_type : str
    
    Returns
    -------
        Instance of a Newton solver class corresponding to 'solver_type'
    """
    def __new__(cls, tol, maxit, report, 
                solver_type:Literal["simplDamp", "adaptDamp", "ord", "nleqerr"]):
        """
        Factory class to create a NewtonSolver instance based on solver_type.

        Depending on the value of `solver_type`, 
        an instance of one of the following Newton solver classes is returned:
        - "simplDamp" : MyNewton_simplDamp
        - "adaptDamp" : MyNewton_adaptDamp (with fixed tau=0.5)
        - "ord"       : MyNewton_ord
        - "nleqerr"   : MyNewton_nleqerr

        Parameters
        ----------
        tol : float
            Absolute tolerance for Newton solver
        maxit : int
            Maximum number of Newton iterations
        report : bool
            If True, the solver prints informations while solving
        solver_type : str
    
        Returns
        -------
            Instance of a Newton solver class corresponding to 'solver_type'
        """

        if solver_type == "simplDamp":
            return MyNewton_simplDamp(tol, maxit, report)
        elif solver_type == "adaptDamp":
            return MyNewton_adaptDamp(tol, maxit, report, tau=0.5)
        elif solver_type == "ord":
            return MyNewton_ord(tol, maxit, report)
        elif solver_type == "nleqerr":
            return MyNewton_nleqerr(tol, maxit, report)
        else:
            raise Exception(f"Newton solver type '{solver_type}' not implemented!")
 