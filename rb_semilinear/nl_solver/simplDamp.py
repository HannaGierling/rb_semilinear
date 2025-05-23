from fenics import * 
from .newton_base import MyNewtonBase
from .utils import _appendDict

class MyNewton_simplDamp(MyNewtonBase):
    """
    Newton solver with simple damping strategy for NonlinearProblem.

    This class implements a damped Newton method where the update_solution step
    `u <- u - λ * du` uses a backtracking strategy to ensure sufficient
    decrease in the Newton correction.

    See algorithm 1.3.2 in 
        Numerik II (Einführung in die Numerische Analysis) 
        by Stefan Funken, Dirk Lebiedz, Karsten Urban
    for details.

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
    lam_fail : bool
        Indicates whether damping failed 
    """

    def __init__(self, tol, maxit=100, report=True, lam_min = 1e-8):
        MyNewtonBase.__init__(self, tol, maxit, report, lam_min)
        self.solInfo = {}
        self.NitInfos = {}

    def update_solution(self, x, dx, relaxation_parameter, nonlinear_problem,
                        iteration):

        # --- empty NitInfos and solInfo dictionaries --- #
        if iteration == 0:
            self.NitInfos = {}
            self.solInfo = {}
            self.lam_fail = False

            # --- logging --- #
            dict = {'nit'               :iteration,
                    '||F(u)||_2'        :self.parameters['absolute_tolerance']+1,
                    '||du||_2'          :norm(dx, "l2"), 
                    'atol'              :self.parameters['absolute_tolerance'],
                    'λ'                 :0, 
                    'λ-FAIL'            :int(self.lam_fail)}
            self.solInfo.update(dict); del self.solInfo["λ"]
            _appendDict(self.NitInfos, dict)
        
        # --- check whether damping failed --- #
        # TODO: is there an option to terminate NewtonSolver based on lam_fail=True??
        if self.lam_fail == True:
            return
        
        # --- logging --- #
        self.NitInfos["||du||_2"][-1]=norm(dx, "l2")

        # --- damping --- #
        lam = 1

        # --- get Jacobian matrix --- #
        J_Fu = PETScMatrix()
        nonlinear_problem.J(J_Fu, x)

        # --- simple damping strategy --- #
        Fu_trial = PETScVector()
        u_trial = x.copy(); du_trial = x.copy()
        while lam > self.lam_min:

            u_trial.zero(); u_trial.axpy(1.0, x); u_trial.axpy(-lam, dx)

            # --- get resiudal vector for u_trial --- #
            nonlinear_problem.F(Fu_trial,u_trial)

            # --- solve linear system --- #
            solve(J_Fu, du_trial, Fu_trial)

            # --- monotonidity test --- #
            if norm(du_trial) <= (1-lam/2)*norm(dx):
                break
            lam = lam/2
        if lam <  self.lam_min: 
            self.lam_fail = True
        

        # --- logging --- #
        dict = {    'nit'               :iteration+1,
                    '||F(u)||_2'        :norm(Fu_trial, "l2"), 
                    '||du||_2'          :0,#wird am beim aufruf von update solution geupdated#norm(du_trial.vector(), "l2"),
                    'atol'              :self.parameters['absolute_tolerance'],
                    'λ'                 :lam, 
                    'λ-FAIL'            :int(self.lam_fail)}
        self.solInfo.update(dict); del self.solInfo["λ"]
        _appendDict(self.NitInfos, dict)

        # --- solution update --- #
        x.axpy(-lam, dx)

