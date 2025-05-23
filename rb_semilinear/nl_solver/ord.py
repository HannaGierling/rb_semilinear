from fenics import * 
from .newton_base import MyNewtonBase
from .utils import _appendDict

class MyNewton_ord(MyNewtonBase):
    """
    ordinary Newton solver for NonlinearProblem

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

    def __init__(self, tol, maxit=100, report=True):
        MyNewtonBase.__init__(self, tol, maxit, report)

    def update_solution(self, x, dx, relaxation_parameter, nonlinear_problem,
                        iteration):

        # --- empty NitInfos and solInfo dictionaries --- #
        if iteration == 0:
            self.NitInfos = {}
            self.solInfo = {}
            # --- logging --- #
            dict = {    'nit'               :iteration,
                        '||F(u)||_2'        :nonlinear_problem.resNorm,
                        '||du||_2'          :norm(dx, "l2"), 
                        'atol'              :self.parameters['absolute_tolerance'],
                        'λ'                 :0, 
            }
            self.solInfo.update(dict); del self.solInfo["λ"]
            _appendDict(self.NitInfos, dict)

        # --- solution update --- #
        x.axpy(-1, dx)

        # --- logging --- #
        dict = {    'nit'               :iteration+1,
                    '||F(u)||_2'        :nonlinear_problem.resNorm,
                    '||du||_2'          :norm(dx, "l2"), 
                    'atol'              :self.parameters['absolute_tolerance'],
                    'λ'                 :1, 
        }
        self.solInfo.update(dict); del self.solInfo["λ"]
        _appendDict(self.NitInfos, dict)


