from fenics import * 
from .newton_base import MyNewtonBase
from .utils import _appendDict

class MyNewton_adaptDamp(MyNewtonBase):
    """
    Adaptively damped Newton solver for semilinear PDEs

    The damping parameter is updated in each iteration using:
        λ = min(sqrt(2 * tau / ||du||), 1.0)
    
    See 
        AN ADAPTIVE NEWTON-METHOD BASED ON A DYNAMICAL SYSTEMS APPROACH
        by MARIO AMREIN AND THOMAS P. WIHLER
    for details

    Parameters
    ----------
    tol : float
        Absolute tolerance for Newton solver
    maxit : int
        Maximum number of Newton iterations
    report : bool
        If True, the solver prints informations while solving
    tau : float
        tolerance used in damping

    Attributes
    ----------
    tau : float
        tolerance used in damping
    solInfo : dict
        Dictionary with information of finial iteration
    NitInfos : dict
        Dictionary with information at each iteration
    lam_fail : bool
        Indicates whether damping failed 
    """
    def __init__(self, tol, maxit=100, report=True, tau=0.5):
        MyNewtonBase.__init__(self, tol, maxit, report)
        self.tau = tau

    def update_solution(self, x, dx, relaxation_parameter, nonlinear_problem,
                        iteration):

        # --- empty NitInfos and solInfo dictionaries --- #
        if iteration == 0:
            self.NitInfos = {}
            self.solInfo = {}
            self.lam_fail = False

            # --- logging --- #
            dict = {'nit'               :iteration,
                    '||F(u)||_2'        :nonlinear_problem.resNorm,
                    '||du||_2'          :norm(dx, "l2"), 
                    'atol'              :self.parameters['absolute_tolerance'],
                    'λ'                 :0, 
                    'λ-FAIL'            :self.lam_fail}
            self.solInfo.update(dict); del self.solInfo["λ"]
            _appendDict(self.NitInfos, dict)       

        #if self.lam_fail == True:
            #return
        
        # --- logging --- #
        self.NitInfos["||du||_2"][-1]=norm(dx, "l2")

        # --- damping --- #
        lam = min(sqrt(2.0*self.tau/norm(dx, norm_type="l2")), 1.0)
        lam = max(lam, 1.e-5)

        # --- solution update --- #
        x.axpy(-lam, dx)

        # --- compute residual norm for logging --- #
        du_trial = x.copy()
        Fu_trial = PETScVector()
        nonlinear_problem.F(Fu_trial,x)
        J_Fu = PETScMatrix()
        nonlinear_problem.J(J_Fu, x)
        solve(J_Fu, du_trial, Fu_trial)

        #if norm(du_trial, norm_type="L2") > (1-lam/2)*norm(dx, norm_type="l2"):
            #self.lam_fail = True
        #if lam < 1.e-5:
            #self.lam_fail = True

        # --- logging --- #
        dict = {    'nit'               :iteration+1,
                    '||F(u)||_2'        :norm(Fu_trial, "l2"), 
                    '||du||_2'          :0,#wird am beim aufruf von update solution geupdated#norm(du_trial.vector(), "l2"),
                    'atol'              :self.parameters['absolute_tolerance'],
                    'λ'                 :lam, 
                    'λ-FAIL'            :self.lam_fail}
        self.solInfo.update(dict); del self.solInfo["λ"]
        _appendDict(self.NitInfos, dict)


