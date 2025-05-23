from fenics import * 
from typing import Literal, Any, Callable

from nl_solver import MyNewtonSolver 


class MySemilinearProblem(NonlinearProblem):
    """
    Base class for semilinear elliptic PDEs in one dimension

    Setup for general semilinear elliptic PDE:
        -div(mu * ∇u) - q(u) = f   in Ω = (0, 1),
        u = u_D                   on ∂Ω.
        
    The problem is discretized using linear Lagrange elemens.

    It is solved using 'solver' with 'initGuess_strategy'.

    Parameters
    ----------
    N_h : int
        The number of discretization intervalls

    q : function     
        The nonlinear reaction term

    f : Expression
        The force term

    u_D : Expression
          The Dirichlet Function

    solver : MyNewtonSolver
        A Newton Solver
    strategy : {"P", "0.5", "0", "LP", None} - specifies how to initialize 'self.u'. 
    
        Options:
        
        - "P":      Solves the Poisson equation (-div(D∇u) = f) 
                        and uses the result as initial guess.
        - "0.5":    Sets 'u = 0.5' (with boundary conditions).
        - "0":      Sets 'u = 0.0' (with boundary conditions).
        - "LP":     Solves a linearized version of the semilinear problem:
                        -div(D∇u) - u = f
                        and uses the result as initial guess.
        - None:     Uses the current value of 'self.u'

    
    Attributes
    ----------
    mu : float
        Parameter value

    mu_expr : Expression
        FEniCS expression wrapping the parameter value

    V_h : FunctionSpace
        Finite element space (P1)

    bcs : list[DirichletBC]
        List of DirichletBC 

    u : Function
        Unknown solution function

    solver : MyNewtonSolver
        solver used to solve this problem.

    initGuess_strategy : {"P", "0.5", "0", "LP", None}
        inital guess strategy used to solve this problem.
    """
    def __init__(self,  N_h:int, 
                        q:Callable[[Any], Any], 
                        f:Expression, 
                        u_D:Expression, 
                        solver:MyNewtonSolver,
                        initGuess_strategy:Literal["0","0.5","P","LP",None]):

        NonlinearProblem.__init__(self)
    
        self.mu     = 1.0
        self.mu_expr = Expression("mu", mu=self.mu, degree=0)
        self.q      = q
        self.f      = f

        self.N_h = N_h
        self.V_h = FunctionSpace(IntervalMesh(N_h, 0, 1), "P", 1)

        # --- Define weak problem --- #
        u = TrialFunction(self.V_h)
        v = TestFunction(self.V_h)
        self.nlwf = self.mu_expr * inner(grad(u), grad(v)) * dx\
                                - inner(q(u), v) * dx\
                                - inner(self.f, v) * dx

        # --- boundary conditions --- #
        bdry = lambda x, on_boundary: on_boundary
        self.bc0 = DirichletBC(self.V_h, Constant(0.0), bdry)
        self.bcs = [DirichletBC(self.V_h, u_D, bdry)]

        # --- unknown solution function --- #
        self.u = Function(self.V_h) 

        # --- residual and derivaitve --- #
        self._l = lambda u: action(self.nlwf, u)
        du = TrialFunction(self.V_h)
        self._a = lambda u: derivative(self._l(u), u, du)

        # --- residual norm (gets updated in self.F()) --- #
        self.resNorm = -1.0

        # --- nonlinear solver and initGuess_strategy --- # 
        self.solver = solver
        self.initGuess_strategy = initGuess_strategy

    def set_mu(self,mu:float):
        """
        Set parameter value of problem 'self.mu' to value 'mu'

        Parameters
        ----------
        mu : float 
            parameter value
        """
        self.mu = mu
        self.f.mu = mu 
        self.mu_expr.mu = mu 
    
    def set_initGuess(self, strategy:Literal["P", "0.5", "0", "LP", None]):
        """
        Set initial guess for the nonlinear solver based on a chosen strategy.

        This method modifies 'self.u' to provide an initial guess before 
        solving a nonlinear problem. 

        Parameters
        ----------
        strategy : {"P", "0.5", "0", "LP", None} - specifies how to initialize 'self.u'. 
        
            Options:
            - "P":      Solves the Poisson equation (-div(D∇u) = f) 
                         and uses the result as initial guess.
            - "0.5":    Sets 'u = 0.5' (with boundary conditions).
            - "0":      Sets 'u = 0.0' (with boundary conditions).
            - "LP":     Solves a linearized version of the semilinear problem:
                            -div(D∇u) - u = f
                            and uses the result as initial guess.
            - None:     Uses the current value of 'self.u' 
        """
        if strategy == "P":
            from l_problem import LinearProblem
            # --- Poisson weak formulation --- #
            u = TrialFunction(self.V_h); v = TestFunction(self.V_h)
            Pwf = self.mu_expr * inner(grad(u),grad(v)) * dx\
                              - inner(self.f, v) * dx
            # --- solve linear problem --- #
            l_problem = LinearProblem(Pwf, self.bcs)
            l_problem.solve(self.u.vector())
        elif strategy == "0.5":
            self.u.vector()[:] = 0.5 
            self.bcs[0].apply(self.u.vector())
        elif strategy == "0":
            self.u.vector()[:] = 0.0 
            self.bcs[0].apply(self.u.vector())
        elif strategy == None:
            self.u
        elif strategy =="LP":
            # --- linearized weak formulation --- #
            u = TrialFunction(self.V_h); v = TestFunction(self.V_h)
            lwf = self.mu_expr * inner(grad(u), grad(v)) * dx \
                              - inner(u,v) * dx \
                              - inner(self.f, v) * dx
            # --- solve linear problem --- #
            from l_problem import LinearProblem
            l_problem = LinearProblem(lwf, self.bcs)
            l_problem.solve(self.u.vector())
        else:
            raise Exception(f"initial guess strategy: {strategy} not implemented")
        
    def set_solver(self, solver):
        """
        Set Newton-type solver for solving the semilinear PDE

        Parameters
        ----------
        solver : MyNewton
        """
        self.solver = solver
    
    def _solve(self, mu:float):
        """ 
        General method to solve semilinear PDE for parameter value 'mu' with 
        with initial guess strategy 'self.initGuess_stratey' using 'self.solver'.

        Parameters
        ----------
        mu : float
            paramter value

        Returns
        -------
        solInfo: dict
            information about solution
        NitInfos : dict
            information about Newton iterations

        """
        self.set_mu(mu)
        self.set_initGuess(self.initGuess_strategy)
        nit, conv = self.solver.solve(self, self.u.vector())
        self.solver.solInfo.update({"converged":int(conv)})

        tmp = {"N":self.N_h, "μ":self.mu}; tmp.update(self.solver.solInfo)
        return tmp, self.solver.NitInfos
    
    def solve(self, mu:float):
        """ 
        Solve semilinear PDE for parameter value 'mu' 
        with initial guess strategy 'self.initGuess_stratey' using 'self.solver'.

        Parameters
        ----------
        mu : float
            paramter value

        Returns
        -------
        solInfo: dict
            information about solution
        NitInfos : dict
            information about Newton iterations

        Note
        ----
        The initial guess strategy is specified in 'self.initGuess_strategy'
        when instantiating a 'MySemilinearProblem' object.
        It can be changed by modifying 'self.initGuess_strategy'.
        """
        return self._solve(mu) 

#    def _l(self, u): 
        #"""
        #define residual of semilinear weak form at given state 'u'

        #Parameters
        #----------
        #u : Function
            #State at which the residual is evaluated.

        #Returns
        #-------
        #Form 
            #UFL form representing the residual of semilinear weak form
        #"""
        ##return action(self.nlwf, self.u)
        #return action(self.nlwf, u)
    
    #def _jacobian(self, u):  
        #"""
        #Define derivative of semilinear weak form at state 'u'

        #Parameters
        #----------
        #u : Function
            #State at which the derivative is evaluated.

        #Returns
        #-------
        #Form 
            #UFL form representing the Jacobian of semilinear weak form 
        #"""
        #du = TrialFunction(self.V_h)
        #return derivative(self._l(u), u, du)
    
    def F(self, b, x):
        """
        Assemble Residual vector of semilinear PDE at state 'x'.
        update 'self.resNorm' and apply Dirichlet boundary condtions 'self.bcs'

        Parameters
        ----------
        b : PETScVector 
            Vector object where Residual vector will be stored
        x : PETScVector
            Vector of degrees of freedom of the state at which the Residual 
            is evaluated
        """
        # --- assemble resiudal vector --- # 
        u = Function(self.V_h)
        u.vector().set_local(x.get_local())
        #assemble(self._l(u), tensor=b)
        assemble(self._l(u), tensor=b)
        # --- compute residuum --- #
        self.bc0.apply(b)
        self.resNorm = norm(b, "l2")
        # --- apply boundary conditions --- #
        [bc.apply(b,x) for bc in self.bcs]
    
    def J(self, A, x):
        """
        Assemble Jacobian matrix of semilinear PDE at state 'x'.
        and apply Dirichlet boundary conditions 'self.bcs'

        Parameters
        ----------
        A : PETScMatrix
            Matrix object where Jacobian matrix will be stored
        x : PETScVector 
            Vector of degrees of freedom of the state, at which the Jacobian 
            is evaluated
        """
        u = Function(self.V_h)
        u.vector().set_local(x.get_local())
        #assemble(self._jacobian(u), tensor = A)
        assemble(self._a(u), tensor=A)
        [bc.apply(A) for bc in self.bcs]

    
class Fisher(MySemilinearProblem):
    """
    elliptic Fisher's equation

    This class implements the steady-state version of the Fisher equation:
        -div(mu * ∇u) - u(1 - u) = 0
    
    It is solved using 'solver' with 'initGuess_strategy'.
    
    Parameters
    ----------
    N_h : int
        Number of mesh intervals for spatial discretization 
    solver : NewtonSolver or PETScSNESSolver
        Solver object used to solve the PDE
    initGuess_strategy : {"P", "0.5", "0", "LP", None} - specifies how to initialize 'self.u'. 
        Options:
        - "P":      Solves the Poisson equation (-div(D∇u) = f) 
                        and uses the result as initial guess.
        - "0.5":    Sets 'u = 0.5' (with boundary conditions).
        - "0":      Sets 'u = 0.0' (with boundary conditions).
        - "LP":     Solves a linearized version of the semilinear problem:
                        -div(D∇u) - u = f.
                        and uses the result as initial guess.
        - None:     Uses the current value of 'self.u' 
    """
    def __init__(self, N_h:int, solver:NewtonSolver | PETScSNESSolver, 
                 initGuess_strategy:Literal["P", "0.5", "0", "LP", None]):
        q = lambda u: u*(1-u)
        u_D = Expression('x[0]<0.5 ? a : b',a=-0.1,b=0.4,degree=1)
        self.u_D = u_D
        MySemilinearProblem.__init__(self,N_h, q, Expression("0", degree=0), 
                                    u_D, solver, initGuess_strategy)

class Fisher_mms(MySemilinearProblem):
    """
    Elliptic Fisher's equation with a manufactured exact solution.

    This class implements the steady-state version of the Fisher equation:

        -div(D * ∇u) - u(1 - u) = f,

    The exact solution is:

        u_ex(x) = a * sin(2π * b * x),

    The source term 'f' is derived using symbolic differentiation with SymPy. 

    It is solved using 'solver' with 'initGuess_strategy'.

    Parameters
    ----------
    N_h : int
        Number of mesh intervals for spatial discretization 
    solver : NewtonSolver or PETScSNESSolver
        Solver object used to solve the PDE
    strategy : {"P", "0.5", "0", "LP", None} - specifies how to initialize 'self.u'. 
        Options:
        - "P":      Solves the Poisson equation (-div(D∇u) = f) 
                        and uses the result as initial guess.
        - "0.5":    Sets 'u = 0.5' (with boundary conditions).
        - "0":      Sets 'u = 0.0' (with boundary conditions).
        - "LP":     Solves a linearized version of the semilinear problem:
                        -div(D∇u) - u = f.
                        and uses the result as initial guess.
        - None:     Uses the current value of 'self.u' 
    """

    def __init__(self, N_h:int, solver:NewtonSolver | PETScSNESSolver, 
                 initGuess_strategy:Literal["P", "0.5", "0", "LP", None]):
        import sympy as sp
        q = lambda u: u*(1-u)
        x, mu, a, b= sp.symbols('x[0] mu a b') 
        u_ex = a * sp.sin(2*sp.pi*b*x)
        f = sp.simplify(-sp.diff(mu*sp.diff(u_ex, x), x) - q(u_ex))
        f_expr = Expression(sp.printing.ccode(f), a=1, b=2.4, mu=1, degree=2)
        self.u_D = Expression(sp.printing.ccode(u_ex), a=1, b=2.4, degree=4)

        MySemilinearProblem.__init__(self,N_h, q, f_expr, self.u_D, solver, 
                                     initGuess_strategy)

    def solve(self, mu:float):#, 
        solInfo, NitInfos = self._solve(mu)
        solInfo.update(self.errornorms())
        return solInfo, NitInfos

    def errornorms(self):
        """
        Computes error norms between the computed solution 'self.u' and the exact 
        solution 'self.u_D'.

        Returns
        -------
        dict 
            A dictionary with error norms:

            - ||u-u_e||_L2: L2 norm 
            - ||u_dofs-u_e_dofs||_inf: Maximum norm 
            - |u-u_e|_H10: H1 seminorm
        """

        # Infinity norm based on nodal values
        u_e_ = interpolate(self.u_D, self.V_h)
        E4 = abs(u_e_.vector().get_local() - self.u.vector().get_local()).max()

        # L2 norm
        E5 = errornorm(self.u_D, self.u, norm_type='L2', degree_rise=5)

        # H1 seminorm
        E6 = errornorm(self.u_D, self.u, norm_type='H10', degree_rise=3)

        errors = {'||u-u_e||_L2': E5, 
                '||u_dofs-u_e_dofs||_inf': E4,
                '|u-u_e|_H10': E6}

        return errors

class SemilinearPoisson(MySemilinearProblem):
    """
    Semilinear Poisson equation with cubic nonlinearity

    This class implements a semilinear elliptic PDE:
        -div(D ∇u) + u³ = 0

    It is solved using 'solver' with 'initGuess_strategy'.

    Parameters
    ----------
    N_h : int
        Number of mesh intervals for spatial discretization 
    solver : NewtonSolver or PETScSNESSolver
        Solver object used to solve the PDE
    initGuess_strategy : {"P", "0.5", "0", "LP", None} - specifies how to initialize 'self.u'. 
        Options:
        - "P":      Solves the Poisson equation (-div(D∇u) = f) 
                        and uses the result as initial guess.
        - "0.5":    Sets 'u = 0.5' (with boundary conditions).
        - "0":      Sets 'u = 0.0' (with boundary conditions).
        - "LP":     Solves a linearized version of the semilinear problem:
                        -div(D∇u) - u = f.
                        and uses the result as initial guess.
        - None:     Uses the current value of 'self.u' 
    """
    def __init__(self, N_h:int, solver:NewtonSolver | PETScSNESSolver, 
                 initGuess_strategy:Literal["P", "0.5", "0", "LP", None]):
        q = lambda u: -u*u*u
        self.u_D = Expression('x[0]<0.5 ? a : b',a=-0.1,b=0.4,degree=1)
        MySemilinearProblem.__init__(self, N_h, q, Expression("0", degree=0), 
                                    self.u_D, solver, initGuess_strategy)
    
class Poisson(MySemilinearProblem):
    """
    linear Poisson problem (special case of semilinear PDE)

    It is solved using 'solver'.

    Parameters
    ----------
    N_h : int
        Number of mesh intervals for spatial discretization 
    solver : NewtonSolver or PETScSNESSolver
        Solver object used to solve the PDE
    initGuess_strategy : {"P", "0.5", "0", "LP", None} - specifies how to initialize 'self.u'. 
        Options:
        - "P":      Solves the Poisson equation (-div(D∇u) = f) 
                        and uses the result as initial guess.
        - "0.5":    Sets 'u = 0.5' (with boundary conditions).
        - "0":      Sets 'u = 0.0' (with boundary conditions).
        - "LP":     Solves a linearized version of the semilinear problem:
                        -div(D∇u) - u = f.
                        and uses the result as initial guess.
        - None:     Uses the current value of 'self.u' 
    """
    def __init__(self, N_h:int, solver:NewtonSolver | PETScSNESSolver, 
                 initGuess_strategy:Literal["P", "0.5", "0", "LP", None]):
        q = lambda u:Expression("0", degree=0) 
        self.u_D = Expression('x[0]<0.5 ? a : b',a=-0.1,b=0.4,degree=1)
        MySemilinearProblem.__init__(self,N_h, q, Expression("0", degree=0), 
                                     self.u_D, solver, initGuess_strategy)
 