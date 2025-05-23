from fenics import *
from petsc4py import PETSc
from typing import Literal
import numpy as np

from sl_problems import MySemilinearProblem

class MyRedNonlinearProblem(NonlinearProblem):
    """
    Class for reduced elliptic nonlinear problem in one dimension
    using reduced basis projection of a given high-fidelity semilinear, elliptic
    problem 'hnl_problem'.

    Parameters
    ----------
    hnl_problem : MySemilinearProblem
        High-fidelity nonlinear problem, providing weak forms and mesh/function space info.

    RB : np.ndarray
        Reduced basis matrix of shape (n_dofs, N), where each column is a reduced basis vector.

    solver : NewtonSolver|PETScSNESSolver
        A Newton solver

    initGuess_strategy : Literal["P", "0.5", "0", None] - Strategy for the initial guess:

        - "P":      Solves the Poisson equation (-div(D∇u) = f).
        - "0.5":    Sets 'u = 0.5' (with boundary conditions).
        - "0":      Sets 'u = 0.0' (with boundary conditions).
        - None:     Uses the current value of 'self.u_rbc'

    Attributes
    ----------
    hnl_problem : MySemilinearProblem
        Reference to high-fidelity semilinear problem.

    RB : np.ndarray
        Reduced basis matrix.

    RB0 : np.ndarray
        Modified reduced basis matrix (homogeneous Dirichlet boundary conditions applied)

    N : int
        Dimension of the reduced basis 

    u_rbc : PETScVector
        PETSc vector with current reduced basis coefficients.

    solver : NewtonSolver | PETScSNESSolver
        solver used to solve this problem.

    initGuess_strategy : {"P", "0.5", "0", "LP", None}
        inital guess strategy used to solve this problem.
    """
    def __init__(self, hnl_problem:MySemilinearProblem, RB:np.ndarray, 
                 solver:NewtonSolver|PETScSNESSolver,
                 initGuess_strategy:Literal["P","0.5","0",None]):
        NonlinearProblem.__init__(self)

        # --- corresponding high-fidelity problem  --- #
        self.hnl_problem = hnl_problem

        # --- reduced basis --- #
        self.RB = RB

        # --- reduced basis considering zero Dirichlet boundary condition --- #
        self.RB0 = RB.copy(); self.RB0[0]=0; self.RB0[-1]=0

        # --- reduced basis dimension --- #
        self.N = RB.shape[1]

        # --- unknown reduced basis coefficents --- #
        self.u_rbc = PETScVector(PETSc.Vec().createSeq(self.N))

        # --- nonlinear solver and initGuess_strategy --- # 
        self.solver = solver
        self.initGuess_strategy = initGuess_strategy

        # --- utils...tmp ---
        V_N = FunctionSpace(IntervalMesh(RB.shape[1]-1,0,1),"P",1)
        self._L = inner(TrialFunction(V_N), TestFunction(V_N))*dx
        self._u_func = Function(V_N)
    
    def errornorms(self, u_h:Function|None, 
                    proj_norm:Literal["L2-norm", "2-norm"]):
        """
        Compute error norms between high-fidelity solution 'u_h' and current 
        reduced solution 'self.u_N()' and between 'u_h' and its projection
        onto the reduced space w.r.t. the norm 'proj_norm'.
        If 'u_h' is None, all error values are set to -1.0. 

        Parameters
        ----------
        u_h : Function
            High-fidelity solution
        proj_norm: {"L2-norm", "2-norm"}
            Type of norm used for projection.

        Returns
        -------
        errors : dict
            A dictionary containing
            - "||u_h-u_N||_L2": L2 norm of the error between 'self.u_N()' and 'u_h'.
            - "||u_h-P_N(u_h)||_L2": L2 norm of 'u_h' and its projection.

        """

        if u_h == None: # e.g. if hf-problem did not converge
            return {"||u_h-u_N||_L2"              : -1.0,
                    "||u_h-P_N(u_h)||_L2"         : -1.0}

        if proj_norm not in {"L2-norm", "2-norm"}:
            raise Exception(f"Projection w.r.t. {proj_norm} not implemented!")

        dx_q = dx(metadata={'quadrature_degree':9})

        P_u_h = self.proj(u_h, proj_norm=proj_norm)

        errL2 = assemble((self.u_N()-u_h)**2*dx_q)**0.5                     # same as errornorm(self.u_N(), u_h)
        PerrL2 = assemble((u_h-P_u_h)**2*dx_q)**0.5                         # same as errornorm(u_h, P_u_h)

        errors = {"||u_h-u_N||_L2"              : errL2,
                  "||u_h-P_N(u_h)||_L2"         : PerrL2}

        return errors

    def comp_u_N(self, u_rbc:PETScVector, bcs:list[DirichletBC]=None):
        """
        Compute reduced solution of given reduced basis coefficients 'u_rbc'
        in high-fidelity function space.

        Parameters
        ----------
        u_rbc : PETScVector
            Reduced basis coefficients
        bcs : list[DirichletBC], optional
            Boundary conditions, by default None

        Returns
        -------
        u_N : Function
            Reduced solution in high-fidelity function space.
        """
        # compute reduced solution in high-fidelity function space
        u_N = Function(self.hnl_problem.V_h)
        u_N.vector().set_local(self.RB @ u_rbc.get_local())
        if bcs is not None: [bc.apply(u_N.vector()) for bc in bcs] 
        return u_N

    def u_N(self):
        """
        Return current reduced solution of current reduced basis coefficients
        'self.u_rbc' in high-fidelity function space.

        Returns
        -------
        u_N : Function
            Current reduced solution in high-fidelity function space.
        """
        return self.comp_u_N(self.u_rbc)

    def set_initGuess_rbc(self, 
                          initGuess_strategy:Literal["P", "0.5", "0", None],
                          proj_norm:Literal["L2-norm", "2-norm"]="L2-norm"):
        """
        Set initial guess for reduced basis coefficients 'self.rbc'
        based on given strategy 'initGuess_strategy'.
        Compute high-fidelity function corresponding to initial guess strategy
        and project it onto the reduced space using projection norm 'proj_norm'. 

        Parameters
        ----------
        initGuess_strategy : Literal["P", "0.5", "0", None] - Strategy for the initial guess:
            Options:
            
            - "P":      Solves the Poisson equation (-div(D∇u) = f).
            - "0.5":    Sets 'u = 0.5' (with boundary conditions).
            - "0":      Sets 'u = 0.0' (with boundary conditions).
            - None:     Uses the current value of 'self.u_rbc'

        proj_norm: Literal["L2-norm", "2-norm"]
            Type of norm used for projection. 
        
        """

        V_h = self.hnl_problem.V_h

        if initGuess_strategy == "P":
            from l_problem import LinearProblem
            # --- Poisson weak formulation --- #
            u = TrialFunction(V_h) 
            v = TestFunction(V_h)
            Pwf =self.hnl_problem.mu_expr * inner(grad(u),grad(v)) * dx\
                              - inner(self.hnl_problem.f, v) * dx
            # --- solve linear system --- #
            u_h = Function(self.hnl_problem.V_h)
            l_problem = LinearProblem(Pwf, self.hnl_problem.bcs)
            l_problem.solve(u_h.vector())

        elif initGuess_strategy == "0.5":
            u_h = Function(V_h)
            u_h.vector()[:] = 0.6 #2.0#
            self.hnl_problem.bcs[0].apply(u_h.vector())

        elif initGuess_strategy == "0":
            self.u_N_dof[:] = 0.
            return 

        else:
            raise Exception(f"initial guess strategy '{initGuess_strategy}'\
                                not implemented")

        if proj_norm == "2-norm":#Für POD_2
            self.u_rbc.set_local(self.RB.T @ u_h.vector().get_local())
        elif proj_norm == "L2-norm":
            M = assemble(inner(TrialFunction(V_h),TestFunction(V_h))*dx).array()
            self.u_rbc.set_local(self.RB.T @ M @ u_h.vector().get_local())
        else:
            raise Exception(f"Projection w.r.t. {proj_norm} not implemented!")

    def proj(self, u_h:Function, proj_norm:Literal["L2-norm","2-norm"]):
        """
        Project given high-fidelity function 'u_h' onto reduced basis space
        using norm 'proj_norm' in projetion.

        Parameters
        ----------
        u_h : Function
            High fidelity function
        proj_norm: Literal["L2-norm", "2-norm"]
            Type of norm used for projection. 

        Returns
        -------
        u_N : Function 
            Projected function in high-fidelity function space.

        """
    
        V_h = u_h.function_space()
        if proj_norm=="L2-norm":
            M = assemble(inner(TrialFunction(V_h),TestFunction(V_h))*dx).array()
            u_rbc = self.RB @ self.RB.T @ M @ u_h.vector().get_local()
        elif proj_norm== "2-norm":
            u_rbc = self.RB @ self.RB.T @ u_h.vector().get_local()
        else:
            raise Exception(f"Projection w.r.t. {proj_norm} not implemented!")

        u_N = Function(V_h)
        u_N.vector().set_local(u_rbc)
        return u_N

    def _J(self, u_rbc:PETScVector):
        """
        Assemble reduced Jacobian matrix for semilinear problem at given reduced 
        basis coefficients 'u_rbc'.

        Parameters
        ----------
        u_rbc : PETScVector
            Reduced basis coefficients at which reduced Jacobian is computed.

        Returns
        -------
        J_N : np.ndarray
            reduced Jacobian matrix 
        """
        # --- reduced Function u_N --- #
        u_N = self.comp_u_N(u_rbc, self.hnl_problem.bcs)
        #u_rbc = self.RB.T @ u_N.vector().get_local()          ####??? Notwendig????

        # --- high fidelity Jacobian matrix at u_N --- #
        du = TrialFunction(self.hnl_problem.V_h)
        J_h = assemble(derivative(action(self.hnl_problem.nlwf, u_N), u_N, du))
        self.hnl_problem.bc0.apply(J_h)

        # --- project J_h onto reduced space --- #
        J_N = self.RB0.T @ J_h.array() @ self.RB0
        return J_N

    #def _F(self, u_rbc:PETScVector, ret_Gh = False):
    def _F(self, u_rbc:PETScVector):
        """
        Assemble reduced resicual vector for semilinear problem at given reduced 
        basis coefficients 'u_rbc'.

        Parameters
        ----------
        u_rbc : PETScVector
            Reduced basis coefficients at which reduced recidual is computed.

        Returns
        -------
        G_N : np.ndarray
            reduced residual vector
        """
        # --- reduced Function u_N --- #
        u_N = self.comp_u_N(u_rbc, self.hnl_problem.bcs)
        #u_rbc = self.RB.T @ u_N.vector().get_local()          ####??? Notwendig????

        # --- high fidelity residual vector at u_N --- #
        G_h = assemble(action(self.hnl_problem.nlwf, u_N))
        self.hnl_problem.bc0.apply(G_h)

        # --- project G_h into V_N --- #
        G_N = self.RB0.T @ G_h.get_local()
        return G_N
        #G_N = self.RB.T @ G_h.get_local()
        #if ret_Gh == False:
            #return G_N
        #else:
        #   return G_N, norm(G_h)
    

    def J(self, A:PETScMatrix, x:PETScVector):
        """
        Assign reduced Jacobian matrix 'self._J(x)' at given reduced basis coefficients 'x'
        to PETScMatrix 'A'.
    
        Parameters
        ----------
        A : PETScMatrix
            Matrix object to be filled with the reduced Jacobian matrix.
        x : PETScVector
            Reduced basis coefficients at which reduced Jacobian is evaluated.
        """

        # --- compute reduced Jacobian --- #
        J_ = self._J(x)

        #TODO: following assembly is actually unnecessary... only performed
        #       to initialize the PETScMatrix correct. how can this be done more efficient?
        assemble(self._L, tensor=A)
        # --- initialize A --- #
        #layout = TensorLayout()
        #layout.rows = J_.shape[0]
        #layout.cols = J_.shape[1]
        #A.init(layout)

        # --- assign J_ to A --- #
        J_petsc = PETSc.Mat().createDense(J_.shape, array=J_)
        A.mat().zeroEntries()
        J_petsc.copy(result=A.mat())
    
    def F(self, b:PETScMatrix, x:PETScVector):
        """
        Assign reduced residual vector 'self._F(x)' at given reduced basis coefficients 'x'
        to PETScVector 'b'.

        Parameters
        ----------
        b : PETScMatrix
            Vector object to be filled with reduced residual vector.
        x : PETScVector
            Reduced basis coefficients at which reduced residual is evaluated.
        """
        F_ = self._F(x)

        #TODO: following assembly is actually unnecessary... only performed
        #       to initialize the PETScMatrix correct. how can this be done more efficient?
        assemble(action(self._L, self._u_func), tensor=b)
        # --- initialize b --- #
        #b.init(len(F_))

        b.set_local(F_)

    def solve(self, mu:float):
        """
        Solve reduced nonlinear problem for given parameter value 'mu'
        with initial guess strategy 'self.initGuess_strategy' using 'self.solver'.

        Parameters
        ----------
        mu : float
            parameter values

        Returns
        -------
        solInfo: dict
            information about solution
        NitInfos : dict
            information about Newton iterations

        Note
        ----
        The initial guess strategy is specified in 'self.initGuess_strategy'
        when instantiating a 'MyRedNonlinearProblem' object.
        It can be changed by modifying 'self.initGuess_strategy'.
        """

        self.hnl_problem.set_mu(mu)
        self.set_initGuess_rbc(self.initGuess_strategy)
        nit, conv = self.solver.solve(self, self.u_rbc)    
        self.solver.solInfo.update({"converged":int(conv)})

        tmp = {"N":self.N, "μ":mu}; tmp.update(self.solver.solInfo)
        return tmp, self.solver.NitInfos









