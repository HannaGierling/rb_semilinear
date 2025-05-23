from fenics import * 

class LinearProblem():
    """ Problem class for linear elliptic PDEs

    Parameters
    ----------
    lin_weakform : form
        weak formulation of linear problem
    bcs : list[DirichletBC]
        list of DirichletBC 
    
    Attributes
    ----------
    a : form
        bilinear form (lhs)
    L : form
        linear form (rhs)
    bcs : list[DirichletBC]
        list of DirichletBC 
    """
    def __init__(self, lin_weakform:Form, bcs:list[DirichletBC]):
        # bilinear form
        self.a = lhs(lin_weakform)
        # linear form
        self.L = rhs(lin_weakform) 
        self.bcs = bcs

    def solve(self, u_vec:PETScVector):
        """Method for solving a linear variational problem.
    
        Solves of the form :math:`a(u,v) = L(v)\\,\\forall v\\in V`
        using PETSc's LU-solver.
        """
        linSolver = PETScLUSolver()
        A = assemble(self.a)
        b = assemble(self.L)
        [bc.apply(A,b) for bc in self.bcs]
        linSolver.solve(A, u_vec, b)

    