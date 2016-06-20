from petsc4py import PETSc
from slepc4py import SLEPc
from dolfin import info
import numpy as np


def eig(AA, BB):
    '''Now we would like to get the eigenvaules'''
    # Setup the eigensolver
    E = SLEPc.EPS().create()
    E.setOperators(AA ,BB)
    E.setType(E.Type.GD)
    nev = 3  # Number of eigenvalues
    E.setDimensions(nev, PETSc.DECIDE)
    E.setWhichEigenpairs(E.Which.SMALLEST_REAL)
    E.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    E.setFromOptions()

    # Solve the eigensystem
    E.solve()

    info('System size: %i' % AA.size[0])
    its = E.getIterationNumber()
    info('Number of iterations of the method: %i' % its)
    nconv = E.getConverged()
    info('Number of converged eigenpairs: %d' % nconv)
    if nconv > 0:
        return np.array([E.getEigenvalue(i).real for i in range(nev)])
    else:
        return np.array([])

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    import stokes_df
    import stokes

    print '\tMatNest'
    block = eig(*stokes.matrices(10))

    print '\tDolfin'
    df = eig(*stokes_df.matrices(10))

    info('\nError %r' % np.linalg.norm(block-df))
    info('Values %r' % df)

    # FIXME: Which EPS.TYPE can and should be used?
    # FIXME: Shell for cbc.block Jacobi etc ...
    # FIXME: What about parallel? Does cbc.block work in parallel?
