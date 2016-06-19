cbc.block has its own suite of linear solvers but misses routines for computing eigenvalues of the assembled systems.
Rather than implementing the eigenvalue algorithms for cbc.block containers the idea here is to come up with a representation
which would allow us to hook up into slepc4py. The right way seems to be PETSc's MatNest. For objects which are represented by
their action, e.g. Gauss-Seidel, we will explore Shell.
