from dolfin import *
from block import *
from block.dolfin_util import *
from block.iterative import *
from block.algebraic.petsc import *

import os

dolfin.set_log_level(15)
if MPI.size(None) > 1:
    print "Stokes demo does not work in parallel because of old-style XML mesh files"
    exit()

mesh = UnitSquareMesh(20, 20)
sub_domains = FacetFunction('size_t', mesh, 0)
CompiledSubDomain('near(x[0], 0.)').mark(sub_domains, 1)
CompiledSubDomain('near(x[0], 1.)').mark(sub_domains, 2)

# Define function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)

# No-slip boundary condition for velocity
noslip = Constant((0, 0))
bc0 = DirichletBC(V, noslip, sub_domains, 0)

# Inflow boundary condition for velocity
inflow = Expression(("-sin(x[1]*pi)", "0.0"))
bc1 = DirichletBC(V, inflow, sub_domains, 1)

# Boundary condition for pressure at outflow
zero = Constant(0)
bc2 = DirichletBC(Q, zero, sub_domains, 2)

# Define variational problem and assemble matrices
v, u = TestFunction(V), TrialFunction(V)
q, p = TestFunction(Q), TrialFunction(Q)

f = Constant((0, 0))

a11 = inner(grad(v), grad(u))*dx
a12 = div(v)*p*dx
a21 = div(u)*q*dx

I  = assemble(p*q*dx)

bcs = [[bc0, bc1], bc2]

# System
AA = block_assemble([[a11, a12],
                     [a21,  0 ]])

# Preconditioner
BB = block_assemble([[a11, 0],
                     [0,  I]])

block_bc(bcs, True).apply(AA)
block_bc(bcs, True).apply(AA)

# Extract the individual submatrices
[[A00, A01], [A10, A11]] = AA
[[B00, _], [_, B11]] = BB

# Get petsc4py objects
A00, A01, A10, A11 = [as_backend_type(mat).mat() for mat in (A00, A01, A10, A11)]
B00, B11 = [as_backend_type(mat).mat() for mat in (B00, B11)]

from petsc4py import PETSc
null = PETSc.Mat()

comm = mpi_comm_world()
# Create matnest objects
AA = PETSc.Mat().createNest([[A00, A01], [A10, A11]], comm=comm)
BB = PETSc.Mat().createNest([[B00, null], [null, B11]], comm=comm)

from slepc4py import SLEPc
Print = PETSc.Sys.Print
# Now we would like to get the eigenvaules
xr, tmp = AA.getVecs()
xi, tmp = AA.getVecs()

# Setup the eigensolver
E = SLEPc.EPS().create()
E.setOperators(AA ,BB)
E.setType(E.Type.GD)
E.setDimensions(10, PETSc.DECIDE)
E.setWhichEigenpairs(E.Which.SMALLEST_REAL)
E.setProblemType(SLEPc.EPS.ProblemType.GHEP)
E.setFromOptions()

# Solve the eigensystem
E.solve()

Print("")
its = E.getIterationNumber()
Print("Number of iterations of the method: %i" % its)
sol_type = E.getType()
Print("Solution method: %s" % sol_type)
nev, ncv, mpd = E.getDimensions()
Print("Number of requested eigenvalues: %i" % nev)
tol, maxit = E.getTolerances()
Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))
nconv = E.getConverged()
Print("Number of converged eigenpairs: %d" % nconv)
if nconv > 0:
    Print("")
    Print("        k          ||Ax-kx||/||kx|| ")
    Print("----------------- ------------------")
    for i in range(nconv):
        k = E.getEigenpair(i, xr, xi)
        Print(" %12f" % k.real)
    Print("")
