from dolfin import *

def matrices(N):
    '''
    Stokes system and preconditioner over mixed function space.
    '''
    mesh = UnitSquareMesh(N, N)
    sub_domains = FacetFunction('size_t', mesh, 0)
    CompiledSubDomain('near(x[0], 0.)').mark(sub_domains, 1)
    CompiledSubDomain('near(x[0], 1.)').mark(sub_domains, 2)

    # Define function spaces
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    W = MixedFunctionSpace([V, Q])

    # No-slip boundary condition for velocity
    noslip = Constant((0, 0))
    bc0 = DirichletBC(W.sub(0), noslip, sub_domains, 0)

    # Inflow boundary condition for velocity
    inflow = Expression(("-sin(x[1]*pi)", "0.0"))
    bc1 = DirichletBC(W.sub(0), inflow, sub_domains, 1)

    # Boundary condition for pressure at outflow
    zero = Constant(0)
    bc2 = DirichletBC(W.sub(1), zero, sub_domains, 2)

    # Define variational problem and assemble matrices
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    f = Constant((0, 0))

    a = inner(grad(v), grad(u))*dx + div(v)*p*dx + div(u)*q*dx
    b = inner(grad(v), grad(u))*dx + p*q*dx
    L = inner(f, v)*dx

    bcs = [bc0, bc1, bc2]

    # System
    A, _ = assemble_system(a, L, bcs)

    # Preconditioner
    B, _ = assemble_system(b, L, bcs)

    AA = as_backend_type(A).mat()
    BB = as_backend_type(B).mat()

    return AA, BB
