from dolfinx.io import gmsh as gmshio, XDMFFile, VTXWriter
from mpi4py import MPI
from dolfinx.fem import Function, functionspace, form, dirichletbc, locate_dofs_topological, locate_dofs_geometrical
from ufl import dx, grad, inner, TrialFunction, TestFunction, nabla_grad, nabla_div, Identity
from dolfinx.fem.petsc import assemble_matrix
from slepc4py import SLEPc
from petsc4py import PETSc
import numpy as np

import time

mesh_data = gmshio.read_from_msh(
    "beam.msh", 
    MPI.COMM_WORLD, 
    0, 
    gdim = 3
)

mesh = mesh_data.mesh
cell_tags = mesh_data.cell_tags
facet_tags = mesh_data.facet_tags

# Material parameters
E = 210e9
ni = 0.3
rho = 7850

lambda_ = ni*E / ((1+ni)*(1-2*ni))
mu = E/(2*(1+ni))

el_deg = 2
# set up test and trial functions
V = functionspace(mesh, ("CG", el_deg, (3, )))
u = TrialFunction(V)
v = TestFunction(V)

#defining strain and stress tensors 
def epsilon(u):
    return 0.5 * (nabla_grad(u) + nabla_grad(u).T)

def sigma(u):
    return lambda_ * nabla_div(u) * Identity(len(u)) + 2 * mu * epsilon(u)


# Dirichlet BC (on faces)
bc_1 = dirichletbc(
       PETSc.ScalarType((0,0,0)), 
       locate_dofs_topological(V, mesh.topology.dim - 1, facet_tags.find(2)), 
       V
)


# define and assemble matrices
start_time = time.time()
k = inner(sigma(u),epsilon(v)) * dx
m = rho * inner(u,v) * dx

K = assemble_matrix(form(k), bcs =[bc_1])
M = assemble_matrix(form(m), bcs =[bc_1], diag=0)


print("Assemble...")
K.assemble()
M.assemble()

print("Solver setup...")
# Solver setup
N_modes = 2
solver = SLEPc.EPS().create()
solver.setDimensions(N_modes)
solver.setProblemType(SLEPc.EPS.ProblemType.GHEP)

st = SLEPc.ST().create()
st.setType(SLEPc.ST.Type.SINVERT)
st.setShift(0.0)

ksp = st.getKSP()
ksp.setType('preonly')      
pc = ksp.getPC()
pc.setType('lu')
pc.setFactorSolverType('mumps') 

st.setFromOptions()



solver.setST(st)
solver.setOperators(K,M)

print("Solve...")
solver.solve()

tol, maxit = solver.getTolerances()
nconv = solver.getConverged()

if mesh.comm.rank==0:
    print("Number of iterations of the method: %i" % solver.getIterationNumber())
    print("Solution method: %s" % solver.getType())
    print("")
    print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))


# Save the k
vals = [(i, solver.getEigenvalue(i)) for i in range(nconv)]

# Sort k by real part
# vals.sort(key=lambda x: x[1].real) ### CHECK

xr = Function(V)

ghosts=V.dofmap.index_map.ghosts

eig_vector = []
eig_matrix = np.zeros((len(xr.x.array), nconv), dtype=complex)
eig_freq = []
# print(max(ghosts))

vtx = VTXWriter(mesh.comm, "fields/Modes_3D.bp", [xr])
if nconv>0:
    for i, k in vals:
        solver.getEigenpair(i, xr.x.petsc_vec)
        fn = np.sqrt(k)/(2 * np.pi) 
        eig_freq.append(fn)
        if mesh.comm.rank==0:
            print("%12f Hz" % fn)

        xr.x.scatter_forward()

        vect = xr.x.petsc_vec.getArray()
        # print("LUNGHEZZA EIG")
        # print(len(vect))
        # print(end-start)
        eig_vector.append(vect.copy())
        # eig_vector_2[:,i] = vect
        # xr_ = mesh.comm.gather(xr.x.array, root=0)
        vtx.write(np.round(fn, 2))
        eig_matrix[:,i]=xr.x.array
        # np.delete(eig_matrix, ghosts, axis=0)