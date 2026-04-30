from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
from dolfinx import mesh 
from dolfinx.fem import Function, functionspace, form, dirichletbc
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.io import VTXWriter
from dolfinx import default_scalar_type
from ufl import dx, sym, grad, tr, inner, TrialFunction, TestFunction, Identity
import basix
import numpy as np
from scipy.spatial import cKDTree

from nastran2fenicsx import read_fem

comm = MPI.COMM_WORLD

# Read .fem on rank 0, broadcast to all ranks
if comm.rank == 0:
    data = read_fem("beam.fem", mode="optistruct")
else:
    data = None
data = comm.bcast(data, root=0)

# Build mesh: P1 geometry, rank 0 owns all cells/coords, others start empty
basix_cell = basix.CellType.tetrahedron if data["cell_type"] == "tetrahedron" else basix.CellType.hexahedron
geom_elem = basix.ufl.element(basix.ElementFamily.P, basix_cell, 1, shape=(3,))

if comm.rank == 0:
    cells = data["cells"]
    coords = data["coords"]
else:
    cells = np.empty((0, data["cells"].shape[1]), dtype=np.int64)
    coords = np.empty((0, 3), dtype=np.float64)

msh = mesh.create_mesh(comm, cells, geom_elem, coords)

# Material tags: use original_cell_index to map back through DOLFINx's
# post-partitioning reorder, then build a per-cell tag array
num_cells = msh.topology.index_map(msh.topology.dim).size_local
orig_idx = msh.topology.original_cell_index[:num_cells]

unique_mats = sorted(data["materials"].keys())
mat_to_tag = {mid: i for i, mid in enumerate(unique_mats)}
all_tags = np.array([mat_to_tag[mid] for mid in data["cell_mat_ids"]], dtype=np.int32)
tag_values = all_tags[orig_idx]

mat_tags = mesh.meshtags(msh, msh.topology.dim,
                          np.arange(num_cells, dtype=np.int32), tag_values)

# Material parameters as DG0 (piecewise-constant per element) fields
DG0 = functionspace(msh, ("DG", 0))
E = Function(DG0)
ni = Function(DG0)
rho = Function(DG0)

for mid, props in data["materials"].items():
    mask = mat_tags.values == mat_to_tag[mid]
    cell_ids = mat_tags.indices[mask]
    E.x.array[cell_ids] = props["E"]
    ni.x.array[cell_ids] = props["nu"]
    rho.x.array[cell_ids] = props["rho"]

lambda_ = ni * E / ((1 + ni) * (1 - 2 * ni))
mu = E / (2 * (1 + ni))

# Test and trial functions on P2 space (P1 geometry, P2 displacement)
el_deg = data["element_order"]
V = functionspace(msh, ("CG", el_deg, (3,)))
u = TrialFunction(V)
v = TestFunction(V)

# Strain and stress tensors
def epsilon(u):
    return sym(grad(u))

def sigma(u):
    return lambda_ * tr(epsilon(u)) * Identity(len(u)) + 2 * mu * epsilon(u)

# Dirichlet BCs: one per component, matching SPC nodes by coordinate
# (the .fem gives us NIDs, not facet tags, so we match dof coords back to
#  the original node positions)
bcs = []
node_coords = data["coords"]

for component in range(3):
    constrained_nodes = [idx for idx, dof in data["spcs"] if dof == component]
    if not constrained_nodes:
        continue

    V_sub, sub_to_parent = V.sub(component).collapse()
    sub_coords = V_sub.tabulate_dof_coordinates()


    if len(sub_coords) == 0:
        continue

    tree = cKDTree(sub_coords)
    target_coords = node_coords[constrained_nodes]
    dists, closest = tree.query(target_coords)

    matched = closest[dists < 1e-6]
    parent_dofs = np.array(sub_to_parent, dtype=np.int32)[matched]
    parent_dofs = np.sort(parent_dofs)
    
    bc = dirichletbc(default_scalar_type(0), parent_dofs, V.sub(component))
    bcs.append(bc)


        
# Define and assemble matrices
k = inner(sigma(u), epsilon(v)) * dx
m = rho * inner(u, v) * dx

K = assemble_matrix(form(k), bcs=bcs)
M = assemble_matrix(form(m), bcs=bcs, diag=0)

print("Assemble...")
K.assemble()
M.assemble()

print("Solver setup...")
N_modes = 10
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
solver.setOperators(K, M)

print("Solve...")
solver.solve()

tol, maxit = solver.getTolerances()
nconv = solver.getConverged()

if msh.comm.rank == 0:
    print("Number of iterations of the method: %i" % solver.getIterationNumber())
    print("Solution method: %s" % solver.getType())
    print("")
    print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))
    print(f"{nconv} eigenvalues converged")

# Save mode shapes
xr = Function(V)
vtx = VTXWriter(msh.comm, "modes.bp", [xr])

eig_freq = []
for i in range(min(nconv, N_modes)):
    lam_eig = solver.getEigenpair(i, xr.x.petsc_vec)
    fn = np.sqrt(abs(lam_eig.real)) / (2 * np.pi)
    eig_freq.append(fn)
    if msh.comm.rank == 0:
        print("%12f Hz" % fn)

    xr.x.scatter_forward()
    vtx.write(np.round(fn, 2))

vtx.close()

if msh.comm.rank == 0:
    print("Modes written to modes.bp")