"""
fem2dolfinx: read a Nastran .fem, solve normal modes with DOLFINx + SLEPc.
P1 geometry + P2 function space (identical results for straight-sided elements).
"""
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc

import basix
import dolfinx
from dolfinx import mesh as dmesh, fem
from dolfinx.io import VTXWriter
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
import ufl

from nastran2fenicsx import read_fem

# ── Config ──
FEMFILE = "beam.fem"
N_MODES = 10
SIGMA = 0.0  # small positive shift avoids singular K


# ── Build mesh (P1 geometry) ──

def build_mesh(data):
    basix_cell = basix.CellType.tetrahedron if data["cell_type"] == "tetrahedron" else basix.CellType.hexahedron
    coord_elem = basix.ufl.element(basix.ElementFamily.P, basix_cell, 1, shape=(3,))
    domain = dmesh.create_mesh(MPI.COMM_WORLD, data["cells"], coord_elem, data["coords"])

    num_cells = domain.topology.index_map(domain.topology.dim).size_local
    unique_mats = np.unique(data["cell_mat_ids"])
    mat_to_tag = {mid: i for i, mid in enumerate(unique_mats)}
    tag_values = np.array([mat_to_tag[mid] for mid in data["cell_mat_ids"][:num_cells]], dtype=np.int32)
    mat_tags = dmesh.meshtags(domain, domain.topology.dim,
                              np.arange(num_cells, dtype=np.int32), tag_values)
    return domain, mat_tags


# ── Materials as DG0 fields ──

def assign_materials(domain, mat_tags, data):
    DG0 = fem.functionspace(domain, ("DG", 0))
    E = fem.Function(DG0)
    nu = fem.Function(DG0)
    rho = fem.Function(DG0)

    unique_mats = sorted(data["materials"].keys())
    mat_to_tag = {mid: i for i, mid in enumerate(unique_mats)}

    for mid, props in data["materials"].items():
        mask = mat_tags.values == mat_to_tag[mid]
        cells = mat_tags.indices[mask]
        E.x.array[cells] = props["E"]
        nu.x.array[cells] = props["nu"]
        rho.x.array[cells] = props["rho"]

    return E, nu, rho


# ── Variational forms (P2 function space on P1 geometry) ──

def build_forms(domain, data, E, nu, rho):
    V = fem.functionspace(domain, ("Lagrange", data["element_order"], (3,)))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    eps = lambda w: ufl.sym(ufl.grad(w))
    sig = lambda w: 2.0 * mu * eps(w) + lam * ufl.tr(eps(w)) * ufl.Identity(3)

    k_form = ufl.inner(sig(u), eps(v)) * ufl.dx
    m_form = rho * ufl.inner(u, v) * ufl.dx

    return V, k_form, m_form


# ── Boundary conditions ──

def apply_bcs(V, data):
    bcs = []
    node_coords = data["coords"]

    for component in range(3):
        constrained_nodes = [idx for idx, dof in data["spcs"] if dof == component]
        if not constrained_nodes:
            continue

        V_sub, sub_to_parent = V.sub(component).collapse()
        sub_coords = V_sub.tabulate_dof_coordinates()

        constrained_dofs = []
        for nidx in constrained_nodes:
            dists = np.linalg.norm(sub_coords - node_coords[nidx], axis=1)
            closest = np.argmin(dists)
            if dists[closest] < 1e-6:
                constrained_dofs.append(closest)

        parent_dofs = np.array(sub_to_parent, dtype=np.int32)[constrained_dofs]
        bc = fem.dirichletbc(PETSc.ScalarType(0), parent_dofs, V.sub(component))
        bcs.append(bc)

    return bcs


# ── Eigensolver ──

def solve_modes(K, M, n_modes, sigma=1.0):
    solver = SLEPc.EPS().create()
    solver.setDimensions(n_modes)
    solver.setProblemType(SLEPc.EPS.ProblemType.GHEP)

    st = SLEPc.ST().create()
    st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(sigma)

    ksp = st.getKSP()
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")


    st.setFromOptions()

    solver.setST(st)
    solver.setOperators(K, M)
    solver.solve()

    nconv = solver.getConverged()
    print(f"{nconv} eigenvalues converged")

    eigenvalues, eigenvectors = [], []
    vr, vi = K.createVecs()

    for i in range(min(nconv, n_modes)):
        lam = solver.getEigenpair(i, vr, vi)
        freq = np.sqrt(abs(lam.real)) / (2.0 * np.pi)
        eigenvalues.append(lam.real)
        eigenvectors.append(vr.copy())
        print(f"  Mode {i+1}: {freq:.4f} Hz")

    solver.destroy()
    return eigenvalues, eigenvectors


# ── Main ──

data = read_fem(FEMFILE)
print(f"{len(data['coords'])} nodes, {len(data['cells'])} {data['cell_type']} elements")
print(f"{len(data['spcs'])} constrained DOFs, {len(data['forces'])} forces")

domain, mat_tags = build_mesh(data)
E, nu, rho = assign_materials(domain, mat_tags, data)
V, k_form, m_form = build_forms(domain, data, E, nu, rho)

print(f"Function space: P{data['element_order']}, {V.dofmap.index_map.size_global * V.dofmap.bs} DOFs")

bcs = apply_bcs(V, data)
print(f"{len(bcs)} BC objects: {[len(bc.dof_indices()[0]) for bc in bcs]} DOFs each")

K = fem.petsc.assemble_matrix(fem.form(k_form), bcs=bcs)
K.assemble()
M = fem.petsc.assemble_matrix(fem.form(m_form), bcs=bcs, diag=0)
M.assemble()

eigenvalues, eigenvectors = solve_modes(K, M, N_MODES, SIGMA)

# Write mode shapes
vtx = VTXWriter(domain.comm, "modes.bp", [fem.Function(V, name="mode")])
for i, evec in enumerate(eigenvectors):
    mode_func = fem.Function(V, name="mode")
    mode_func.x.array[:] = evec.getArray()
    vtx.write(i)
vtx.close()

print("Modes written to modes.bp")