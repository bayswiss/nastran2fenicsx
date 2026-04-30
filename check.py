"""
Read .fem, build mesh, mark BC dofs as 1, export to VTX for ParaView check.
"""
import numpy as np
from mpi4py import MPI
import basix
import dolfinx
from dolfinx import mesh as dmesh, fem
from dolfinx.io import VTXWriter
from petsc4py import PETSc

from nastran2fenicsx import read_fem

FEMFILE = "beam.fem"

data = read_fem(FEMFILE)
print(f"{len(data['coords'])} nodes, {len(data['cells'])} {data['cell_type']} elements")
print(f"{len(data['spcs'])} constrained DOFs")

# Build mesh
basix_cell = basix.CellType.tetrahedron if data["cell_type"] == "tetrahedron" else basix.CellType.hexahedron
# coord_elem = basix.ufl.element(basix.ElementFamily.P, basix_cell, data["element_order"], shape=(3,))
coord_elem = basix.ufl.element(basix.ElementFamily.P, basix_cell, data["element_order"], shape=(3,))

domain = dmesh.create_mesh(MPI.COMM_WORLD, data["cells"], coord_elem, data["coords"])

# Function space
V = fem.functionspace(domain, ("Lagrange", data["element_order"], (3,)))
bc_marker = fem.Function(V, name="bc_marker")
bc_marker.x.array[:] = 0.0

# Mark constrained DOFs
node_coords = data["coords"]
dof_coords = V.tabulate_dof_coordinates()
bs = V.dofmap.bs

found = 0
for node_idx, component in data["spcs"]:
    coord = node_coords[node_idx]
    dists = np.linalg.norm(dof_coords - coord, axis=1)
    closest = np.argmin(dists)
    dist = dists[closest]
    if dist < 1e-6:
        bc_marker.x.array[closest * bs + component] = 1.0
        found += 1
    else:
        print(f"  WARNING: node {node_idx} comp {component} no match (dist={dist:.3e})")

print(f"Marked {found}/{len(data['spcs'])} DOFs")

# Write
vtx = VTXWriter(domain.comm, "bc_check.bp", [bc_marker])
vtx.write(0)
vtx.close()
print("Written to bc_check.bp")