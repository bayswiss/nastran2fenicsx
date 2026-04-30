"""
Three outputs to compare:
  1. mesh_P2.xdmf  — quadratic geometry mesh (XDMF instead of VTX, to rule out writer issue)
  2. mesh_P1_V2.bp  — P1 geometry + P2 function space (the pragmatic approach)
  3. Prints geometry coords from the P2 mesh so we can sanity-check programmatically
"""
import numpy as np
from mpi4py import MPI
import basix
import dolfinx
from dolfinx import mesh as dmesh, fem
from dolfinx.io import VTXWriter, XDMFFile
from pyNastran.bdf.bdf import BDF

FEMFILE = "beam.fem"
PERM = [0, 1, 2, 3, 4, 6, 7, 5, 8, 9]

# ── Read ──
model = BDF(mode="optistruct")
model.read_bdf(FEMFILE)

nid_list = sorted(model.nodes.keys())
nid_to_idx = {nid: i for i, nid in enumerate(nid_list)}
coords = np.array([model.nodes[nid].get_position() for nid in nid_list])

cells_permuted = []
for eid in sorted(model.elements.keys()):
    elem = model.elements[eid]
    if elem.type == "CTETRA" and len(elem.node_ids) == 10:
        nids = elem.node_ids
        cells_permuted.append([nid_to_idx[nids[p]] for p in PERM])

cells_permuted = np.array(cells_permuted, dtype=np.int64)
cells_linear = cells_permuted[:, :4]
print(f"{len(coords)} nodes, {len(cells_permuted)} elements")


# ── Test 1: P2 geometry mesh, written to XDMF ──
print("\n--- Test 1: P2 geometry, XDMF output ---")
ce2 = basix.ufl.element(basix.ElementFamily.P, basix.CellType.tetrahedron, 2, shape=(3,))
mesh2 = dmesh.create_mesh(MPI.COMM_WORLD, cells_permuted, ce2, coords)

with XDMFFile(mesh2.comm, "mesh_P2.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh2)
print("Written to mesh_P2.xdmf")

# Quick sanity: check mesh geometry bounding box
x = mesh2.geometry.x
print(f"  Geometry nodes: {x.shape[0]}")
print(f"  Bounding box: [{x.min(axis=0)}] to [{x.max(axis=0)}]")


# ── Test 2: P1 geometry + P2 function space ──
print("\n--- Test 2: P1 geometry + P2 function space ---")
ce1 = basix.ufl.element(basix.ElementFamily.P, basix.CellType.tetrahedron, 1, shape=(3,))
mesh1 = dmesh.create_mesh(MPI.COMM_WORLD, cells_linear, ce1, coords)

V2 = fem.functionspace(mesh1, ("Lagrange", 2, (3,)))
f = fem.Function(V2, name="test")
f.x.array[:] = 1.0

vtx = VTXWriter(mesh1.comm, "mesh_P1_V2.bp", [f])
vtx.write(0)
vtx.close()
print("Written to mesh_P1_V2.bp")
print(f"  P1 mesh nodes: {mesh1.geometry.x.shape[0]}")
print(f"  P2 function space DOFs: {V2.dofmap.index_map.size_global * V2.dofmap.bs}")