"""
Read a Nastran/OptiStruct .fem or .bdf file and return plain numpy arrays
ready for DOLFINx. No DOLFINx dependency here.
"""
import numpy as np
from pyNastran.bdf.bdf import BDF

# Nastran element type -> (basix cell name, vertices per element, function space order)
CELL_MAP = {
    "CTETRA4":  ("tetrahedron", 4, 1),
    "CTETRA10": ("tetrahedron", 4, 2),   # 4 vertices for geometry, order 2 for function space
    "CHEXA8":   ("hexahedron", 8, 1),
    "CHEXA20":  ("hexahedron", 8, 2),
}


def read_fem(filename):
    model = BDF(mode="optistruct")
    model.read_bdf(filename)

    # --- Nodes ---
    nid_list = sorted(model.nodes.keys())
    nid_to_idx = {nid: i for i, nid in enumerate(nid_list)}
    coords = np.array([model.nodes[nid].get_position() for nid in nid_list])

    # --- Elements ---
    cells = []
    cell_mat_ids = []
    cell_type_key = None

    for eid in sorted(model.elements.keys()):
        elem = model.elements[eid]
        etype = elem.type
        nids = elem.node_ids
        nnodes = len(nids)
        key = f"{etype}{nnodes}"

        if key not in CELL_MAP:
            continue

        if cell_type_key is None:
            cell_type_key = key
        elif key != cell_type_key:
            print(f"WARNING: skipping {key} element {eid}, only {cell_type_key} supported")
            continue

        basix_cell, n_verts, order = CELL_MAP[key]

        # Only take vertex nodes for geometry
        vertex_ids = [nid_to_idx[nids[i]] for i in range(n_verts)]
        cells.append(vertex_ids)

        pid = elem.pid
        prop = model.properties[pid]
        mid = prop.mid
        cell_mat_ids.append(mid)

    cells = np.array(cells, dtype=np.int64)
    cell_mat_ids = np.array(cell_mat_ids, dtype=np.int32)

    # --- Materials ---
    materials = {}
    for mid, mat in model.materials.items():
        materials[mid] = {"E": mat.e, "nu": mat.nu, "rho": mat.rho}

    # --- SPCs ---
    spcs = []
    for spc_id, spc_list in model.spcs.items():
        for spc_set in spc_list:
            for spc_card in spc_set if isinstance(spc_set, list) else [spc_set]:
                card_type = spc_card.type
                if card_type == "SPC1":
                    components = str(spc_card.components)
                    for nid in spc_card.node_ids:
                        if nid in nid_to_idx:
                            for c in components:
                                dof = int(c) - 1
                                if dof < 3:
                                    spcs.append((nid_to_idx[nid], dof))
                elif card_type == "SPC":
                    for nid, comp, _ in zip(spc_card.node_ids,
                                            spc_card.components,
                                            spc_card.enforced):
                        if nid in nid_to_idx:
                            for c in str(comp):
                                dof = int(c) - 1
                                if dof < 3:
                                    spcs.append((nid_to_idx[nid], dof))

    # --- Nodal forces ---
    forces = []
    for load_id, load_list in model.loads.items():
        for load_set in load_list:
            for load_card in load_set if isinstance(load_set, list) else [load_set]:
                if load_card.type == "FORCE":
                    nid = load_card.node_id
                    if nid in nid_to_idx:
                        xyz = load_card.scaled_vector
                        forces.append((nid_to_idx[nid], xyz[0], xyz[1], xyz[2]))

    # --- Pack ---
    basix_cell, n_verts, element_order = CELL_MAP[cell_type_key]

    return {
        "coords": coords,
        "cells": cells,
        "cell_type": basix_cell,
        "element_order": element_order,  # for function space (2 for CTETRA10)
        "cell_mat_ids": cell_mat_ids,
        "materials": materials,
        "spcs": spcs,
        "forces": forces,
        "nid_to_idx": nid_to_idx,
    }