# Copyright (C) 2026 Antonio Baiano Svizzero
#
# This file is part of nastran2fenicsx (https://github.com/bayswiss/nastran2fenicsx)
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Read a Nastran/OptiStruct .fem or .bdf file and return plain numpy arrays
# ready for DOLFINx.

import numpy as np
from pyNastran.bdf.bdf import BDF

# Maps "<Nastran element type><node count>" to the corresponding
# (basix cell name, vertices per element, function space order).
# CTETRA10 and CHEXA20 are quadratic elements: we only keep the corner
# vertices for the geometry and let DOLFINx add the midside nodes via
# a higher-order function space.
CELL_MAP = {
    "CTETRA4":  ("tetrahedron", 4, 1),
    "CTETRA10": ("tetrahedron", 4, 2),
    # "CHEXA8":   ("hexahedron",  8, 1), <--- NOT TESTED
    # "CHEXA20":  ("hexahedron",  8, 2), <--- NOT TESTED
}


def flatten(card_list):
    """Return a flat list of cards.

    pyNastran sometimes wraps cards inside an inner list (e.g. when an
    SPCADD or LOAD card combines several sets). This unwraps one level
    so the caller can iterate cards uniformly.
    """
    out = []
    for s in card_list:
        if isinstance(s, list):
            out.extend(s)
        else:
            out.append(s)
    return out


def spc_dofs(components):
    """Convert a Nastran component string to 0-based translational DOFs.

    Nastran encodes constrained DOFs as digits: 1,2,3 = translations in
    x,y,z; 4,5,6 = rotations. We drop rotational DOFs (solid elements
    don't have rotational DOFs) and shift to 0-based indexing.
    Example: '123' -> [0, 1, 2].
    """
    return [int(c) - 1 for c in str(components) if c in "123"]


def read_fem(filename, mode):
    # Parse the deck. mode="optistruct" enables OptiStruct-specific cards.
    model = BDF(mode=mode)
    model.read_bdf(filename)

    # --- Nodes ---
    # Build a stable NID -> contiguous 0-based index mapping. Sorting
    # makes the output reproducible across files where NIDs may be
    # non-contiguous or out of order. coords[i] is the position of the
    # node with NID nid_list[i].
    nid_list = sorted(model.nodes.keys())
    nid_to_idx = {nid: i for i, nid in enumerate(nid_list)}

    coords = np.array(
        [model.nodes[nid].get_position() for nid in nid_list],
        dtype=np.float64,
    )

    # --- Elements ---
    # Walk all elements, keep only the ones whose type we support, and
    # enforce a single cell type for the whole mesh (DOLFINx meshes
    # cannot mix tets and hexes). For each kept element we store its
    # connectivity (using the new 0-based indices) and the material ID
    # of its property.
    cells = []
    cell_mat_ids = []
    cell_type_key = None  # locks in the cell type on the first valid element

    for eid in sorted(model.elements.keys()):
        elem = model.elements[eid]
        nids = elem.node_ids
        key = f"{elem.type}{len(nids)}"

        # Skip element types we don't handle (bars, rigid links, shells, ...).
        if key not in CELL_MAP:
            continue

        # Lock in the first supported cell type; reject any later element
        # that doesn't match it.
        if cell_type_key is None:
            cell_type_key = key
        elif key != cell_type_key:
            print(f"skipping {key} element {eid}, only {cell_type_key} supported")
            continue

        # Keep only corner vertices for the geometry (see CELL_MAP comment).
        n_verts = CELL_MAP[key][1]
        cells.append([nid_to_idx[n] for n in nids[:n_verts]])
        cell_mat_ids.append(model.properties[elem.pid].mid)

    if cell_type_key is None:
        raise ValueError(f"No supported elements found in {filename}")

    cells = np.array(cells, dtype=np.int64)
    cell_mat_ids = np.array(cell_mat_ids, dtype=np.int32)

    # --- Materials ---
    # Flatten each MAT card to the three properties we need downstream:
    # Young's modulus, Poisson's ratio, density. Keyed by Nastran MID.
    materials = {}
    for mid, m in model.materials.items():
        materials[mid] = {"E": m.e, "nu": m.nu, "rho": m.rho}

    # --- SPCs (single-point constraints, i.e. fixed DOFs) ---
    # Output is a list of (node_index, dof) pairs with 0-based indices.
    # SPC1 = same DOFs applied to many nodes.
    # SPC  = per-node DOF spec, may include enforced displacements
    #        (we ignore the enforced value and treat them as zero BCs).
    spcs = []
    for spc_list in model.spcs.values():
        for card in flatten(spc_list):
            if card.type == "SPC1":
                dofs = spc_dofs(card.components)
                for nid in card.node_ids:
                    if nid in nid_to_idx:
                        idx = nid_to_idx[nid]
                        for d in dofs:
                            spcs.append((idx, d))
            elif card.type == "SPC":
                for nid, comp, _ in zip(card.node_ids, card.components, card.enforced):
                    if nid in nid_to_idx:
                        idx = nid_to_idx[nid]
                        for d in spc_dofs(comp):
                            spcs.append((idx, d))

    # --- Nodal forces ---
    # Output is a list of (node_index, fx, fy, fz). scaled_vector already
    # includes the FORCE card's magnitude, so no extra multiplication
    # is needed. Distributed loads (PLOAD*, GRAV, ...) are ignored.
    forces = []
    for load_list in model.loads.values():
        for card in flatten(load_list):
            if card.type == "FORCE" and card.node_id in nid_to_idx:
                xyz = card.scaled_vector
                forces.append((nid_to_idx[card.node_id], xyz[0], xyz[1], xyz[2]))

    # --- Pack ---
    # element_order drives the function space order on the DOLFINx side
    # (1 for linear elements, 2 for quadratic).
    basix_cell, _, element_order = CELL_MAP[cell_type_key]

    return {
        "coords": coords,
        "cells": cells,
        "cell_type": basix_cell,
        "element_order": element_order,
        "cell_mat_ids": cell_mat_ids,
        "materials": materials,
        "spcs": spcs,
        "forces": forces,
        "nid_to_idx": nid_to_idx,
    }


if __name__ == "__main__":
    read_fem("beam.fem", mode="optistruct")