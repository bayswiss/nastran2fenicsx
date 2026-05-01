# nastran2fenicsx

Read Nastran-style bulk data files (`.bdf`/`.fem`) and solve them with [DOLFINx](https://github.com/FEniCS/dolfinx) + SLEPc/PETSc.

A small, MPI-ready tool that takes a Nastran-style bulk data file (`.bdf`), turns it into a DOLFINx mesh with material, boundary-condition and load data attached, and runs the analysis with FEniCSx + SLEPc/PETSc.

The current implementation focuses on **modal analysis** of solid 3D structures (SOL 103) meshed with **CTETRA** elements (linear and quadratic). The parser is generic enough that adding new solution types is mostly a matter of writing a new driver script on top of `read_fem(...)`.

---

## What it does today

- Parses a `.bdf` solver deck via [pyNastran](https://github.com/SteveDoyle2/pyNastran).
- Builds a distributed DOLFINx mesh: linear (P1) geometry, with a P1 or P2 displacement field selected automatically from the element order.
- Assembles per-element material properties (E, ν, ρ) from `MAT1` cards as DG0 fields.
- Applies translational `SPC` / `SPC1` constraints by matching DOF coordinates against the original GRID positions.
- Solves GHEP `K φ = λ M φ` with SLEPc.

## What it does *not* do (yet)

- **Element types**: only `CTETRA4` and `CTETRA10` are tested. `CHEXA8` / `CHEXA20` paths are stubbed but **not validated**. Shells (`CQUAD4`/`CTRIA3`), beams (`CBAR`/`CBEAM`), rigids (`RBE2`/`RBE3`), and concentrated masses (`CONM2`) are still not handled.
- **Solutions**: only the modal eigenproblem is implemented. Direct and modal frequency response are on the roadmap.
- **Loads**: `FORCE` cards are read; distributed loads (`PLOAD*`, `GRAV`) and enforced non-zero displacements are silently dropped today.

---
## Prerequisites
This tool requires **FEniCSx** with complex build of PETSc. We recommend installing it via ```conda```.

**Install FEniCSx via Conda:**
    
```bash
conda create -n fenicsx-env
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich petsc=*=complex*
```

> **Why complex PETSc?** SOL 103 (modal) only needs the real build, but the complex build is selected here so the same environment will also run the frequency-response drivers (SOL 108 / 111, planned).

Alternative ways to install FEniCSx: [https://github.com/FEniCS/dolfinx?tab=readme-ov-file#installation](https://github.com/FEniCS/dolfinx?tab=readme-ov-file#installation)

pyNastran and scipy are needed also.
```bash
pip install pyNastran scipy
```

## Quickstart

Serial:

```bash
python SOL_103_MPI.py
```

Parallel:

```bash
mpirun -n 4 python SOL_103_MPI.py
```

The input filename is currently set inside SOL_103_MPI.py:

```python
data = read_fem("beam_2nd.bdf", mode="msc")
```

`mode` is the pyNastran dialect flag (`"msc"`, `"nx"`, `"optistruct"`, `"zona"`, `"autodesk"`). Pick the one that matches the deck you're feeding in.

Output:

- console: list of converged natural frequencies in Hz.
- `modes.bp`: mode shapes, time-stepped with the frequency value as the time tag, openable in ParaView.

## Units

The solver is unit-agnostic — what comes out is determined by what went in. A typical SI choice is:

| quantity         | unit    |
|------------------|---------|
| length           | m       |
| Young's modulus  | Pa      |
| density          | kg/m³   |
| force            | N       |
| frequency output | Hz      |

If you authored your deck in N–mm–t (common in automotive), use E in MPa and ρ in t/mm³ — the eigenfrequencies will still come out in Hz because the dimensionless √(λ)/2π is the same.

## Roadmap

- [ ] Direct frequency response (SOL 108)
- [ ] Modal frequency response (SOL 111)
- [ ] `CHEXA8` / `CHEXA20` validation
- [ ] Shell elements (`CQUAD4`, `CTRIA3`)
- [ ] Rigid elements (`RBE2`, `RBE3`)
- [ ] Concentrated masses (`CONM2`)
- [ ] Distributed loads (`PLOAD2`, `PLOAD4`, `GRAV`)
- [ ] Tests + CI
- [ ] Pip-installable package

Issues and PRs welcome.

## Acknowledgments

- [pyNastran](https://github.com/SteveDoyle2/pyNastran).
- [FEniCSx / DOLFINx](https://fenicsproject.org/).

## Disclaimer

This is an independent open-source project. It is **not affiliated with, endorsed by, sponsored by, or otherwise connected to** any commercial finite-element vendor. "Nastran" is used here as the common name of the publicly documented bulk-data card format that originated with NASA's NASTRAN in the 1960s and is today read by many independent codes (pyNastran, MYSTRAN, and others). All third-party product names and trademarks are property of their respective owners.

The bulk-data files in this repository (`beam_1st.bdf`, `beam_2nd.bdf`) were authored by the project author from scratch and contain only standard, publicly documented Nastran cards.