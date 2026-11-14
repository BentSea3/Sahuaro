# Sahuaro: CIF to LAMMPS Conversion Tool

## 1. Introduction

**Sahuaro** is a command-line utility that converts a CIF crystal structure and a simple Sahuaro input file into LAMMPS-compatible `in.data` (structure) and `in.coeffs` (force field parameter) files. It is designed for quick setup of metal–organic frameworks, zeolites, and other periodic materials in LAMMPS.

## 2. Installation

Sahuaro requires Python 3.9 or later and NumPy.

Clone the repository and create a Python virtual environment:

```bash
git clone https://github.com/BentSea3/Sahuaro.git
cd Sahuaro
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

To install the testing dependencies and run the tests:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

## 3. Usage

From the repository directory, run:

```bash
python sahuaro/Sahuaro.py path/to/sahuaro_input --output-dir out
```

- **`sahuaro_input`**: input file with sections `PATHS:`, `ATOMS:`, `BONDS:`, etc.
- **`--log`**: write progress messages to `sahuaro.log` instead of the terminal.
- **`--output-dir`**: directory to place `in.data` and `in.coeffs` (defaults to the current directory).

The output directory is created if it does not exist. Use `--help` to see the available arguments:

```bash
python sahuaro/Sahuaro.py --help
```

## 4. Input File Format (`sahuaro_input`)

1. **`PATHS:`**
   - `CIF = <path>`
     - Relative paths are resolved against the directory containing your sahuaro_input file.
2. **`ATOMS:`**, **`BONDS:`**, **`ANGLES:`**, **`PDIHEDRALS:`**, **`IDIHEDRALS:`**
   - Tables with header row listing column names, then one entry per line.
3. **`CONNECTIVITY_TYPES:`**
   - e.g. `BOND_STYLE: harmonic`
     - Keys correspond to LAMMPS style names (see registry).
4. **`CONSIDERATIONS:`**
   Set processing flags (case-insensitive, values lowercased). Supported keys:

   | Key | Values | Description |
   |---|---|---|
   | `ONLY_ATOMS` | `yes` / `no` | If `yes`, omit bonds/angles/... tables in `in.data`. |
   | `INCLUDE_TILT` | `yes` / `no` | If `yes`, include triclinic tilt factors in header. |
   | `CHARGES_FROM_CIF` | `yes` / `no` | If `yes`, override atom charges with CIF values. |
   | `Q_FROM_CHARGETRANSFER` | `yes` / `no` | If `yes`, read `charge_transfer.dat` and compute deltas. |
   | `REPLICATE` | `<nx> <ny> <nz>` | Supercell replication factors. |
   | `FILL_PARAMETERS` | `yes` / `no` | If `yes`, auto-fill `in.coeffs` using style registry. |
   | `WRITE_LOG` | `yes` / `no` | (Reserved) Write extra log info. |
   | `IMPROPER_CENTRAL` | `first` / `last` | Placement of central atom in improper dihedral. |
   | `KELVIN_TO_KCALMOL` | `yes` / `no` | Convert temperature constants to kcal/mol. |
   | `KJ_TO_KCALMOL` | `yes` / `no` | Convert kJ/mol constants to kcal/mol. |
   | `FACTOR_0.5_HARMONIC` | `yes` / `no` | Halve energy constants for harmonic bonds/angles. |
   | `FACTOR_E_DIH_IMP` | `yes <factor>` / `no` | Multiply dihedral/improper energies by `<factor>`. |

## 5. Output

- **`in.data`**
  - Default `atom_style full`
  - Contains box dimensions, masses, atoms, (optional) bonds, angles, dihedrals, impropers.
- **`in.coeffs`**
  - If `FILL_PARAMETERS = no`, you get a template with only `pair_coeff`, `bond_coeff`, etc., for manual editing.
  - If `yes`, parameters are pulled from `sahuaro_input` tables and validated against `connectivity_registry.py`.

## 6. Limitations

1. **Symmetry**: Only supports CIFs in **P1** symmetry with the full atom list provided. The parser reads atom-site data from a `loop_` table.
2. **Cell shape**: Only **orthorhombic** cells currently supported.
3. **Label matching**: Atom labels in `ATOMS:` must exactly match labels in the CIF.
4. **Connectivity completeness**: If `ONLY_ATOMS = no`, you **must** specify *all* bonds (not just force-field ones) for correct neighbor lists.
5. **Replication cost**: Replicating beyond `4 4 4` for flexible FFs may be very slow in connectivity detection.
6. **Registry coverage**: Only a subset of LAMMPS styles are in `connectivity_registry.py`; more will be added later.
7. **Parameter filling caution**:
   - It is **recommended** to use `FILL_PARAMETERS = no` to generate a template, then manually edit.
   - If `yes`, *the order* of columns in your tables **must** match the order in LAMMPS docs (e.g. for `bond_style harmonic`: `K` then `r0`).
8. **No hybrid styles**: Hybrid pair/dihedral/etc. styles are not yet supported.

## 7. Example sahuaro_input contents

The following is a shortened example. See the `Examples/` directory for complete input files.

```text
PATHS:
  CIF = UiO66_Pristine.cif

ATOMS:
  label type charge radius mass
  C1    1    -0.5   0.77   12.011
  ...

BONDS:
  a1 a2 K r0
  C1 C2 450   1.39
  ...

CONSIDERATIONS:
  ONLY_ATOMS       = no
  INCLUDE_TILT     = yes
  REPLICATE        = 2 2 1
  FILL_PARAMETERS  = no
  IMPROPER_CENTRAL = first
```

## 8. Implemented LAMMPS interaction styles

### Pair styles (`PAIR_STYLE`)

| Style | Required | Optional |
|---|---|---|
| **lj/cut** | epsilon, sigma | cutoff_lj |
| **lj/cut/coul/cut** | epsilon, sigma | cutoff_lj, cutoff_coul |
| **lj/cut/coul/long** | epsilon, sigma | cutoff_lj, cutoff_coul |
| **lj/cut/coul/debye** | epsilon, sigma | cutoff_lj, cutoff_coul |
| **lj/cut/coul/dsf** | epsilon, sigma | cutoff_lj, cutoff_coul |
| **lj/cut/coul/msm** | epsilon, sigma | cutoff_lj, cutoff_coul |
| **lj/cut/coul/wolf** | epsilon, sigma | cutoff_lj, cutoff_coul |
| **lj/cut/tip4p/cut** | epsilon, sigma | cutoff_lj |
| **lj/cut/tip4p/long** | epsilon, sigma | cutoff_lj |
| **lj/switch3/coulgauss/long** | epsilon, sigma, gaussian_width | — |
| **mm3/switch3/coulgauss/long** | epsilon, sigma, gaussian_width | — |

### Bond styles (`BOND_STYLE`)

| Style | Required | Optional |
|---|---|---|
| **harmonic** | K, r0 | — |
| **morse** | D0, alpha, r0 | — |
| **mm3** | K, r0 | — |
| **class2** | r0, K2, K3, K4 | — |
| **none** | — | — |
| **zero** | bdist | — |

### Angle styles (`ANGLE_STYLE`)

| Style | Required | Optional |
|---|---|---|
| **harmonic** | K, theta0 | — |
| **charmm** | K, theta0, K_ub, r_ub | — |
| **fourier** | K, C0, C1, C2 | — |
| **cosine** | K | — |
| **none** | — | — |

### Dihedral styles (`DIHEDRAL_STYLE`)

| Style | Required | Optional |
|---|---|---|
| **harmonic** | K, d, n | — |
| **charmm** | K, n, d, wfac | — |
| **charmmfsw** | K, n, d, wfac | — |
| **opls** | K1, K2, K3, K4 | — |
| **quadratic** | K, theta0 | — |
| **none** | — | — |

### Improper styles (`IMPROPER_STYLE`)

| Style | Required | Optional |
|---|---|---|
| **harmonic** | K, chi0 | — |
| **cossq** | K, chi0 | — |
| **cvff** | K, d, n | — |
| **distharm** | K, d0 | — |
| **fourier** | K, C0, C1, C2 | all_parm |
| **inversion/harmonic** | K, omega0 | — |
| **sqdistharm** | K, d0sq | — |

## 9. Contact & Future Work

For bugs or feature requests, please contact me directly. Future enhancements include: support for non-P1 CIFs, triclinic cells, complete LAMMPS style coverage, and hybrid styles.
