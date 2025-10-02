
# Sahuaro: CIF-to-LAMMPS Conversion Tool

## 1. Introduction

**Sahuaro** is a command-line utility that converts a CIF crystal structure and a simple Sahuaro input file into LAMMPS-compatible `in.data` (structure) and `in.coeffs` (force field) files. It is designed for quick setup of metal–organic frameworks, zeolites, and other periodic materials in LAMMPS.

## 2. Installation

1. **Python version**: Requires **Python 3.9** or later.  
2. **Dependencies**:  
   - `numpy`  
3. **Files**:  
   - `BasicSahuaro.py` (the main script)  
   - `sahuaro_utils.py` (parsing utilities)  
   - `connectivity_registry.py` (force‑field style registry)  
4. Place these three files in the same directory or ensure they are all on your `PYTHONPATH`.

## 3. Usage

```bash
chmod +x BasicSahuaro.py
./BasicSahuaro.py path/to/sahuaro_input                   [--log]                   [--output-dir OUTPUT_DIR]
```

- **`sahuaro_input`**: your text file with sections `PATHS:`, `ATOMS:`, `BONDS:`, etc.  
- **`--log`**: write progress messages to `sahuaro.log` instead of `stdout`.  
- **`--output-dir`**: directory to place `in.data` and `in.coeffs` (defaults to current folder).  

You may also invoke explicitly with Python:

```bash
python3 BasicSahuaro.py path/to/sahuaro_input --output-dir out
```

## 4. Input File Format (`sahuaro_input`)

1. **`PATHS:`**  
   - `CIF = <path>`  
     - Relative paths are resolved against the directory containing your input file.  
2. **`ATOMS:`**, **`BONDS:`**, **`ANGLES:`**, **`PDIHEDRALS:`**, **`IDIHEDRALS:`**  
   - Tables with header row listing column names, then one entry per line.  
3. **`CONNECTIVITY_TYPES:`**  
   - e.g. `BOND_STYLE: harmonic`  
     - Keys correspond to LAMMPS style names (see registry).  
4. **`CONSIDERATIONS:`**  
   Set processing flags (case-insensitive, values lowercased). Supported keys:

   | Key                      | Values                       | Description |
   |--------------------------|------------------------------|-------------|
   | `ONLY_ATOMS`             | `yes` / `no`                 | If `yes`, omit bonds/angles/... tables in `in.data`. |
   | `INCLUDE_TILT`           | `yes` / `no`                 | If `yes`, include triclinic tilt factors in header. |
   | `CHARGES_FROM_CIF`       | `yes` / `no`                 | If `yes`, override atom charges with CIF values. |
   | `Q_FROM_CHARGETRANSFER`  | `yes` / `no`                 | If `yes`, read `charge_transfer.dat` and compute deltas. |
   | `REPLICATE`              | `<nx> <ny> <nz>`             | Supercell replication factors. |
   | `FILL_PARAMETERS`        | `yes` / `no`                 | If `yes`, auto‑fill `in.coeffs` using style registry. |
   | `WRITE_LOG`              | `yes` / `no`                 | (Reserved) Write extra log info. |
   | `IMPROPER_CENTRAL`       | `first` / `last`             | Placement of central atom in improper dihedral. |
   | `KELVIN_TO_KCALMOL`      | `yes` / `no`                 | Convert temperature constants to kcal/mol. |
   | `KJ_TO_KCALMOL`          | `yes` / `no`                 | Convert kJ/mol constants to kcal/mol. |
   | `FACTOR_0.5_HARMONIC`    | `yes` / `no`                 | Halve energy constants for harmonic bonds/angles. |
   | `FACTOR_E_DIH_IMP`       | `yes <factor>` / `no`        | Multiply dihedral/improper energies by `<factor>`. |

## 5. Output

- **`in.data`**  
  - Default `atom_style full`  
  - Contains box dimensions, masses, atoms, (optional) bonds, angles, dihedrals, impropers.  
- **`in.coeffs`**  
  - If `FILL_PARAMETERS = no`, you get a template with only `pair_coeff`, `bond_coeff`, etc., for manual editing.  
  - If `yes`, parameters are pulled from `sahuaro_input` tables and validated against `connectivity_registry.py`.

## 6. Limitations

1. **Symmetry**: Only supports CIFs in **P1** symmetry (full atom list, no loops).  
2. **Cell shape**: Only **orthorhombic** cells currently supported.  
3. **Label matching**: Atom labels in `ATOMS:` must exactly match labels in the CIF.  
4. **Connectivity completeness**: If `ONLY_ATOMS = no`, you **must** specify _all_ bonds (not just force‑field ones) for correct neighbor lists.  
5. **Replication cost**: Replicating beyond `3 3 3` may be very slow in connectivity detection.  
6. **Registry coverage**: Only a subset of LAMMPS styles are in `connectivity_registry.py`; more will be added later.  
7. **Parameter filling caution**:  
   - It is **recommended** to use `FILL_PARAMETERS = no` to generate a template, then manually edit.  
   - If `yes`, _the order_ of columns in your tables **must** match the order in LAMMPS docs (e.g.\ for `bond_style harmonic`: `K` then `r0`).  
8. **No hybrid styles**: Hybrid pair/dihedral/etc. styles are not yet supported.

## 7. Example

```text
# CIF to LAMMPS input file
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

## 8. Implemented LAMMPS interation styles

# Implemented LAMMPS Styles

## Pair styles (`PAIR_STYLE`)

| Style                           | ✓ required                | ○ optional                      |
|---------------------------------|---------------------------|---------------------------------|
| **lj/cut**                      | epsilon, sigma            | cutoff_lj                       |
| **lj/cut/coul/cut**             | epsilon, sigma            | cutoff_lj, cutoff_coul          |
| **lj/cut/coul/long**            | epsilon, sigma            | cutoff_lj, cutoff_coul          |
| **lj/cut/coul/debye**           | epsilon, sigma            | cutoff_lj, cutoff_coul          |
| **lj/cut/coul/dsf**             | epsilon, sigma            | cutoff_lj, cutoff_coul          |
| **lj/cut/coul/msm**             | epsilon, sigma            | cutoff_lj, cutoff_coul          |
| **lj/cut/coul/wolf**            | epsilon, sigma            | cutoff_lj, cutoff_coul          |
| **lj/cut/tip4p/cut**            | epsilon, sigma            | cutoff_lj                       |
| **lj/cut/tip4p/long**           | epsilon, sigma            | cutoff_lj                       |
| **lj/switch3/coulgauss/long**   | epsilon, sigma, gaussian_width | —                        |
| **mm3/switch3/coulgauss/long**  | epsilon, sigma, gaussian_width | —                        |

## Bond styles (`BOND_STYLE`)

| Style      | ✓ required                             | ○ optional |
|------------|----------------------------------------|------------|
| **harmonic** | K, r0                                | —          |
| **morse**    | D0, alpha, r0                        | —          |
| **mm3**      | K, r0                                | —          |
| **class2**   | r0, K2, K3, K4                       | —          |
| **none**     | —                                   | —          |
| **zero**     | bdist                                | —          |

## Angle styles (`ANGLE_STYLE`)

| Style      | ✓ required                             | ○ optional |
|------------|----------------------------------------|------------|
| **harmonic** | K, theta0                            | —          |
| **charmm**   | K, theta0, K_ub, r_ub               | —          |
| **fourier**  | K, C0, C1, C2                       | —          |
| **cosine**   | K                                   | —          |
| **none**     | —                                   | —          |

## Dihedral styles (`DIHEDRAL_STYLE`)

| Style        | ✓ required                              | ○ optional |
|--------------|-----------------------------------------|------------|
| **harmonic**   | K, d, n                              | —          |
| **charmm**     | K, n, d, wfac                        | —          |
| **charmmfsw**  | K, n, d, wfac                        | —          |
| **opls**       | K1, K2, K3, K4                       | —          |
| **quadratic**  | K, theta0                            | —          |
| **none**       | —                                    | —          |

## Improper styles (`IMPROPER_STYLE`)

| Style                  | ✓ required                                 | ○ optional   |
|------------------------|--------------------------------------------|--------------|
| **harmonic**             | K, chi0                                  | —            |
| **cossq**                | K, chi0                                  | —            |
| **cvff**                 | K, d, n                                  | —            |
| **distharm**             | K, d0                                   | —            |
| **fourier**              | K, C0, C1, C2                           | all_parm     |
| **inversion/harmonic**   | K, omega0                               | —            |
| **sqdistharm**           | K, d0sq                                 | —            |


## 9. Contact & Future Work

For bugs or feature requests, please contact me directly. Future enhancements include: support for non‑P1 CIFs, triclinic cells, complete LAMMPS style coverage, and hybrid styles.
