#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sahuaro.py version 1

Converts system into LAMMPS data + coeffs.

Example:
  ./sahuaro.py path/to/sahuaro_input \
      --log \
      --output-dir=out_folder
"""

# Importing the necessary libraries
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from itertools import combinations, combinations_with_replacement
from datetime import datetime

from sahuaro_utils import parse_file, read_cif_file
from connectivity_registry import STYLE_REGISTRY

def main():
    # Parse command-line arguments
    p = argparse.ArgumentParser(
        description="Convert a Sahuaro input + CIF into LAMMPS in.data + in.coeffs"
    )
    p.add_argument("config",
        help="path to your sahuaro_input file (with ATOMS, BONDS, …)")
    p.add_argument("--log", action="store_true",
        help="write a sahuaro.log instead of printing to stdout")
    p.add_argument("--output-dir", default=".",
        help="where to write in.data / in.coeffs")
    args = p.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load and validate
    data = parse_file(args.config, compat=True)
    # Set up log vs print
    if args.log:
        import logging
        log_path = Path(args.output_dir) / "sahuaro.log"
        logging.basicConfig(
            filename=str(log_path), filemode="w",
            level=logging.INFO,
            format="%(asctime)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        log = logging.info
    else:
        log = print

    # ── VALIDATE CONSIDERATIONS KEYS ─────────────────────────────────────────────
    # define all the keys your code actually supports
    allowed_cons_keys = {
        'ONLY_ATOMS',
        'INCLUDE_TILT',
        'CHARGES_FROM_CIF',
        'REPLICATE',
        'FILL_PARAMETERS',
        'WRITE_LOG',
        'Q_FROM_CHARGETRANSFER',
        'IMPROPER_CENTRAL',
        'KELVIN_TO_KCALMOL',
        'KJ_TO_KCALMOL',
        'FACTOR_0.5_HARMONIC',
        'FACTOR_E_DIH_IMP',
    }
    # find any typos / unsupported considerations
    unknown = set(data.get('CONSIDERATIONS', {})) - allowed_cons_keys
    if unknown:
        msg = f"Error: Unknown CONSIDERATIONS keys: {', '.join(sorted(unknown))}"
        log(msg)
        sys.exit(1)
    
    # Parsing the data from the input file
    
    # To determine if only atom properties are outputted without doing connectivity analysis.
    only_atoms       = data['CONSIDERATIONS'].get('ONLY_ATOMS')   == 'yes'
    # To determine if tilt should be included in the output.
    include_tilt     = data['CONSIDERATIONS'].get('INCLUDE_TILT') == 'yes'
    # To determine if charges are taken from the CIF or input file.
    charges_from_cif = data['CONSIDERATIONS'].get('CHARGES_FROM_CIF','no') == 'yes'
    
    # Defining nx,ny,nz if REPLICATE is used
    replicate_str = data['CONSIDERATIONS'].get('REPLICATE','').strip()
    if replicate_str:
        nx, ny, nz = map(int, replicate_str.split())
    else:
        nx = ny = nz = 1
    
    # To determine if parameters are automatically filled by Sahuaro using the LAMMPS style registry.
    fill_parameters = (
        data['CONSIDERATIONS']
            .get('FILL_PARAMETERS', 'no')
            .strip()
            .lower() == 'yes'
    )
    
    # Override charges from a charge-transfer table?
    q_from_ct = data['CONSIDERATIONS'] \
                    .get('Q_FROM_CHARGETRANSFER','no') \
                    .strip().lower() == 'yes'
    
    log(f"Parsed input file: {data['PATHS']['CIF']}")
    log(f"  ONLY_ATOMS= {only_atoms}, FILL_PARAMETERS= {fill_parameters}, WRITE_LOG= {args.log}")
    log(f"  REPLICATE = {nx} × {ny} × {nz}, CHARGES_FROM_CIF= {charges_from_cif}, INCLUDE_TILT= {include_tilt}")
    
    # Checking conversion flags
    kelvin_to_kcal   = data.get('CONSIDERATIONS', {}).get('KELVIN_TO_KCALMOL','no') == 'yes'
    kj_to_kcal       = data.get('CONSIDERATIONS', {}).get('KJ_TO_KCALMOL','no') == 'yes'
    half_harmonic    = data.get('CONSIDERATIONS', {}).get('FACTOR_0.5_HARMONIC','no') == 'yes'
    
    # Parse FACTOR_E_DIH_IMP
    raw = data['CONSIDERATIONS'] \
            .get('FACTOR_E_DIH_IMP','no') \
            .strip().split()
    if raw[0].lower() == 'yes':
        if len(raw) != 2:
            log("Error: FACTOR_E_DIH_IMP must be “yes <number>”")
            sys.exit(1)
        factor_e_dih_imp = float(raw[1])
    else:
        factor_e_dih_imp = 1.0
    
    log(f"  FACTOR_E_DIH_IMP = {factor_e_dih_imp}")
    
    # Get name from CIF path, date, and output paths.
    flname       = os.path.basename(data['PATHS']['CIF']).split('.')[0]
    today        = datetime.today().strftime('%d-%m-%Y')
    odata_path   = Path(args.output_dir) / 'in.data'
    ocoeffs_path = Path(args.output_dir) / 'in.coeffs'
    
    cfg_dir = Path(args.config).parent
    cif_rel = Path(data['PATHS']['CIF'])
    cif_path = cfg_dir / cif_rel if not cif_rel.is_absolute() else cif_rel
    data['PATHS']['CIF'] = str(cif_path)

    # Parse CIF and PADEF (Pseudo Atom DEFinitions).
    cif_data   = read_cif_file(data['PATHS']['CIF'])
    padef_data = data['PADEF']
    
    log(f"Read CIF: {len(cif_data['_atoms'])} sites, cell = {cif_data['cell']['length']}")
    log(f"Loaded {len(padef_data)} atom types from definitions")
    
    # Always build FRAMEWORK so fw_data is defined even in only_atoms is true.
    fw_data = data.get('FRAMEWORK', {})
    
    # Check for missing bond connectivity when ONLY_ATOMS is False
    if not only_atoms and len(data.get("BONDS", [])) == 0:
        log("ONLY_ATOMS is set to 'no', but no BONDS were provided.")
        log("Please add bond connectivity information to the input file")
        raise ValueError("Missing required bond connectivity.")
    
    # Error out if the user accidentally turned both conversions
    if kelvin_to_kcal and kj_to_kcal:
        msg = "Cannot use both KELVIN_TO_KCALMOL and KJ_TO_KCALMOL at the same time."
        if args.log:
            log(msg)
            sys.exit(1)
        else:
            raise ValueError(msg)
    
    if charges_from_cif and q_from_ct:
        log("Cannot use CHARGES_FROM_CIF and Q_FROM_CHARGETRANSFER simultaneously.")
        raise ValueError("Cannot use CHARGES_FROM_CIF and Q_FROM_CHARGETRANSFER simultaneously.")
    
    # Validate connectivity & pair‐coeff tables if FILL_PARAMETERS = yes
    
    if fill_parameters:
        log("Validating input coefficients with the style registry . . .")
        from connectivity_registry import STYLE_REGISTRY
    
        ct = data['CONNECTIVITY_TYPES']
        
        ct = {
        key.strip().upper(): val.strip().lower() 
        for key, val in data.get('CONNECTIVITY_TYPES', {}).items()
        }
    
        def validate_section(rows, style_key, base_cols, registry_key, section_name):
            """
            rows:      list of dictionaries (e.g. data['BONDS'])
            style_key: e.g. 'BOND_STYLE'
            base_cols: list of column names BEFORE the parameters (e.g. ['a1','a2'])
            registry_key: one of 'bond_style','angle_style', etc. matching STYLE_REGISTRY
            section_name: pretty name for error messages
            """
            # skip entirely if there are no rows
            if not rows:
                return
    
            style = ct.get(style_key.upper())
            # if not specified, default to 'none'
            if style is None:
                style = 'none'
    
            # warn if user set a non-none style but forgot the section
            if style != 'none' and not rows:
                log(f"Warning: {section_name}_STYLE='{style}' but no {section_name.lower()} rows found.")
                return
    
            # skip entirely when style is 'none'
            if style == 'none':
                return
    
            styles_map = STYLE_REGISTRY[registry_key]
            spec = styles_map.get(style)
            if spec is None:
                # raise ValueError(f"Unknown {registry_key} '{style}'. "
                #                  f"Must be one of {list(styles_map)}")
                valid = ", ".join(sorted(styles_map))
                raise ValueError(
                    f"Unknown {registry_key!r} '{style}'.\n"
                    f"Supported values are: {valid}"
                )
    
            req = spec['required']
            opt = spec.get('optional', [])
            n_req = len(req)
            n_max = n_req + len(opt)
    
            for i, row in enumerate(rows, start=1):
                # discover extra keys beyond the base_cols
                extras = [k for k in row if k not in base_cols]
                if not (n_req <= len(extras) <= n_max):
                    raise ValueError(
                        f"{section_name} row {i}: for style '{style}', "
                        f"expected {n_req}–{n_max} params but found {len(extras)}"
                    )
                # check required ones first
                for (param_name, ptype, cond, msg), col in zip(req, extras):
                    val = row[col]
                    if not isinstance(val, ptype):
                        raise ValueError(
                            f"{section_name} row {i} '{col}': expected {ptype.__name__}, got {type(val).__name__}"
                        )
                    if not cond(val):
                        raise ValueError(
                            f"{section_name} row {i} '{col}': value {val} fails constraint ({msg})"
                        )
                # then optional ones
                for (param_name, ptype, cond, msg), col in zip(opt, extras[n_req:]):
                    val = row[col]
                    if not isinstance(val, ptype):
                        raise ValueError(
                            f"{section_name} row {i} '{col}': expected {ptype.__name__}, got {type(val).__name__}"
                        )
                    if not cond(val):
                        raise ValueError(
                            f"{section_name} row {i} '{col}': value {val} fails constraint ({msg})"
                        )
    
        # run all five validations, collecting any error messages
        errors = []
        for rows, style_key, base_cols, registry_key, section_name in [
            (data['BONDS'],      'BOND_STYLE',      ['a1','a2'],            'bond_style',     'Bond'),
            (data['ANGLES'],     'ANGLE_STYLE',     ['a1','a2','a3'],       'angle_style',    'Angle'),
            (data['PDIHEDRALS'], 'DIHEDRAL_STYLE',  ['a1','a2','a3','a4'],  'dihedral_style', 'Dihedral'),
            (data['IDIHEDRALS'], 'IMPROPER_STYLE',  ['a1','a2','a3','a4'],  'improper_style', 'Improper'),
            (data['ATOMS'],      'PAIR_STYLE',      ['label','type','charge','radius','mass'], 'pair_style', 'Pair'),
        ]:
            try:
                validate_section(rows, style_key, base_cols, registry_key, section_name)
            except ValueError as e:
                errors.append(str(e))
    
        if errors:
            raise ValueError(
                "Connectivity/style validation failed with the following problems:\n  - "
                + "\n  - ".join(errors)
            )
    
        log("  All connectivity and pair parameters congruent with LAMMPS style template")
    
    def write_placeholder(out, data, labels):
        out.write("# Complete with appropriate parameters\n\n")
    
        # if only atom‐only mode, write *only* self‐pairs
        if only_atoms:
            out.write("# Pair coefficients (self pairs)\n")
            for i, lab in enumerate(labels, start=1):
                out.write(f"pair_coeff    {i:<3} {i:<3}   # is {lab}-{lab}\n")
            out.write("\n")
            return
    
        # --- Bonds ---
        if data.get('BONDS'):
            out.write("# Bond coefficients\n")
            for i, row in enumerate(data['BONDS'], start=1):
                combo = f"{row['a1']}-{row['a2']}"
                out.write(f"bond_coeff    {i:<3}   # is {combo}\n")
            out.write("\n")
    
        # --- Angles ---
        if data.get('ANGLES'):
            out.write("# Angle coefficients\n")
            for i, row in enumerate(data['ANGLES'], start=1):
                combo = f"{row['a1']}-{row['a2']}-{row['a3']}"
                out.write(f"angle_coeff   {i:<3}   # is {combo}\n")
            out.write("\n")
    
        # --- Dihedrals ---
        if data.get('PDIHEDRALS'):
            out.write("# Dihedral coefficients\n")
            for i, row in enumerate(data['PDIHEDRALS'], start=1):
                combo = f"{row['a1']}-{row['a2']}-{row['a3']}-{row['a4']}"
                out.write(f"dihedral_coeff {i:<3}   # is {combo}\n")
            out.write("\n")
    
        # --- Impropers ---
        if data.get('IDIHEDRALS'):
            out.write("# Improper coefficients\n")
            for i, row in enumerate(data['IDIHEDRALS'], start=1):
                combo = f"{row['a1']}-{row['a2']}-{row['a3']}-{row['a4']}"
                out.write(f"improper_coeff {i:<3}   # is {combo}\n")
            out.write("\n")
    
        # --- Self-pairs ---
        out.write("# Pair coefficients (self pairs)\n")
        for i, lab in enumerate(labels, start=1):
            out.write(f"pair_coeff    {i:<3} {i:<3}   # is {lab}-{lab}\n")
        out.write("\n")
    
    def write_full_parameters(out, data, padef_data, labels):
        """
        Write a fully-populated in.coeffs file from data['BONDS'], data['ANGLES'], etc.,
        converting any “energy”-typed params from K → kcal/mol if requested, optionally
        halving harmonic bond/angle energies, and pulls pair_coefs from data['ATOMS'].
        """
        from connectivity_registry import STYLE_REGISTRY
    
        factor_kcal      = 0.0019872041 if kelvin_to_kcal else 1.0
        factor_kj        = 0.2390057 if kj_to_kcal else 1.0
                
        # helper to build convert mask from a style spec
        def make_mask(spec):
            # spec is a StyleDef with 'required' + optional lists. These correspond to the columns after 'combo'
            all_params = spec['required'] + spec.get('optional', [])
            return ['energy' in desc for (_, _, _, desc) in all_params]
    
        ct = data.get('CONNECTIVITY_TYPES', {})
    
        # --- Bonds ---
        bond_style = ct.get('BOND_STYLE', 'none')
        bond_spec  = STYLE_REGISTRY['bond_style'].get(bond_style, {'required':[], 'optional':[]})    
        # — build list of expected types for each param
        ptypes = [ptype
                  for (_,ptype,_,_) in
                  (bond_spec['required'] + bond_spec.get('optional',[]))]    
        bond_mask  = make_mask(bond_spec)
        if data.get('BONDS'):
            out.write("# Bond coefficients\n")
            for i, row in enumerate(data['BONDS'], start=1):
                a1,a2 = row['a1'], row['a2']
                params = [row[k] for k in row if k not in ('a1','a2')]
    
                # K to kcal/mol, using factor
                if kelvin_to_kcal:
                    for idx, do_convert in enumerate(bond_mask):
                        if do_convert and idx < len(params):
                            params[idx] *= factor_kcal
    
                # kJ to kcal/mol, using factor_kj
                if kj_to_kcal:
                    for idx, is_energy in enumerate(bond_mask):
                        if is_energy and idx < len(params):
                            params[idx] *= factor_kj
                            
                # half‐harmonic: only for harmonic bonds, apply extra 0.5 factor on energy entries
                if half_harmonic and bond_style == 'harmonic':
                    for idx, is_energy in enumerate(bond_mask):
                        if is_energy and idx < len(params):
                            params[idx] *= 0.5
                # format each p according to its type
                formatted = []
                for ptype, p in zip(ptypes, params):
                    if ptype is int:
                        formatted.append(f"{int(p):<6d}")
                    else:
                        formatted.append(f"{p:<12.6f}")
                out.write(
                    f"bond_coeff    {i:<3} " +
                    "  ".join(formatted) +
                    f"   # {a1}-{a2}\n"
                )
    
            out.write("\n")
    
        # --- Angles ---
        angle_style = ct.get('ANGLE_STYLE', 'none')
        angle_spec  = STYLE_REGISTRY['angle_style'].get(angle_style, {'required':[], 'optional':[]})
        # — build list of expected types for each param
        ptypes = [ptype
                  for (_,ptype,_,_) in
                  (angle_spec['required'] + angle_spec.get('optional',[]))]
        angle_mask  = make_mask(angle_spec)
        if data.get('ANGLES'):
            out.write("# Angle coefficients\n")
            for i, row in enumerate(data['ANGLES'], start=1):
                a1,a2,a3 = row['a1'], row['a2'], row['a3']
                params = [row[k] for k in row if k not in ('a1','a2','a3')]
    
                # K to kcal/mol using Boltzmann constant
                if kelvin_to_kcal:
                    for idx, do_convert in enumerate(angle_mask):
                        if do_convert and idx < len(params):
                            params[idx] *= factor_kcal
                
                # kJ to kcal/mol, using factor_kj
                if kj_to_kcal:
                    for idx, is_energy in enumerate(angle_mask):
                        if is_energy and idx < len(params):
                            params[idx] *= factor_kj
                            
                # half‐harmonic: only for harmonic angles, apply 0.5 factor on energy entries
                if half_harmonic and angle_style == 'harmonic':
                    for idx, is_energy in enumerate(angle_mask):
                        if is_energy and idx < len(params):
                            params[idx] *= 0.5
    
                # format each p according to its type
                formatted = []
                for ptype, p in zip(ptypes, params):
                    if ptype is int:
                        formatted.append(f"{int(p):<6d}")
                    else:
                        formatted.append(f"{p:<12.6f}")
                out.write(
                    f"angle_coeff   {i:<3} " +
                    "  ".join(formatted) +
                    f"   # {a1}-{a2}-{a3}\n"
                )
    
            out.write("\n")
    
        # --- Dihedrals ---
        dih_style = ct.get('DIHEDRAL_STYLE', 'none')
        dih_spec  = STYLE_REGISTRY['dihedral_style'].get(dih_style, {'required':[], 'optional':[]})
        # — build list of expected types for each param
        ptypes = [ptype
                for (_,ptype,_,_) in
                   (dih_spec['required'] + dih_spec.get('optional',[]))]
        dih_mask  = make_mask(dih_spec)
        if data.get('PDIHEDRALS'):
            out.write("# Dihedral coefficients\n")
            for i, row in enumerate(data['PDIHEDRALS'], start=1):
                a1,a2,a3,a4 = row['a1'], row['a2'], row['a3'], row['a4']
                params = [row[k] for k in row if k not in ('a1','a2','a3','a4')]
    
                if kelvin_to_kcal:
                    for idx, do_convert in enumerate(dih_mask):
                        if do_convert and idx < len(params):
                            params[idx] *= factor_kcal
                            
                # kJ to kcal/mol, using factor_kj
                if kj_to_kcal:
                    for idx, is_energy in enumerate(dih_mask):
                        if is_energy and idx < len(params):
                            params[idx] *= factor_kj
    
                if factor_e_dih_imp != 1.0:
                    for idx, is_e in enumerate(dih_mask):
                        if is_e and idx < len(params):
                            params[idx] *= factor_e_dih_imp
                            
                # format each p according to its type
                formatted = []
                for ptype, p in zip(ptypes, params):
                    if ptype is int:
                        formatted.append(f"{int(p):<6d}")
                    else:
                        formatted.append(f"{p:<12.6f}")
                out.write(
                    f"dihedral_coeff {i:<3} " +
                    "  ".join(formatted) +
                    f"   # {a1}-{a2}-{a3}-{a4}\n"
                )
    
            out.write("\n")
    
        # --- Impropers ---
        imp_style = ct.get('IMPROPER_STYLE', 'none')
        imp_spec  = STYLE_REGISTRY['improper_style'].get(imp_style, {'required':[], 'optional':[]})
        # — build list of expected types for each param
        ptypes = [ptype
                 for (_,ptype,_,_) in
                    (imp_spec['required'] + imp_spec.get('optional',[]))]
        imp_mask  = make_mask(imp_spec)
        if data.get('IDIHEDRALS'):
            out.write("# Improper coefficients\n")
            for i, row in enumerate(data['IDIHEDRALS'], start=1):
                a1,a2,a3,a4 = row['a1'], row['a2'], row['a3'], row['a4']
                params = [row[k] for k in row if k not in ('a1','a2','a3','a4')]
    
                if kelvin_to_kcal:
                    for idx, do_convert in enumerate(imp_mask):
                        if do_convert and idx < len(params):
                            params[idx] *= factor_kcal
                # kJ to kcal/mol, using factor_kj
                if kj_to_kcal:
                    for idx, is_energy in enumerate(imp_mask):
                        if is_energy and idx < len(params):
                            params[idx] *= factor_kj
    
                if factor_e_dih_imp != 1.0:
                    for idx, is_e in enumerate(imp_mask):
                        if is_e and idx < len(params):
                            params[idx] *= factor_e_dih_imp
                            
                # format each p according to its type
                formatted = []
                for ptype, p in zip(ptypes, params):
                    if ptype is int:
                        formatted.append(f"{int(p):<6d}")
                    else:
                        formatted.append(f"{p:<12.6f}")
                out.write(
                    f"improper_coeff {i:<3} " +
                    "  ".join(formatted) +
                    f"   # {a1}-{a2}-{a3}-{a4}\n"
                )
    
            out.write("\n")
    
        # --- Self-pairs ---
        out.write("# Pair coefficients (self pairs)\n")
        pair_style = ct.get('PAIR_STYLE', 'none')
        pair_spec  = STYLE_REGISTRY['pair_style'].get(pair_style, {'required':[], 'optional':[]})
        pair_mask  = make_mask(pair_spec)
        atom_rows  = data.get('ATOMS', [])
        if atom_rows:
            atom_map = {row['label']: row for row in atom_rows}
            core     = {'label','type','charge','radius','mass'}
            extras = [k for k in atom_rows[0] if k not in core]
            for i, lab in enumerate(labels, start=1):
                row    = atom_map[lab]
                # collect **all** extras
                params = [float(row.get(col, 0.0)) for col in extras]
    
                # apply any energy conversions
                if kelvin_to_kcal:
                    for idx, do_convert in enumerate(pair_mask):
                        if do_convert and idx < len(params):
                            params[idx] *= factor_kcal
                if kj_to_kcal:
                    for idx, is_energy in enumerate(pair_mask):
                        if is_energy   and idx < len(params):
                            params[idx] *= factor_kj
    
                # now format every param, not just the first two
                param_str = "  ".join(f"{p:<12.6f}" for p in params)
                out.write(
                    f"pair_coeff    {i:<3} {i:<3}  "
                    f"{param_str}   # {lab}-{lab}\n"
                )
        out.write("\n")
    
    # Organizing pseudoatom labels, positions, supercell & net‐charge check
    
    # 1) Label maps
    # labels is a list that contains the string label of each pseudoatom.
    # label_k is a dictionary that associates a numeric id value to each string label.
    labels     = list(padef_data.keys())
    label_k    = {lab:i for i,lab in enumerate(labels)}
    
    # Redundant, fix later
    pa_labels  = labels
    atom_types = len(labels)
    
    # 2) Base CIF pseudoatom sites
    pa_number  = len(cif_data['_atoms'])
    idx        = np.arange(pa_number)   # original site indices
    pal        = np.array([
        label_k[row['label']]
        for row in cif_data['_atoms']
    ], dtype=int)
    
    # 3) Unit‐cell box & atom positions
    xhi = cif_data['cell']['length']['a']
    yhi = cif_data['cell']['length']['b']
    zhi = cif_data['cell']['length']['c']
    xlo = ylo = zlo = 0.0 
    box = np.array([xhi, yhi, zhi])
    
    pos = np.array([
        [row['x']*xhi, row['y']*yhi, row['z']*zhi]
        for row in cif_data['_atoms']
    ])
    
    # 4) cell replication (nx,ny,nz from CONSIDERATIONS['REPLICATE'])
    if (nx, ny, nz) != (1,1,1):
        base_pos = pos.copy()
        base_pal = pal.copy()
        base_idx = idx.copy()
        new_pos = []; new_pal = []; new_idx = []
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    shift = np.array([ix*xhi, iy*yhi, iz*zhi])
                    for j in range(len(base_pos)):
                        new_pos.append(base_pos[j] + shift)
                        new_pal.append(base_pal[j])
                        new_idx.append(base_idx[j])
        pos = np.vstack(new_pos)
        pal = np.array(new_pal, dtype=int)
        idx = np.array(new_idx, dtype=int)
        xhi *= nx; yhi *= ny; zhi *= nz
        box = np.array([xhi, yhi, zhi])
    
    if (nx,ny,nz)!=(1,1,1):
        log(f"{len(pos)} sites. New cell = {xhi:.2f}×{yhi:.2f}×{zhi:.2f} Å ({nx}×{ny}×{nz})")
    
    # 5) Update atom count
    N = len(pos)
    
    # 6) Build charge arrays
    pcharges_base = np.array([padef_data[lab]['charge'] for lab in labels])
    qcif_base     = np.array([float(atom.get('charge',0.0))
                              for atom in cif_data['_atoms']])
    
    # 7) Select “static” charges and check neutrality (skip if using charge-transfer)
    if not q_from_ct:
        charges    = qcif_base[idx] if charges_from_cif else pcharges_base[pal]
        net_charge = charges.sum()
        log(f"Net system charge: {net_charge:.6f}")
        if abs(net_charge) > 1e-6:
            log("Warning: system is not neutral. Check input charges.")
    else:
        # will be filled below by the charge-transfer algorithm
        charges = None
    
    # 8) Box‐size warnings based on using a pair interaction cutoff of 12A.
    for dim, L in zip(('x','y','z'), box):
        if L < 24.0:
            log(f"{dim}-box = {L:.2f} Å (<24 Å). "
                  "Consider using REPLICATE.")
    
    if only_atoms:
        # 1) Atom‐only data file
        def write_atom_only(out):
            out.write(f"{flname} LAMMPS input file. Created on {today} using Sahuaro.\n\n")
            out.write(f"{N} atoms\n\n")
            out.write(f"{len(labels)} atom types\n\n")
            out.write(f"{xlo} {xhi} xlo xhi\n")
            out.write(f"{ylo} {yhi} ylo yhi\n")
            out.write(f"{zlo} {zhi} zlo zhi\n\n")
            if include_tilt:
                out.write("0.000000 0.000000 0.000000 xy xz yz\n\n")
            out.write("Masses\n\n")
            for i, label in enumerate(labels, start=1):
                m = padef_data[label]['mass']
                out.write(f"{i:<5}{m:<12.6f} # {label}\n")
            out.write("\nAtoms\n\n")
            for i in range(N):
                id_     = i+1
                molid   = 0
                labelid = pal[i]+1
                q       = charges[i]
                x, y, z = pos[i]
                out.write(f"{id_:<5}{molid:<3}{labelid:<3}{q:<10.6f}"
                          f"{x:>12.6f}{y:>12.6f}{z:>12.6f}\n")
            out.write("\n")
    
        # write the data file
        log(f"Writing LAMMPS data file to {odata_path}")
        with open(odata_path, 'w') as f:
            write_atom_only(f)
    
        # write the parameter file
        log(f"Writing FF parameters to {ocoeffs_path} (fill_parameters={fill_parameters})")
        if fill_parameters:
            with open(ocoeffs_path, 'w') as out:
                write_full_parameters(out, data, padef_data, labels)
        else:
            with open(ocoeffs_path, 'w') as f:
                write_placeholder(f, data, labels)
    
        # Stop execution if  
        if args.log:
            log("ONLY_ATOMS: wrote data + parameters")
            sys.exit(0)
        else:
            sys.exit("ONLY_ATOMS: wrote data + parameters")
    
    if not only_atoms:
        # radii list parsed from the pseudo_atoms.def file.
        radii = np.array([padef_data[label]['radii'] for label in pa_labels])
        
        # Implementing RASPA's algorithm to determine the distance threshold for the bond connectivity table, it uses the covalent radii specified
        # in the pseudo_atoms.def file.
        K = atom_types
        threshold = np.zeros((K, K))
        for k1 in range(K):
            for k2 in range(K):
                threshold[k1, k2] = 0.56 + radii[k1] + radii[k2]
        
        # Obtaining the number of bond, angle, dihedral and improper potential types.
        bond_types = len(fw_data["bonds"])
        angle_types = len(fw_data["bends"])
        dihedral_types = len(fw_data["tors"])
        improper_types = len(fw_data["itors"])
    
    if not only_atoms:
        # Determine which connectivity types were provided in the input.
        has_angles = angle_types > 0
        has_dihedrals = dihedral_types > 0
        has_impropers = improper_types > 0
    
    if not only_atoms:
        # PBC‐aware distance routines
        
        def fix_diff(rs, box):
            # minimum‐image convention in each dimension
            if rs[0] > 0.5*box[0]:
                rs[0] = box[0] - rs[0]
            if rs[1] > 0.5*box[1]:
                rs[1] = box[1] - rs[1]
            if rs[2] > 0.5*box[2]:
                rs[2] = box[2] - rs[2]
            return rs
        
        def dist2(v1, v2, box):
            rs = fix_diff(np.abs(v1 - v2), box)
            return np.linalg.norm(rs)
        
        def disti(va, vb):
            # wraps in the *supercell* box
            return dist2(va, vb, box)
        
        def calc_bonds(pos, box):
            bs, ds = [], []
            for i in range(1, N):
                for j in range(i):
                    if pal[j] not in valid_bonds[pal[i]]:
                        continue
                    d = dist2(pos[i], pos[j], box)
                    if d < threshold[pal[i], pal[j]]:
                        bs.append((i, j))
                        ds.append(d)
            return bs, np.array(ds)
        
        # Building dictionary of "valid" bonds, using pseudo atom string label as index.
        label_valid_bonds = {}
        for entry in fw_data['bonds']:
            a1, a2 = entry['combo']
            label_valid_bonds.setdefault(a1, []).append(a2)
            label_valid_bonds.setdefault(a2, []).append(a1)
        
        # List of valid bonds according to the label_k dictionary.
        valid_bonds = [
            [label_k[l2] for l2 in label_valid_bonds[l1]]
            for l1 in labels
        ]
        
        log("Detecting bonds . . .")
        # find all bonds in the *supercell*
        (bonds, ds)  = calc_bonds(pos, box)
        bonds_number = len(bonds)
    
        log(f"  Found {bonds_number} bonds")
        
        # Neighbor‐list construction.
        # Function that determines the bonded neighbor of each pseudoatom.
        def calc_neighborhood(bonds):
            ns = []
            for i in range(N):
                ni = []
                for j1, j2 in bonds:
                    if i == j1:
                        ni.append(j2)
                    elif i == j2:
                        ni.append(j1)
                ns.append(ni)
            return ns
        
        con_list = calc_neighborhood(bonds)
    
    if not only_atoms:
        if q_from_ct:
            # 1) Read the table file
            ct_path = Path(data['PATHS']['CIF']).parent / 'charge_transfer.dat'
            try:
                raw = np.genfromtxt(str(ct_path), dtype=None, encoding=None)
            except OSError:
                log(f"Charge transfer table not found at {ct_path!r}")
                raise FileNotFoundError(f"Charge transfer table not found at {ct_path!r}")
            # columns: 0→donor label, 1→acceptor label, 2→deltaq
            donors = [row[0] for row in raw]
            accepts = [row[1] for row in raw]
            dq = [row[2] for row in raw]
        
            # 2) map labels→indices
            pal_ct = [
              list(map(label_k.get, donors)),
              list(map(label_k.get, accepts)),
              dq
            ]
        
            # 3) compute per-atom deltaq
            charg = []
            for i in range(len(pal)):
                q_trans = 0.0
                main     = pal[i]
                for nei in con_list[i]:
                    nbr = pal[nei]
                    for u,v,deltaq in zip(*pal_ct):
                        if   (u, v) == (main, nbr):   q_trans -= deltaq
                        elif (u, v) == (nbr, main):   q_trans += deltaq
                charg.append(q_trans)
        
            # 4) override charges array and re-check neutrality
            charges      = np.array(charg)
            net_charge_ct = charges.sum()
            log(f"Net system charge after charge-transfer: {net_charge_ct:.6f}")
            if abs(net_charge_ct) > 1e-6:
                log("Warning: system is not charge neutral. Check your charge transfer table.")
    
    if not only_atoms:    
        if has_angles:
            log("Detecting angles . . .")
            valid_bends = [list(map(lambda label: label_k[label], entry['combo'])) for entry in fw_data['bends']]
            
            # bends connectivity list.
            triplets = []
            def bend_angle_triplets(i):
                js = con_list[i]
                itriplets = []
                for a in range(len(js)):
                    for b in range(a+1, len(js)):
                        combo = [pal[js[a]], pal[i], pal[js[b]]]
                        if combo in valid_bends:
                            itriplets.append((js[a], i, js[b]))
                        elif list(reversed(combo)) in valid_bends:
                            itriplets.append((js[b], i, js[a]))
                return itriplets
            
            for i in range(N):
                triplets.extend(bend_angle_triplets(i))
            
            # Calculating the angle of each triplet, taking into account PBCs.
            def disti(va, vb):
                rs = np.abs(va - vb)
                if rs[0] > 0.5*box[0]:
                    rs[0] = box[0]-rs[0]
                if rs[1] > 0.5*box[1]:
                    rs[1] = box[1]-rs[1]
                if rs[2] > 0.5*box[2]:
                    rs[2] = box[2]-rs[2]
                d_ab = np.sqrt(np.dot(rs, rs))
                return d_ab
            
            def bond_angle(triplet):
                a = triplet[0]
                i = triplet[1]
                b = triplet[2]
                va = pos[a]
                vi = pos[i]
                vb = pos[b]
                
                d_ia = disti(vi, va)
                d_ib = disti(vi, vb)
                d_ab = disti(va, vb)
                
                ang = np.arccos((d_ia**2 + d_ib**2 - d_ab**2)/(2*d_ia*d_ib))
                return (ang*180)/np.pi
            
            triplets_angles = [bond_angle(triplet) for triplet in triplets]
    
    if not only_atoms:
        if has_angles:
            # Regarding bends continued
            # Triplet angle comparison. If two angles involving the same pseudoatoms numerically differ more than t_deg degrees,
            # they will be clasified as different angle types in the output. Useful for complex systems/forcefields.
            def same_angle(a1, a2):
                t_deg = 2.0
                diff = abs(a1-a2)
                return min(360.0-diff, diff) < t_deg
            
            # Classifying triplets using angle.
            trip_classif_angle = {}
            triplets_angle_type = []
            
            def classif_angle(triplet, angle):
                (a, i, b) = triplet
                al = pal[a]
                il = pal[i]
                bl = pal[b]
                alab = pa_labels[al]
                ilab = pa_labels[il]
                blab = pa_labels[bl]
                tlab = (alab,ilab,blab)
                if tlab not in trip_classif_angle:
                    trip_classif_angle[tlab] = []
                for i, other_angle in enumerate(trip_classif_angle[tlab]):
                    if same_angle(angle, other_angle):
                        triplets_angle_type.append(i)
                        return
                triplets_angle_type.append(len(trip_classif_angle[tlab]))
                trip_classif_angle[tlab].append(angle)
            
            for triplet, angle in zip(triplets, triplets_angles):
                classif_angle(triplet, angle)
                    
            fw_data_bends = []
            for x in fw_data['bends']:
                [alab, ilab, blab] = x['combo']
                angles = trip_classif_angle[(alab, ilab, blab)]
                for i, angle in enumerate(angles):
                    fw_data_bends.append([alab, ilab, blab, i])
    
            log(f"  Found {len(triplets)} angle triplets, {len(fw_data_bends)} unique angle types")
    
    if not only_atoms:
        if has_dihedrals:
            log("Detecting dihedrals . . .")
            valid_tors = [list(map(lambda label: label_k[label], entry['combo'])) for entry in fw_data['tors']]
            
            quadruplets = []
            def tor_angle_quadruplets(i, j):
                ni = con_list[i]
                nj = con_list[j]
                quads = []
                for a in ni:
                    for b in nj:
                        if (a == b) or (a == j) or (b == i):
                            continue
                        combo = [pal[a], pal[i], pal[j], pal[b]]
                        if combo in valid_tors:
                            quads.append((a, i, j, b))
                        elif list(reversed(combo)) in valid_tors:
                            quads.append((b, j, i, a))
                return quads
            
            for (i, j) in bonds:
                quadruplets.extend(tor_angle_quadruplets(i, j))
    
            log(f"  Found {len(quadruplets)} dihedral quadruplets")
    
    if not only_atoms:
        if has_impropers:
            log("Detecting impropers . . .")
            # Regarding impropers
            # Building the improper dihedral connectivity list. WHERE the central atom of the improper dihedral is listed in the input
            # is important, as different LAMMPS dihedral_style implementations require different placements.
            
            valid_itors2 = [list(map(lambda label: label_k[label], entry['combo'])) for entry in fw_data['itors']]
            
            # This gets the consideration flag from the input
            central_first = data.get("CONSIDERATIONS", {}).get("IMPROPER_CENTRAL", "first") == "first"
            
            # Define neighbor_combinations based on the central atom position. If its not 'first' its assumed 'last' (most of the improper
            # dihedral styles in LAMMPS fall into either category.
            def neighbor_combinations(i):
                ni = con_list[i]
                if len(ni) < 3:
                    return []
                combs = combinations(ni, 3)
                return [[i, p[0], p[1], p[2]] if central_first else [p[0], p[1], p[2], i] for p in combs]
            
            # Define improper quadruplets based on central atom position.
            def improper_quadruplets(i):
                iquads = []
                for p in neighbor_combinations(i):
                    pal0, pal1, pal2, pal3 = pal[p[0]], pal[p[1]], pal[p[2]], pal[p[3]]
                    
                    if central_first:
                        variants = [
                            [pal0, pal1, pal2, pal3],
                            [pal0, pal1, pal3, pal2],
                            [pal0, pal2, pal1, pal3],
                            [pal0, pal2, pal3, pal1],
                            [pal0, pal3, pal1, pal2],
                            [pal0, pal3, pal2, pal1],
                        ]
                        permutations = [
                            [p[0], p[1], p[2], p[3]],
                            [p[0], p[1], p[3], p[2]],
                            [p[0], p[2], p[1], p[3]],
                            [p[0], p[2], p[3], p[1]],
                            [p[0], p[3], p[1], p[2]],
                            [p[0], p[3], p[2], p[1]],
                        ]
                    else:
                        variants = [
                            [pal0, pal1, pal2, pal3],
                            [pal0, pal2, pal1, pal3],
                            [pal1, pal2, pal0, pal3],
                            [pal1, pal0, pal2, pal3],
                            [pal2, pal1, pal0, pal3],
                            [pal2, pal0, pal1, pal3],
                        ]
                        permutations = [
                            [p[0], p[1], p[2], p[3]],
                            [p[0], p[2], p[1], p[3]],
                            [p[1], p[2], p[0], p[3]],
                            [p[1], p[0], p[2], p[3]],
                            [p[2], p[1], p[0], p[3]],
                            [p[2], p[0], p[1], p[3]],
                        ]
            
                    for variant, perm in zip(variants, permutations):
                        if variant in valid_itors2:
                            iquads.append(perm)
                            break  # Stop at first valid match
                return iquads
            
            # Construct the list of improper quadruplets
            iquadruplets = []
            for i in range(N):
                iquadruplets.extend(improper_quadruplets(i))
            
            log(f"  Found {len(iquadruplets)} improper quadruplets")
    
    if not only_atoms:
        # Writing lammps framework input file.
        def write_lammps(out):
            out.write(f"{flname} LAMMPS input file. Created on {today} using Sahuaro.\n")
            out.write('\n')
            out.write(f"{len(pos)} atoms\n")
            out.write(f"{len(bonds)} bonds\n")
            if has_angles:
                out.write(f"{len(triplets)} angles\n")
            if has_dihedrals:
                out.write(f"{len(quadruplets)} dihedrals\n")
            if has_impropers:
                out.write(f"{len(iquadruplets)} impropers\n")
            out.write('\n')
            out.write(f"{len(labels)} atom types\n")
            out.write(f"{len(fw_data['bonds'])} bond types\n")
            if has_angles:
                out.write(f"{len(fw_data_bends)} angle types\n")
            if has_dihedrals:
                out.write(f"{len(fw_data['tors'])} dihedral types\n")
            if has_impropers:
                out.write(f"{len(fw_data['itors'])} improper types\n")
            out.write('\n')
            out.write(f"{xlo} {xhi} xlo xhi\n")
            out.write(f"{ylo} {yhi} ylo yhi\n")
            out.write(f"{zlo} {zhi} zlo zhi\n")
            out.write('\n')
            if include_tilt:
                out.write(f"0.000000 0.000000 0.000000 xy xz yz\n")
                out.write('\n')
            else:
                out.write('\n')
            out.write(f"Masses\n\n")
            for i, label in enumerate(labels):
                masas = padef_data[labels[i]]['mass']
                out.write(f"{i+1}  {masas}   # {label}")
                out.write('\n')
            out.write('\n')
            bond_types = [entry['combo'] for entry in fw_data['bonds']]
            if has_angles:
                bend_types = fw_data_bends
            if has_dihedrals:
                tors_types = [entry['combo'] for entry in fw_data['tors']]
            if has_impropers:
                itors_types = [entry['combo'] for entry in fw_data['itors']]
            out.write(f"Atoms\n\n")
            for i in range(N):
                id = i+1
                molid = 0
                labelid = pal[i]+1
                q = float(charges[i])
                [x, y, z] = pos[i]
                out.write(f"{id : <5} {molid : <3} {labelid : <3} {q : <10} {x : <15.8f} {y : <15.8f} {z : <15.8f}\n")
            out.write('\n')
            out.write(f"Bonds\n\n")
            for i, b in enumerate(bonds):
                id = i+1
                combo = list(map(lambda paid: labels[pal[paid]], b))
                cid = -10000
                try:
                    cid = bond_types.index(combo)
                except:
                    cid = bond_types.index(list(reversed(combo)))
                out.write(f"{id : <5} {cid+1 : <5} {b[0]+1 : <5} {b[1]+1 : <5}\n")
            out.write('\n')
            if has_angles:
                out.write(f"Angles\n\n")
                for i, bend in enumerate(triplets):
                    id = i+1
                    angleid = triplets_angle_type[i]
                    combo = list(map(lambda paid: labels[pal[paid]], bend))
                    combo.append(angleid)
                    cid = bend_types.index(combo)
                    out.write(f"{id : <5} {cid+1 : <5} {bend[0]+1 : <5} {bend[1]+1 : <5} {bend[2]+1 : <5}\n")
                out.write('\n')
            if has_dihedrals:
                out.write(f"Dihedrals\n\n")
                for i, tor in enumerate(quadruplets):
                    id = i+1
                    combo = list(map(lambda paid: labels[pal[paid]], tor))
                    cid = tors_types.index(combo)
                    out.write(f"{id : <5} {cid+1 : <5} {tor[0]+1 : <5} {tor[1]+1 : <5} {tor[2]+1 : <5} {tor[3]+1 : <5}\n")
                out.write('\n')
            if has_impropers:
                out.write(f"Impropers\n\n")
                for i, itor in enumerate(iquadruplets):
                    id = i+1
                    combo = list(map(lambda paid: labels[pal[paid]], itor))
                    cid = itors_types.index(combo)
                    out.write(f"{id : <5} {cid+1 : <5} {itor[0]+1 : <5} {itor[1]+1 : <5} {itor[2]+1 : <5} {itor[3]+1 : <5}\n")
        
        log(f"Writing LAMMPS data file to {odata_path}")
        with open(odata_path, 'w') as out:
             write_lammps(out)
    
    if not only_atoms:
    
        # 1) Write LAMMPS data (with bonds/angles/…)
        with open(odata_path, 'w') as f:
            write_lammps(f)
    
        # 2) Write FF parameters (auto‐filled or placeholder)
        log(f"Writing FF parameters to {ocoeffs_path} (fill_parameters={fill_parameters})")
        if fill_parameters:
            with open(ocoeffs_path, 'w') as out:
                write_full_parameters(out, data, padef_data, labels)
        else:
            with open(ocoeffs_path, 'w') as f:
                write_placeholder(f, data, labels)

if __name__=="__main__":
    main()
