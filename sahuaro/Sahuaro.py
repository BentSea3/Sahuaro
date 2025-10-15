#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sahuaro.py version 1

Converts system into LAMMPS data + coeffs.

Example:
  ./Sahuaro.py path/to/sahuaro_input \
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

# Ppropuesta 1: SahuaroConfig CLASS for settings

class SahuaroConfig:
    """
    A class to hold and manage all system configuration settings from the
    input file's CONSIDERATIONS section and command-line arguments.
    """
    def __init__(self, data, args):
        cons = data.get('CONSIDERATIONS', {}) # Using init method

        # General considerations 
        self.only_atoms = cons.get('ONLY_ATOMS') == 'yes' # Creating attributes for the object
        self.include_tilt = cons.get('INCLUDE_TILT') == 'yes'
        self.charges_from_cif = cons.get('CHARGES_FROM_CIF', 'no') == 'yes'
        self.fill_parameters = cons.get('FILL_PARAMETERS', 'no').strip().lower() == 'yes'
        self.q_from_ct = cons.get('Q_FROM_CHARGETRANSFER', 'no').strip().lower() == 'yes'
        self.central_first = cons.get("IMPROPER_CENTRAL", "first") == "first"

        # System box replication considerations
        replicate_str = cons.get('REPLICATE', '').strip()
        if replicate_str:
            self.nx, self.ny, self.nz = map(int, replicate_str.split())
        else:
            self.nx = self.ny = self.nz = 1

        # Unit conversion considerations
        self.kelvin_to_kcal = cons.get('KELVIN_TO_KCALMOL', 'no') == 'yes'
        self.kj_to_kcal = cons.get('KJ_TO_KCALMOL', 'no') == 'yes'
        self.half_harmonic = cons.get('FACTOR_0.5_HARMONIC', 'no') == 'yes'

        # Energy factor considerations and error mesage
        raw = cons.get('FACTOR_E_DIH_IMP', 'no').strip().split()
        if raw[0].lower() == 'yes':
            if len(raw) != 2:
                print("Error: FACTOR_E_DIH_IMP must be “yes <number>”")
                sys.exit(1)
            self.factor_e_dih_imp = float(raw[1])
        else:
            self.factor_e_dih_imp = 1.0

        # I/O file path considerations
        self.flname = os.path.basename(data['PATHS']['CIF']).split('.')[0]
        self.today = datetime.today().strftime('%d-%m-%Y')
        self.odata_path = Path(args.output_dir) / 'in.data'
        self.ocoeffs_path = Path(args.output_dir) / 'in.coeffs'


# Propuesta 2: LammpsWriter CLASS for file output

class LammpsWriter:
    """
    A class responsible for writing all LAMMPS-formatted output files
    (in.data and in.coeffs).
    """
    def __init__(self, config, data, padef_data, labels, log_func):
        self.config = config
        self.data = data
        self.padef_data = padef_data
        self.labels = labels
        self.log = log_func # nota

    def write_coeffs_file(self):
        """Writes the in.coeffs file, either as a placeholder or fully populated."""
        self.log(f"Writing FF parameters to {self.config.ocoeffs_path} (fill_parameters={self.config.fill_parameters})")
        if self.config.fill_parameters:
            with open(self.config.ocoeffs_path, 'w') as out:
                self._write_full_parameters(out)
        else:
            with open(self.config.ocoeffs_path, 'w') as f:
                self._write_placeholder(f)

    def write_data_file(self, system_data):
        """Writes the in.data file, for atoms-only or the full system."""
        self.log(f"Writing LAMMPS data file to {self.config.odata_path}")
        with open(self.config.odata_path, 'w') as out:
            if self.config.only_atoms:
                self._write_atom_only_data(out, system_data)
            else:
                self._write_full_data(out, system_data)

    def _write_placeholder(self, out):
        out.write("# Complete with appropriate parameters\n\n")

        if self.config.only_atoms:
            out.write("# Pair coefficients (self pairs)\n")
            for i, lab in enumerate(self.labels, start=1):
                out.write(f"pair_coeff    {i:<3} {i:<3}    # is {lab}-{lab}\n")
            out.write("\n")
            return

        if self.data.get('BONDS'):
            out.write("# Bond coefficients\n")
            for i, row in enumerate(self.data['BONDS'], start=1):
                combo = f"{row['a1']}-{row['a2']}"
                out.write(f"bond_coeff    {i:<3}    # is {combo}\n")
            out.write("\n")

        if self.data.get('ANGLES'):
            out.write("# Angle coefficients\n")
            for i, row in enumerate(self.data['ANGLES'], start=1):
                combo = f"{row['a1']}-{row['a2']}-{row['a3']}"
                out.write(f"angle_coeff   {i:<3}    # is {combo}\n")
            out.write("\n")

        if self.data.get('PDIHEDRALS'):
            out.write("# Dihedral coefficients\n")
            for i, row in enumerate(self.data['PDIHEDRALS'], start=1):
                combo = f"{row['a1']}-{row['a2']}-{row['a3']}-{row['a4']}"
                out.write(f"dihedral_coeff {i:<3}    # is {combo}\n")
            out.write("\n")

        if self.data.get('IDIHEDRALS'):
            out.write("# Improper coefficients\n")
            for i, row in enumerate(self.data['IDIHEDRALS'], start=1):
                combo = f"{row['a1']}-{row['a2']}-{row['a3']}-{row['a4']}"
                out.write(f"improper_coeff {i:<3}    # is {combo}\n")
            out.write("\n")

        out.write("# Pair coefficients (self pairs)\n")
        for i, lab in enumerate(self.labels, start=1):
            out.write(f"pair_coeff    {i:<3} {i:<3}    # is {lab}-{lab}\n")
        out.write("\n")

    def _write_full_parameters(self, out):
        factor_kcal = 0.0019872041 if self.config.kelvin_to_kcal else 1.0
        factor_kj = 0.2390057 if self.config.kj_to_kcal else 1.0
        
        def make_mask(spec):
            all_params = spec['required'] + spec.get('optional', [])
            return ['energy' in desc for (_, _, _, desc) in all_params]

        ct = self.data.get('CONNECTIVITY_TYPES', {})

        # Write all bonds
        bond_style = ct.get('BOND_STYLE', 'none')
        bond_spec = STYLE_REGISTRY['bond_style'].get(bond_style, {'required': [], 'optional': []})
        ptypes = [ptype for (_, ptype, _, _) in (bond_spec['required'] + bond_spec.get('optional', []))]
        bond_mask = make_mask(bond_spec)
        if self.data.get('BONDS'):
            out.write("# Bond coefficients\n")
            for i, row in enumerate(self.data['BONDS'], start=1):
                a1, a2 = row['a1'], row['a2']
                params = [row[k] for k in row if k not in ('a1', 'a2')]
                
                if self.config.kelvin_to_kcal:
                    for idx, do_convert in enumerate(bond_mask):
                        if do_convert and idx < len(params): params[idx] *= factor_kcal
                if self.config.kj_to_kcal:
                    for idx, is_energy in enumerate(bond_mask):
                        if is_energy and idx < len(params): params[idx] *= factor_kj
                if self.config.half_harmonic and bond_style == 'harmonic':
                    for idx, is_energy in enumerate(bond_mask):
                        if is_energy and idx < len(params): params[idx] *= 0.5
                
                formatted = [f"{int(p):<6d}" if ptype is int else f"{p:<12.6f}" for ptype, p in zip(ptypes, params)]
                out.write(f"bond_coeff    {i:<3} {'  '.join(formatted)}    # {a1}-{a2}\n")
            out.write("\n")
        
        # Write all angles
        angle_style = ct.get('ANGLE_STYLE', 'none')
        angle_spec = STYLE_REGISTRY['angle_style'].get(angle_style, {'required':[], 'optional':[]})
        ptypes = [ptype for (_,ptype,_,_) in (angle_spec['required'] + angle_spec.get('optional',[]))]
        angle_mask = make_mask(angle_spec)
        if self.data.get('ANGLES'):
            out.write("# Angle coefficients\n")
            for i, row in enumerate(self.data['ANGLES'], start=1):
                a1,a2,a3 = row['a1'], row['a2'], row['a3']
                params = [row[k] for k in row if k not in ('a1','a2','a3')]

                if self.config.kelvin_to_kcal:
                    for idx, do_convert in enumerate(angle_mask):
                        if do_convert and idx < len(params): params[idx] *= factor_kcal
                if self.config.kj_to_kcal:
                    for idx, is_energy in enumerate(angle_mask):
                        if is_energy and idx < len(params): params[idx] *= factor_kj
                if self.config.half_harmonic and angle_style == 'harmonic':
                    for idx, is_energy in enumerate(angle_mask):
                        if is_energy and idx < len(params): params[idx] *= 0.5
                
                formatted = [f"{int(p):<6d}" if ptype is int else f"{p:<12.6f}" for ptype, p in zip(ptypes, params)]
                out.write(f"angle_coeff   {i:<3} {'  '.join(formatted)}    # {a1}-{a2}-{a3}\n")
            out.write("\n")

        # Write all dihedrals
        dih_style = ct.get('DIHEDRAL_STYLE', 'none')
        dih_spec = STYLE_REGISTRY['dihedral_style'].get(dih_style, {'required':[], 'optional':[]})
        ptypes = [ptype for (_,ptype,_,_) in (dih_spec['required'] + dih_spec.get('optional',[]))]
        dih_mask = make_mask(dih_spec)
        if self.data.get('PDIHEDRALS'):
            out.write("# Dihedral coefficients\n")
            for i, row in enumerate(self.data['PDIHEDRALS'], start=1):
                a1, a2, a3, a4 = row['a1'], row['a2'], row['a3'], row['a4']
                params = [row[k] for k in row if k not in ('a1', 'a2', 'a3', 'a4')]

                if self.config.kelvin_to_kcal:
                    for idx, do_convert in enumerate(dih_mask):
                        if do_convert and idx < len(params): params[idx] *= factor_kcal
                if self.config.kj_to_kcal:
                    for idx, is_energy in enumerate(dih_mask):
                        if is_energy and idx < len(params): params[idx] *= factor_kj
                if self.config.factor_e_dih_imp != 1.0:
                    for idx, is_e in enumerate(dih_mask):
                        if is_e and idx < len(params): params[idx] *= self.config.factor_e_dih_imp
                
                formatted = [f"{int(p):<6d}" if ptype is int else f"{p:<12.6f}" for ptype, p in zip(ptypes, params)]
                out.write(f"dihedral_coeff {i:<3} {'  '.join(formatted)}    # {a1}-{a2}-{a3}-{a4}\n")
            out.write("\n")
        
        # Write all improper dihedrals
        imp_style = ct.get('IMPROPER_STYLE', 'none')
        imp_spec = STYLE_REGISTRY['improper_style'].get(imp_style, {'required':[], 'optional':[]})
        ptypes = [ptype for (_,ptype,_,_) in (imp_spec['required'] + imp_spec.get('optional',[]))]
        imp_mask = make_mask(imp_spec)
        if self.data.get('IDIHEDRALS'):
            out.write("# Improper coefficients\n")
            for i, row in enumerate(self.data['IDIHEDRALS'], start=1):
                a1, a2, a3, a4 = row['a1'], row['a2'], row['a3'], row['a4']
                params = [row[k] for k in row if k not in ('a1', 'a2', 'a3', 'a4')]

                if self.config.kelvin_to_kcal:
                    for idx, do_convert in enumerate(imp_mask):
                        if do_convert and idx < len(params): params[idx] *= factor_kcal
                if self.config.kj_to_kcal:
                    for idx, is_energy in enumerate(imp_mask):
                        if is_energy and idx < len(params): params[idx] *= factor_kj
                if self.config.factor_e_dih_imp != 1.0:
                    for idx, is_e in enumerate(imp_mask):
                        if is_e and idx < len(params): params[idx] *= self.config.factor_e_dih_imp
                
                formatted = [f"{int(p):<6d}" if ptype is int else f"{p:<12.6f}" for ptype, p in zip(ptypes, params)]
                out.write(f"improper_coeff {i:<3} {'  '.join(formatted)}    # {a1}-{a2}-{a3}-{a4}\n")
            out.write("\n")

        # Writing the coefficients of the force field
        out.write("# Pair coefficients (self pairs)\n")
        pair_style = ct.get('PAIR_STYLE', 'none')
        pair_spec = STYLE_REGISTRY['pair_style'].get(pair_style, {'required':[], 'optional':[]})
        pair_mask = make_mask(pair_spec)
        atom_rows = self.data.get('ATOMS', [])
        if atom_rows:
            atom_map = {row['label']: row for row in atom_rows}
            core = {'label', 'type', 'charge', 'radius', 'mass'}
            extras = [k for k in atom_rows[0] if k not in core]
            for i, lab in enumerate(self.labels, start=1):
                row = atom_map[lab]
                params = [float(row.get(col, 0.0)) for col in extras]

                if self.config.kelvin_to_kcal:
                    for idx, do_convert in enumerate(pair_mask):
                        if do_convert and idx < len(params): params[idx] *= factor_kcal
                if self.config.kj_to_kcal:
                    for idx, is_energy in enumerate(pair_mask):
                        if is_energy and idx < len(params): params[idx] *= factor_kj
                
                param_str = "  ".join(f"{p:<12.6f}" for p in params)
                out.write(f"pair_coeff    {i:<3} {i:<3}  {param_str}    # {lab}-{lab}\n")
        out.write("\n")

    def _write_atom_only_data(self, out, system_data):
        """Writes the in.data file when only_atoms is true."""
        N = system_data['N']
        pal = system_data['pal']
        charges = system_data['charges']
        pos = system_data['pos']
        xlo, ylo, zlo = 0.0, 0.0, 0.0
        xhi, yhi, zhi = system_data['box']

        out.write(f"{self.config.flname} LAMMPS input file. Created on {self.config.today} using Sahuaro.\n\n")
        out.write(f"{N} atoms\n\n")
        out.write(f"{len(self.labels)} atom types\n\n")
        out.write(f"{xlo} {xhi} xlo xhi\n")
        out.write(f"{ylo} {yhi} ylo yhi\n")
        out.write(f"{zlo} {zhi} zlo zhi\n\n")
        if self.config.include_tilt:
            out.write("0.000000 0.000000 0.000000 xy xz yz\n\n")
        out.write("Masses\n\n")
        for i, label in enumerate(self.labels, start=1):
            m = self.padef_data[label]['mass']
            out.write(f"{i:<5}{m:<12.6f} # {label}\n")
        out.write("\nAtoms\n\n")
        for i in range(N):
            id_ = i + 1
            molid = 0
            labelid = pal[i] + 1
            q = charges[i]
            x, y, z = pos[i]
            out.write(f"{id_:<5}{molid:<3}{labelid:<3}{q:<10.6f}"
                      f"{x:>12.6f}{y:>12.6f}{z:>12.6f}\n")
        out.write("\n")

    def _write_full_data(self, out, system_data):
        """Writes the in.data file with full connectivity."""
        pos = system_data['pos']
        pal = system_data['pal']
        charges = system_data['charges']
        bonds = system_data['bonds']
        xlo, ylo, zlo = 0.0, 0.0, 0.0
        xhi, yhi, zhi = system_data['box']

        has_angles = system_data.get('has_angles', False)
        has_dihedrals = system_data.get('has_dihedrals', False)
        has_impropers = system_data.get('has_impropers', False)
        
        triplets = system_data.get('triplets', [])
        quadruplets = system_data.get('quadruplets', [])
        iquadruplets = system_data.get('iquadruplets', [])
        
        fw_data = self.data.get('FRAMEWORK', {})
        fw_data_bends = system_data.get('fw_data_bends', [])

        out.write(f"{self.config.flname} LAMMPS input file. Created on {self.config.today} using Sahuaro.\n\n")
        out.write(f"{len(pos)} atoms\n")
        out.write(f"{len(bonds)} bonds\n")
        if has_angles: out.write(f"{len(triplets)} angles\n")
        if has_dihedrals: out.write(f"{len(quadruplets)} dihedrals\n")
        if has_impropers: out.write(f"{len(iquadruplets)} impropers\n")
        out.write('\n')
        out.write(f"{len(self.labels)} atom types\n")
        out.write(f"{len(fw_data['bonds'])} bond types\n")
        if has_angles: out.write(f"{len(fw_data_bends)} angle types\n")
        if has_dihedrals: out.write(f"{len(fw_data['tors'])} dihedral types\n")
        if has_impropers: out.write(f"{len(fw_data['itors'])} improper types\n")
        out.write('\n')
        out.write(f"{xlo} {xhi} xlo xhi\n")
        out.write(f"{ylo} {yhi} ylo yhi\n")
        out.write(f"{zlo} {zhi} zlo zhi\n")
        out.write('\n')
        if self.config.include_tilt:
            out.write(f"0.000000 0.000000 0.000000 xy xz yz\n\n")
        else:
            out.write('\n')
        out.write(f"Masses\n\n")
        for i, label in enumerate(self.labels):
            masas = self.padef_data[label]['mass']
            out.write(f"{i+1}  {masas}    # {label}\n")
        out.write('\n')

        bond_types = [entry['combo'] for entry in fw_data['bonds']]
        if has_angles: bend_types = fw_data_bends
        if has_dihedrals: tors_types = [entry['combo'] for entry in fw_data['tors']]
        if has_impropers: itors_types = [entry['combo'] for entry in fw_data['itors']]

        out.write(f"Atoms\n\n")
        for i in range(len(pos)):
            out.write(f"{i+1 : <5} {0 : <3} {pal[i]+1 : <3} {float(charges[i]) : <10} "
                      f"{pos[i][0] : <15.8f} {pos[i][1] : <15.8f} {pos[i][2] : <15.8f}\n")
        out.write('\n')
        
        out.write(f"Bonds\n\n")
        for i, b in enumerate(bonds):
            combo = [self.labels[pal[b[0]]], self.labels[pal[b[1]]]]
            try:
                cid = bond_types.index(combo)
            except ValueError:
                cid = bond_types.index(list(reversed(combo)))
            out.write(f"{i+1 : <5} {cid+1 : <5} {b[0]+1 : <5} {b[1]+1 : <5}\n")
        out.write('\n')

        if has_angles:
            out.write(f"Angles\n\n")
            triplets_angle_type = system_data.get('triplets_angle_type', [])
            for i, bend in enumerate(triplets):
                angleid = triplets_angle_type[i]
                combo = [self.labels[pal[paid]] for paid in bend]
                combo.append(angleid)
                cid = bend_types.index(combo)
                out.write(f"{i+1 : <5} {cid+1 : <5} {bend[0]+1 : <5} {bend[1]+1 : <5} {bend[2]+1 : <5}\n")
            out.write('\n')
        
        if has_dihedrals:
            out.write(f"Dihedrals\n\n")
            for i, tor in enumerate(quadruplets):
                combo = [self.labels[pal[paid]] for paid in tor]
                cid = tors_types.index(combo)
                out.write(f"{i+1 : <5} {cid+1 : <5} {tor[0]+1 : <5} {tor[1]+1 : <5} {tor[2]+1 : <5} {tor[3]+1 : <5}\n")
            out.write('\n')

        if has_impropers:
            out.write(f"Impropers\n\n")
            for i, itor in enumerate(iquadruplets):
                combo = [self.labels[pal[paid]] for paid in itor]
                cid = itors_types.index(combo)
                out.write(f"{i+1 : <5} {cid+1 : <5} {itor[0]+1 : <5} {itor[1]+1 : <5} {itor[2]+1 : <5} {itor[3]+1 : <5}\n")

def main():
    # Parses the command line arguments using argparsse
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
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True) # Output folder
    
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

    # Call the SahuaroConfig class to handle settings
    config = SahuaroConfig(data, args)

    # Validate the considerations
    allowed_cons_keys = {
        'ONLY_ATOMS', 'INCLUDE_TILT', 'CHARGES_FROM_CIF', 'REPLICATE',
        'FILL_PARAMETERS', 'WRITE_LOG', 'Q_FROM_CHARGETRANSFER',
        'IMPROPER_CENTRAL', 'KELVIN_TO_KCALMOL', 'KJ_TO_KCALMOL',
        'FACTOR_0.5_HARMONIC', 'FACTOR_E_DIH_IMP',
    }
    unknown = set(data.get('CONSIDERATIONS', {})) - allowed_cons_keys
    if unknown:
        msg = f"Error: Unknown CONSIDERATIONS keys: {', '.join(sorted(unknown))}"
        log(msg)
        sys.exit(1)
    
    log(f"Parsed input file: {data['PATHS']['CIF']}")
    log(f"  ONLY_ATOMS= {config.only_atoms}, FILL_PARAMETERS= {config.fill_parameters}, WRITE_LOG= {args.log}")
    log(f"  REPLICATE = {config.nx} × {config.ny} × {config.nz}, CHARGES_FROM_CIF= {config.charges_from_cif}, INCLUDE_TILT= {config.include_tilt}")
    log(f"  FACTOR_E_DIH_IMP = {config.factor_e_dih_imp}")
    
    cfg_dir = Path(args.config).parent
    cif_rel = Path(data['PATHS']['CIF'])
    cif_path = cfg_dir / cif_rel if not cif_rel.is_absolute() else cif_rel
    data['PATHS']['CIF'] = str(cif_path)

    # Parse CIF and PADEF (Pseudo Atom DEFinitions).
    cif_data = read_cif_file(data['PATHS']['CIF'])
    padef_data = data['PADEF']
    
    log(f"Read CIF: {len(cif_data['_atoms'])} sites, cell = {cif_data['cell']['length']}")
    log(f"Loaded {len(padef_data)} atom types from definitions")
    
    fw_data = data.get('FRAMEWORK', {})
    
    if not config.only_atoms and len(data.get("BONDS", [])) == 0:
        log("ONLY_ATOMS is set to 'no', but no BONDS were provided.")
        log("Please add bond connectivity information to the input file")
        raise ValueError("Missing required bond connectivity.")
    
    if config.kelvin_to_kcal and config.kj_to_kcal:
        msg = "Cannot use both KELVIN_TO_KCALMOL and KJ_TO_KCALMOL at the same time."
        if args.log: log(msg); sys.exit(1)
        else: raise ValueError(msg)
    
    if config.charges_from_cif and config.q_from_ct:
        log("Cannot use CHARGES_FROM_CIF and Q_FROM_CHARGETRANSFER simultaneously.")
        raise ValueError("Cannot use CHARGES_FROM_CIF and Q_FROM_CHARGETRANSFER simultaneously.")
    
    if config.fill_parameters:
        log("Validating input coefficients with the style registry . . .")
        
        ct = {
            key.strip().upper(): val.strip().lower() 
            for key, val in data.get('CONNECTIVITY_TYPES', {}).items()
        }
    
        def validate_section(rows, style_key, base_cols, registry_key, section_name):
            if not rows: return
            style = ct.get(style_key.upper(), 'none')
            if style == 'none': return

            if style != 'none' and not rows:
                log(f"Warning: {section_name}_STYLE='{style}' but no {section_name.lower()} rows found.")
                return

            styles_map = STYLE_REGISTRY[registry_key]
            spec = styles_map.get(style)
            if spec is None:
                valid = ", ".join(sorted(styles_map))
                raise ValueError(
                    f"Unknown {registry_key!r} '{style}'.\n"
                    f"Supported values are: {valid}"
                )
            
            req, opt = spec['required'], spec.get('optional', [])
            n_req, n_max = len(req), len(req) + len(opt)
            
            for i, row in enumerate(rows, start=1):
                extras = [k for k in row if k not in base_cols]
                if not (n_req <= len(extras) <= n_max):
                    raise ValueError(
                        f"{section_name} row {i}: for style '{style}', "
                        f"expected {n_req}–{n_max} params but found {len(extras)}"
                    )
                all_params = req + opt
                for (param_name, ptype, cond, msg), col in zip(all_params, extras):
                    val = row[col]
                    if not isinstance(val, ptype):
                        raise ValueError(
                            f"{section_name} row {i} '{col}': expected {ptype.__name__}, got {type(val).__name__}"
                        )
                    if not cond(val):
                        raise ValueError(
                            f"{section_name} row {i} '{col}': value {val} fails constraint ({msg})"
                        )

        errors = []
        validations = [
            (data.get('BONDS', []),      'BOND_STYLE',     ['a1','a2'],          'bond_style',     'Bond'),
            (data.get('ANGLES', []),     'ANGLE_STYLE',    ['a1','a2','a3'],     'angle_style',    'Angle'),
            (data.get('PDIHEDRALS', []), 'DIHEDRAL_STYLE', ['a1','a2','a3','a4'], 'dihedral_style', 'Dihedral'),
            (data.get('IDIHEDRALS', []), 'IMPROPER_STYLE', ['a1','a2','a3','a4'], 'improper_style', 'Improper'),
            (data.get('ATOMS', []),      'PAIR_STYLE',     ['label','type','charge','radius','mass'], 'pair_style', 'Pair'),
        ]
        for v_args in validations:
            try:
                validate_section(*v_args)
            except ValueError as e:
                errors.append(str(e))
        
        if errors:
            raise ValueError(
                "Connectivity/style validation failed with the following problems:\n  - "
                + "\n  - ".join(errors)
            )
        
        log("  All connectivity and pair parameters congruent with LAMMPS style template")
    
    labels = list(padef_data.keys())
    label_k = {lab: i for i, lab in enumerate(labels)}
    pa_labels = labels
    atom_types = len(labels)
    
    pa_number = len(cif_data['_atoms'])
    idx = np.arange(pa_number)
    pal = np.array([label_k[row['label']] for row in cif_data['_atoms']], dtype=int)
    
    xhi = cif_data['cell']['length']['a']
    yhi = cif_data['cell']['length']['b']
    zhi = cif_data['cell']['length']['c']
    pos = np.array([[row['x']*xhi, row['y']*yhi, row['z']*zhi] for row in cif_data['_atoms']])
    
    if (config.nx, config.ny, config.nz) != (1, 1, 1):
        base_pos, base_pal, base_idx = pos.copy(), pal.copy(), idx.copy()
        new_pos, new_pal, new_idx = [], [], []
        for ix in range(config.nx):
            for iy in range(config.ny):
                for iz in range(config.nz):
                    shift = np.array([ix * xhi, iy * yhi, iz * zhi])
                    new_pos.extend(base_pos + shift)
                    new_pal.extend(base_pal)
                    new_idx.extend(base_idx)
        pos = np.array(new_pos)
        pal = np.array(new_pal, dtype=int)
        idx = np.array(new_idx, dtype=int)
        xhi *= config.nx; yhi *= config.ny; zhi *= config.nz
        log(f"{len(pos)} sites. New cell = {xhi:.2f}×{yhi:.2f}×{zhi:.2f} Å ({config.nx}×{config.ny}×{config.nz})")

    box = np.array([xhi, yhi, zhi])
    N = len(pos)
    
    pcharges_base = np.array([padef_data[lab]['charge'] for lab in labels])
    qcif_base = np.array([float(atom.get('charge', 0.0)) for atom in cif_data['_atoms']])
    
    if not config.q_from_ct:
        charges = qcif_base[idx] if config.charges_from_cif else pcharges_base[pal]
        net_charge = charges.sum()
        log(f"Net system charge: {net_charge:.6f}")
        if abs(net_charge) > 1e-6:
            log("Warning: system is not neutral. Check input charges.")
    else:
        charges = None # Will be filled later
    
    for dim, L in zip(('x', 'y', 'z'), box):
        if L < 24.0:
            log(f"{dim}-box = {L:.2f} Å (<24 Å). Consider using REPLICATE.")
    
    # Call LammpsWriter class to handle file output
    writer = LammpsWriter(config, data, padef_data, labels, log)
    
    if config.only_atoms:
        system_data_for_writer = {
            'N': N, 'pal': pal, 'charges': charges, 'pos': pos, 'box': box
        }
        writer.write_data_file(system_data_for_writer)
        writer.write_coeffs_file()
        
        log_msg = "ONLY_ATOMS: wrote data + parameters"
        if args.log: log(log_msg); sys.exit(0)
        else: sys.exit(log_msg)
    
    # Nota
    
    radii = np.array([padef_data[label]['radii'] for label in pa_labels])
    K = atom_types
    threshold = np.zeros((K, K))
    for k1 in range(K):
        for k2 in range(K):
            threshold[k1, k2] = 0.56 + radii[k1] + radii[k2]
    
    has_angles = len(fw_data.get("bends", [])) > 0
    has_dihedrals = len(fw_data.get("tors", [])) > 0
    has_impropers = len(fw_data.get("itors", [])) > 0
    
    def fix_diff(rs, box):
        rs[rs[:, 0] > 0.5 * box[0], 0] -= box[0]
        rs[rs[:, 1] > 0.5 * box[1], 1] -= box[1]
        rs[rs[:, 2] > 0.5 * box[2], 2] -= box[2]
        return rs

    def dist2(v1, v2, box):
        diff = v1 - v2
        diff = diff - box * np.rint(diff / box) # More robust PBC wrapping
        return np.linalg.norm(diff)

    def calc_bonds(pos, box):
        bs, ds = [], []
        for i in range(1, N):
            for j in range(i):
                if pal[j] not in valid_bonds[pal[i]]: continue
                d = dist2(pos[i], pos[j], box)
                if d < threshold[pal[i], pal[j]]:
                    bs.append((i, j))
                    ds.append(d)
        return bs, np.array(ds)

    label_valid_bonds = {}
    for entry in fw_data['bonds']:
        a1, a2 = entry['combo']
        label_valid_bonds.setdefault(a1, []).append(a2)
        label_valid_bonds.setdefault(a2, []).append(a1)
    
    valid_bonds = [[label_k[l2] for l2 in label_valid_bonds.get(l1, [])] for l1 in labels]

    log("Detecting bonds . . .")
    (bonds, ds) = calc_bonds(pos, box)
    log(f"  Found {len(bonds)} bonds")
    
    con_list = [[] for _ in range(N)]
    for j1, j2 in bonds:
        con_list[j1].append(j2)
        con_list[j2].append(j1)
        
    if config.q_from_ct:
        ct_path = Path(data['PATHS']['CIF']).parent / 'charge_transfer.dat'
        try:
            raw = np.genfromtxt(str(ct_path), dtype=None, encoding=None)
        except OSError:
            log(f"Charge transfer table not found at {ct_path!r}")
            raise FileNotFoundError(f"Charge transfer table not found at {ct_path!r}")
        
        donors, accepts, dq_vals = [row[0] for row in raw], [row[1] for row in raw], [row[2] for row in raw]
        pal_ct = [list(map(label_k.get, donors)), list(map(label_k.get, accepts)), dq_vals]
        
        charg = []
        for i in range(N):
            q_trans = 0.0
            main = pal[i]
            for nei in con_list[i]:
                nbr = pal[nei]
                for u, v, deltaq in zip(*pal_ct):
                    if (u, v) == (main, nbr): q_trans -= deltaq
                    elif (u, v) == (nbr, main): q_trans += deltaq
            charg.append(q_trans)
        
        charges = np.array(charg)
        net_charge_ct = charges.sum()
        log(f"Net system charge after charge-transfer: {net_charge_ct:.6f}")
        if abs(net_charge_ct) > 1e-6:
            log("Warning: system is not charge neutral. Check your charge transfer table.")

    # Prepare data for the writer
    system_data_for_writer = {
        'pos': pos, 'pal': pal, 'charges': charges, 'bonds': bonds, 'box': box,
        'has_angles': has_angles, 'has_dihedrals': has_dihedrals, 'has_impropers': has_impropers
    }
            
    if has_angles:
        log("Detecting angles . . .")
        valid_bends = [[label_k[label] for label in entry['combo']] for entry in fw_data['bends']]
        
        triplets = []
        for i in range(N):
            js = con_list[i]
            for a_idx in range(len(js)):
                for b_idx in range(a_idx + 1, len(js)):
                    j1, j2 = js[a_idx], js[b_idx]
                    combo = [pal[j1], pal[i], pal[j2]]
                    if combo in valid_bends:
                        triplets.append((j1, i, j2))
                    elif list(reversed(combo)) in valid_bends:
                        triplets.append((j2, i, j1))

        def bond_angle(triplet):
            a, i, b = triplet
            v_ia = pos[a] - pos[i]
            v_ib = pos[b] - pos[i]
            v_ia -= box * np.rint(v_ia / box)
            v_ib -= box * np.rint(v_ib / box)
            cosine_angle = np.dot(v_ia, v_ib) / (np.linalg.norm(v_ia) * np.linalg.norm(v_ib))
            return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

        triplets_angles = [bond_angle(triplet) for triplet in triplets]
        
        trip_classif_angle = {}
        triplets_angle_type = []
        
        def classif_angle(triplet, angle):
            tlab = tuple(pa_labels[pal[p]] for p in triplet)
            if tlab not in trip_classif_angle:
                trip_classif_angle[tlab] = []
            
            for i, other_angle in enumerate(trip_classif_angle[tlab]):
                diff = abs(angle - other_angle)
                if min(360.0 - diff, diff) < 2.0:
                    triplets_angle_type.append(i)
                    return
            
            triplets_angle_type.append(len(trip_classif_angle[tlab]))
            trip_classif_angle[tlab].append(angle)

        for triplet, angle in zip(triplets, triplets_angles):
            classif_angle(triplet, angle)
        
        fw_data_bends = []
        for x in fw_data['bends']:
            alab, ilab, blab = x['combo']
            tlab_key = (alab, ilab, blab)
            if tlab_key in trip_classif_angle:
                angles = trip_classif_angle[tlab_key]
                for i in range(len(angles)):
                    fw_data_bends.append([alab, ilab, blab, i])

        log(f"  Found {len(triplets)} angle triplets, {len(fw_data_bends)} unique angle types")
        system_data_for_writer['triplets'] = triplets
        system_data_for_writer['triplets_angle_type'] = triplets_angle_type
        system_data_for_writer['fw_data_bends'] = fw_data_bends

    if has_dihedrals:
        log("Detecting dihedrals . . .")
        valid_tors = [[label_k[label] for label in entry['combo']] for entry in fw_data['tors']]
        quadruplets = []
        for i, j in bonds:
            for a in con_list[i]:
                if a == j: continue
                for b in con_list[j]:
                    if b == i or b == a: continue
                    combo = [pal[a], pal[i], pal[j], pal[b]]
                    if combo in valid_tors:
                        quadruplets.append((a, i, j, b))
                    elif list(reversed(combo)) in valid_tors:
                        quadruplets.append((b, j, i, a))
        log(f"  Found {len(quadruplets)} dihedral quadruplets")
        system_data_for_writer['quadruplets'] = quadruplets
        
    if has_impropers:
        log("Detecting impropers . . .")
        valid_itors = [[label_k[label] for label in entry['combo']] for entry in fw_data['itors']]
        iquadruplets = []
        for i in range(N):
            if len(con_list[i]) < 3: continue
            for p1, p2, p3 in combinations(con_list[i], 3):
                if config.central_first:
                    potential_quad = (i, p1, p2, p3)
                else: # last
                    potential_quad = (p1, p2, p3, i)
                
                # Check all permutations of the outer 3 atoms
                pals_quad = [pal[idx] for idx in potential_quad]
                center_pal = pals_quad[0] if config.central_first else pals_quad[3]
                outer_pals = sorted([p for p in pals_quad if p != center_pal]) # Not right, need to check permutations
                
                # Simplified check, might not catch all permutations but works for many cases
                # A full check would involve itertools.permutations
                if pals_quad in valid_itors:
                    iquadruplets.append(potential_quad)
                    continue # Found a match, move to next combination
        
        iquadruplets = []
        for i in range(N):
            if len(con_list[i]) < 3: continue
            for p_tuple in combinations(con_list[i], 3):
                p_list = list(p_tuple)
                # This logic is complex, keeping the original logic for now
                # as it was more specific about permutations.
                if config.central_first:
                    perms = [[i,p_list[0],p_list[1],p_list[2]], [i,p_list[0],p_list[2],p_list[1]],
                             [i,p_list[1],p_list[0],p_list[2]], [i,p_list[1],p_list[2],p_list[0]],
                             [i,p_list[2],p_list[0],p_list[1]], [i,p_list[2],p_list[1],p_list[0]]]
                else: # last
                    perms = [[p_list[0],p_list[1],p_list[2],i], [p_list[0],p_list[2],p_list[1],i],
                             [p_list[1],p_list[0],p_list[2],i], [p_list[1],p_list[2],p_list[0],i],
                             [p_list[2],p_list[0],p_list[1],i], [p_list[2],p_list[1],p_list[0],i]]
                
                for perm in perms:
                    pals_perm = [pal[idx] for idx in perm]
                    if pals_perm in valid_itors:
                        iquadruplets.append(perm)
                        break
                        
        log(f"  Found {len(iquadruplets)} improper quadruplets")
        system_data_for_writer['iquadruplets'] = iquadruplets

    # Call the writer
    writer.write_data_file(system_data_for_writer)
    writer.write_coeffs_file()

if __name__ == "__main__":
    main()
