from typing import Any, Callable, List, Tuple

# A parameter specification: (name, type, constraint, description)
ParamSpec = Tuple[str, type, Callable[[Any], bool], str]

# A style entry defines required and optional parameter specs
StyleDef = dict[str, List[ParamSpec]]

# Registry mapping connectivity category -> style name -> definition
# Its important to add the 'energy' description for the energy unit conversion
# when the appropiate flag is used.

STYLE_REGISTRY: dict[str, dict[str, StyleDef]] = {
    # Pair interactions
    'pair_style': {
        'lj/cut': {
            'required': [
                ('epsilon', float, lambda v: True, 'energy'),
                ('sigma',   float, lambda v: True, 'distance'),
            ],
            'optional': [
                ('cutoff_lj', float, lambda v: True, 'distance'),
            ]
        },
        # Coulombic variants: cut, long, debye, dsf, msm, wolf
        **{f'lj/cut/coul/{m}': {
            'required': [
                ('epsilon', float, lambda v: True, 'energy'),
                ('sigma',   float, lambda v: True, 'distance'),
            ],
            'optional': [
                ('cutoff_lj', float, lambda v: True, 'distance'),
                ('cutoff_coul', float, lambda v: True, 'distance'),
            ]
        } for m in ['cut','long','debye','dsf','msm','wolf']},
        # TIP4P variants: 2 required, optional LJ cutoff
        **{f'lj/cut/tip4p/{m}': {
            'required': [
                ('epsilon', float, lambda v: True, 'energy'),
                ('sigma',   float, lambda v: True, 'distance'),
            ],
            'optional': [
                ('cutoff_lj', float, lambda v: True, 'distance'),
            ]
        } for m in ['cut','long']},
        # YAFF
        **{f'{m}/switch3/coulgauss/long': {
            'required': [
                ('epsilon',        float, lambda v: True, 'energy'),
                ('sigma',          float, lambda v: True, 'distance'),
                ('gaussian_width', float, lambda v: True, 'distance'),
            ],
            'optional': []
        } for m in ['lj','mm3']},
        # Buckingham
        'buck': {
            'required': [
                ('A', float, lambda v: True, 'energy'),
                ('rho',   float, lambda v: v>0, 'distance'),
                ('C',   float, lambda v: True, 'energy*distance^6'),
            ],
            'optional': [
                ('cutoff', float, lambda v: True, 'distance'),
                ('cutoff2', float, lambda v: True, 'distance'),
            ]
        },
        # Buckingham variants
        **{f'buck/coul/{m}': {
            'required': [
                ('A', float, lambda v: True, 'energy'),
                ('rho',   float, lambda v: v>0, 'distance'),
                ('C',   float, lambda v: True, 'energy*distance^6'),
            ],
            'optional': [
                ('cutoff', float, lambda v: True, 'distance'),
                ('cutoff2', float, lambda v: True, 'distance'),
            ]
        } for m in ['cut','long','msm']},
    },
    # Bond interactions
    'bond_style': {
        'harmonic': {
            'required': [
                ('K',  float, lambda v: True, 'energy/distance^2'),
                ('r0', float, lambda v: True, 'distance'),
            ],
            'optional': []
        },
        'morse': {
            'required': [
                ('D0',    float, lambda v: True, 'energy'),
                ('alpha', float, lambda v: v>0, 'inverse distance'),
                ('r0',    float, lambda v: True, 'distance'),
            ],
            'optional': []
        },
        'mm3': {
            'required': [
                ('K',  float, lambda v: True, 'energy/distance^2'),
                ('r0', float, lambda v: True, 'distance'),
            ],
            'optional': []
        },
        'class2': {
            'required': [
                ('r0',   float, lambda v: True, 'distance'),
                ('K2',   float, lambda v: True, 'energy/distance^2'),
                ('K3',   float, lambda v: True, 'energy/distance^3'),
                ('K4',   float, lambda v: True, 'energy/distance^4'),
            ],
            'optional': []
        },
        'none': {
            'required': [],
            'optional': []
        },
        'zero': {
            'required': [
                ('bdist', float, lambda v: True, 'distance'),
            ],
            'optional': []
        },
    },
    # Angle interactions
    'angle_style': {
        'harmonic': {
            'required': [
                ('K',      float, lambda v: True, 'energy'),
                ('theta0', float, lambda v: True, 'degrees'),
            ],
            'optional': []
        },
        'charmm': {
            'required': [
                ('K',     float, lambda v: True, 'energy'),
                ('theta0',float, lambda v: True, 'degrees'),
                ('K_ub',  float, lambda v: True, 'energy/distance^2'),
                ('r_ub',  float, lambda v: True, 'distance'),
            ],
            'optional': []
        },
        'fourier': {
            'required': [
                ('K',  float, lambda v: True, 'energy'),
                ('C0', float, lambda v: True, 'unitless'),
                ('C1', float, lambda v: True, 'unitless'),
                ('C2', float, lambda v: True, 'unitless'),
            ],
            'optional': []
        },
        'cosine': {
            'required': [
                ('K', float, lambda v: True, 'energy'),
            ],
            'optional': []
        },
        'none': {
            'required': [],
            'optional': []
        },
    },
    # Dihedral interactions
    'dihedral_style': {
        'harmonic': {
            'required': [
                ('K', float, lambda v: True, 'energy'),
                ('d', int,   lambda v: v in (1,-1),       '±1'),
                ('n', int,   lambda v: v>=0,             'integer ≥0'),
            ],
            'optional': []
        },
        'charmm': {
            'required': [
                ('K',     float, lambda v: True,         'energy'),
                ('n',     int,   lambda v: v>=0,         'integer ≥0'),
                ('d',     int,   lambda v: True,         'integer (degrees)'),
                ('wfac',  float, lambda v: v in (0.0,0.5,1.0), '0.0,0.5,1.0'),
            ],
            'optional': []
        },
        'charmmfsw': {
            'required': [
                ('K',    float, lambda v: True,          'energy'),
                ('n',    int,   lambda v: v>=0,          'integer ≥0'),
                ('d',    int,   lambda v: True,          'integer (degrees)'),
                ('wfac', float, lambda v: v in (0.0,0.5,1.0), '0.0,0.5,1.0'),
            ],
            'optional': []
        },
        'opls': {
            'required': [
                ('K1', float, lambda v: True, 'energy'),
                ('K2', float, lambda v: True, 'energy'),
                ('K3', float, lambda v: True, 'energy'),
                ('K4', float, lambda v: True, 'energy'),
            ],
            'optional': []
        },
        'quadratic': {
            'required': [
                ('K',      float, lambda v: True, 'energy'),
                ('theta0', float, lambda v: True, 'degrees'),
            ],
            'optional': []
        },
        'none': {
            'required': [],
            'optional': []
        },
    },
    # Improper interactions
    'improper_style': {
        'harmonic': {
            'required': [
                ('K',    float, lambda v: True, 'energy'),
                ('chi0', float, lambda v: True, 'degrees'),
            ],
            'optional': []
        },
        'cossq': {
            'required': [
                ('K',    float, lambda v: True, 'energy'),
                ('chi0', float, lambda v: True, 'degrees'),
            ],
            'optional': []
        },
        'cvff': {
            'required': [
                ('K', float, lambda v: True,          'energy'),
                ('d', int,   lambda v: v in (1,-1),   '±1'),
                ('n', int,   lambda v: v in (0,1,2,3,4,6), '0,1,2,3,4,6'),
            ],
            'optional': []
        },
        'distharm': {
            'required': [
                ('K',  float, lambda v: True, 'energy/distance^2'),
                ('d0', float, lambda v: True, 'distance'),
            ],
            'optional': []
        },
        'fourier': {
            'required': [
                ('K',    float, lambda v: True, 'energy'),
                ('C0',   float, lambda v: True, 'unitless'),
                ('C1',   float, lambda v: True, 'unitless'),
                ('C2',   float, lambda v: True, 'unitless'),
            ],
            'optional': [
                ('all_parm', int, lambda v: v in (0,1), '0 or 1'),
            ]
        },
        'inversion/harmonic': {
            'required': [
                ('K',      float, lambda v: True, 'energy'),
                ('omega0', float, lambda v: True, 'degrees'),
            ],
            'optional': []
        },
        'sqdistharm': {
            'required': [
                ('K',    float, lambda v: True, 'energy/distance^4'),
                ('d0sq', float, lambda v: True, 'distance^2'),
            ],
            'optional': []
        },
    },
}
