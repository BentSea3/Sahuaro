import json
import re

####################
# COMMON UTILITIES #
####################

def data_to_json(data):
    return json.dumps(
        data,
        ensure_ascii=False,
        check_circular=False,
        allow_nan=True,
        indent=4,
    )

def parse_number_string(string):
    try:
        return int(string)
    except ValueError:
        pass
    try:
        return float(string)
    except ValueError:
        pass
    return None

def is_number(x):
    return isinstance(x, (int, float))

def parse_row_maybe_numbers(line):
    row = line.split()
    data = []
    for entry in row:
        numeric = parse_number_string(entry)
        if is_number(numeric):
            data.append(numeric)
        else:
            data.append(entry)
    return data

def parse_line(line):
    return parse_row_maybe_numbers(line)

########################
# SAHUARO FILE PARSER  #
########################

class ParserError(Exception):
    pass

def parse_file(path, compat=False):
    with open(path) as stream:
        return parse_input(stream, compat)

def parse_input(stream, compat=False):
    state = 'PREAMBLE'
    paths = {}
    headers = {}
    atoms = []
    bonds = []
    angles = []
    pdihedrals = []
    idihedrals = []
    considerations = {}
    connectivity_types = {}
    for i, line in enumerate(stream):
        line = line.strip()
        if not line or line[0] == '#':
            continue
        if line[-1] == ':':
            state = line[:-1]
            continue
        if state == 'PATHS':
            match = re.search(r'([^=\s]*)\s*=\s*(.*)', line)
            if not match:
                raise ParserError(f"Malformed PATHS line {i+1}")
            paths[match.group(1)] = match.group(2)
            continue
        if state == 'CONSIDERATIONS':
            match = re.search(r'([^=\s]*)\s*=\s*(.*)', line)
            if not match:
                raise ParserError(f"Malformed CONSIDERATIONS line {i+1}")
            considerations[match.group(1)] = match.group(2).strip().lower()
            continue
        if state == 'CONNECTIVITY_TYPES':
            match = re.search(r'([^:\s]+)\s*:\s*(.*)', line)
            if not match:
                raise ParserError(f"Malformed CONNECTIVITY_TYPES line {i+1}")
            key   = match.group(1).strip().upper()
            value = match.group(2).strip().lower()
            connectivity_types[key] = value
            continue
        if state in ('ATOMS', 'BONDS', 'ANGLES', 'PDIHEDRALS', 'IDIHEDRALS'):
            parse_table_row(state, headers, locals()[state.lower()], line)
            continue

    data = {
        'PATHS': paths,
        'ATOMS': atoms,
        'BONDS': bonds,
        'ANGLES': angles,
        'PDIHEDRALS': pdihedrals,
        'IDIHEDRALS': idihedrals,
        'CONSIDERATIONS': considerations,
        'CONNECTIVITY_TYPES': connectivity_types,
    }

    if compat:
        data['PADEF'] = {
            x['label']: {
                'mass': x['mass'],
                'charge': x['charge'],
                'radii': x['radius'],
                'type': x['type'],
                #'lj_eps': x['epsilon'],
                #'lj_sig': x['sigma'],
            }
            for x in atoms
        }
        data['FRAMEWORK'] = {
            'bonds': [{
                'combo': [x['a1'], x['a2']],
            } for x in bonds],
            'bends': [{
                'combo': [x['a1'], x['a2'], x['a3']],
            } for x in angles],
            'tors': [{
                'combo': [x['a1'], x['a2'], x['a3'], x['a4']],
            } for x in pdihedrals],
            'itors': [{
                'combo': [x['a1'], x['a2'], x['a3'], x['a4']],
            } for x in idihedrals],
        }

    return data

def parse_table_row(state, headers, rows, line):
    if state not in headers:
        headers[state] = parse_line(line)
    else:
        rows.append(dict(zip(headers[state], parse_line(line))))

#####################
# CIF FILE PARSING  #
#####################

class CifParsingError(Exception):
    pass

def read_cif_file(path):
    with open(path) as cif_file:
        return parse_cif_file(cif_file)

def parse_cif_file(cif_file):
    state = 'info'
    data = {}
    columns = []
    ncols = 0
    atoms = []
    for i, line in enumerate(cif_file):
        if not re.search(r'[^\s]', line):
            continue
        if state == 'info':
            if line[0] == '_':
                entry = parse_cif_underscored(line)
                insert_into_data(data, entry['path'], entry['value'])
            elif line.startswith('loop_'):
                state = 'column'
        elif state == 'column':
            if line[0] == '_':
                path = underscored_to_path(line)
                if not path:
                    raise CifParsingError(f"Unknown CIF matrix column name {line} at line {i}")
                columns.append(path[-1])
                ncols += 1
            else:
                state = 'matrix'
                first_row = parse_cif_matrix_line(line)
                if len(first_row) != ncols:
                    raise CifParsingError(f"Mismatch between CIF row and number of columns at line {i}")
                atoms.append(dict(zip(columns, first_row)))
        elif state == 'matrix':
            row = parse_cif_matrix_line(line)
            if len(row) != ncols:
                raise CifParsingError(f"Mismatch between CIF row and number of columns at line {i}")
            atoms.append(dict(zip(columns, row)))
        else:
            raise CifParsingError(f"Unknown CIF parsing state '{state}' at line {i}")

    data['_columns'] = columns
    data['_atoms'] = atoms
    return data

def cif_data_to_json(cif_data):
    return data_to_json(cif_data)

def cif_data_to_csv(cif_data):
    columns = cif_data['_columns']
    content = ', '.join(columns)
    for row in cif_data['_atoms']:
        line = ', '.join(str(row[col]) for col in columns)
        content += '\n' + line
    return content

def insert_into_data(data, path, value):
    ref = data
    for key in path[:-1]:
        if key not in ref:
            ref[key] = {}
        ref = ref[key]
    ref[path[-1]] = value
    return data

def underscored_to_path(underscored):
    parts = underscored.strip().split('_')
    return [p for p in parts if p]

def parse_underscore_value(string):
    string = string.strip()
    if not string:
        return None
    if re.match(r'[\'\"]', string[0]):
        return string[1:-1]
    if re.match(r'[a-zA-Z]', string[0]):
        return string
    numeric = parse_number_string(string)
    return numeric if is_number(numeric) else string

def parse_cif_underscored(line):
    match = re.search(r'\s', line)
    blank_at = len(line) if not match else match.start()
    left = line[:blank_at]
    right = line[blank_at:]
    path = underscored_to_path(left)
    value = parse_underscore_value(right)
    return { 'path': path, 'value': value }

def parse_cif_matrix_line(line):
    return parse_row_maybe_numbers(line)
