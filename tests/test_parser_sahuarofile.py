import pytest
import io
from sahuaro.sahuaro_utils import parse_input, ParserError

# def test_parse_input():
#     with open("tests/data/sahuaro_input") as stream:
#         returndata = parse_input(stream, False)
#         print(type(returndata))
#         print(returndata)

# Defining the full example input. Data from the file
# located in 'data' folder
EXAMPLE_SAHUARO_INPUT_DATA = """
# CIF to LAMMPS input file. Sahuaro.

PATHS:
CIF = irmof1.cif

ATOMS:
label  type  charge  radius  mass     P1     P2
Zn1    Zn     1.275  1.6      65.37    0.42   2.7
O1     O     -1.5    0.68     15.9994  700.0  2.98
O2     O     -0.6    0.68     15.9994  70.5   3.11
C1     C      0.475  0.720    12.0107  47.0   3.74
C2     C      0.125  0.720    12.0107  47.86  3.47
C3     C     -0.15   0.720    12.0107  47.86  3.47
H1     H      0.15   0.320    1.00794  7.65   2.85

CONSIDERATIONS:
WRITE_LOG = yes
ONLY_ATOMS = no
IMPROPER_CENTRAL = first
INCLUDE_TILT = yes
CHARGES_FROM_CIF = no
#REPLICATE = 2 2 2
FILL_PARAMETERS = yes
KELVIN_TO_KCALMOL = yes
KJ_TO_KCALMOL = no
FACTOR_0.5_HARMONIC = yes
FACTOR_E_DIH_IMP = no
Q_FROM_CHARGETRANSFER = no

CONNECTIVITY_TYPES:
PAIR_STYLE: lj/cut/coul/long
BOND_STYLE: harmonic
ANGLE_STYLE: harmonic
DIHEDRAL_STYLE: harmonic
IMPROPER_STYLE: cvff

BONDS:
a1  a2  P1               P2
Zn1 O2  0.0              1.92
Zn1 O1  0.0              1.94
C3  H1  366001.13136396  0.95
C3  C3  483413.91047488  1.36
C2  C3  483413.91047488  1.36
C1  C2  353750.919316375 1.42
O2  C1  543840.64928424  1.25

ANGLES:
a1  a2  a3  P1               P2
C1  C2  C3  34926.5543205787 120.0
C2  C3  H1  37263.15559911   120.0
C3  C3  H1  37263.15559911   120.0
C3  C2  C3  90640.10821404   120.0
C3  C3  C2  90640.10821404   120.0
O2  C1  O2  135960.162321060 130.0
O2  C1  C2  54882.4848123699 115.0

PDIHEDRALS:
a1  a2  a3  a4 P1              P2   P3
O2  C1  C2  C3 1258.890391861  -1    2
C1  C2  C3  H1 1510.668470234  -1    2
C1  C2  C3  C3 1510.668470234  -1    2
H1  C3  C3  H1 1510.668470234  -1    2
C2  C3  C3  H1 1510.668470234  -1    2
C2  C3  C3  C2 1510.668470234  -1    2
H1  C3  C2  C3 1510.668470234  -1    2
C3  C2  C3  C3 1510.668470234  -1    2

IDIHEDRALS:
a1  a2  a3  a4  P1              P2  P3
C1  O2  O2  C2  5035.561567446  -1   2
C2  C3  C3  C1  5035.561567446  -1   2
C3  C3  C2  H1  186.3157779955  -1   2
"""

### Sahuaro input parser tests ###

def test_parse_input_full_valid_data():
    """
    General input structure and count tests.
    """
    # Using the IO package to deal with string data
    stream = io.StringIO(EXAMPLE_SAHUARO_INPUT_DATA)
    data = parse_input(stream, compat=False)

    # Sanity check to see if the function returns a dictionary as expected
    # and the section headers in the input file exist as top-level keys of
    # the nested dictionary.
    assert isinstance(data, dict)
    expected_keys = [
        'PATHS', 'ATOMS', 'BONDS', 'ANGLES', 'PDIHEDRALS',
        'IDIHEDRALS', 'CONSIDERATIONS', 'CONNECTIVITY_TYPES'
    ]
    for key in expected_keys:
        assert key in data

    # Assertions for specific sections of the input file
    # PATHS section
    assert data['PATHS']['CIF'] == 'irmof1.cif'

    # ATOMS section
    assert len(data['ATOMS']) == 7
    h1_atom = data['ATOMS'][-1] # The atom defined last
    assert h1_atom['label'] == 'H1'
    assert h1_atom['charge'] == 0.15
    assert h1_atom['mass'] == 1.00794

    # CONSIDERATIONS section
    assert data['CONSIDERATIONS']['IMPROPER_CENTRAL'] == 'first'
    assert len(data['CONSIDERATIONS']) == 11

    # CONNECTIVITY_TYPES section
    assert data['CONNECTIVITY_TYPES']['IMPROPER_STYLE'] == 'cvff'
    assert len(data['CONNECTIVITY_TYPES']) == 5

    # BONDS
    assert len(data['BONDS']) == 7
    last_bond = data['BONDS'][-1]
    assert last_bond['a1'] == 'O2'
    assert last_bond['a2'] == 'C1'
    assert last_bond['P2'] == 1.25

    # ANGLES
    assert len(data['ANGLES']) == 7

    # PDIHEDRALS
    assert len(data['PDIHEDRALS']) == 8
    first_pdihedral = data['PDIHEDRALS'][0]
    assert first_pdihedral['a1'] == 'O2'
    assert first_pdihedral['a4'] == 'C3'
    assert first_pdihedral['P2'] == -1

    # IDIHEDRALS
    assert len(data['IDIHEDRALS']) == 3

def test_compatibility_mode():
    """
    Tests that the compat=True flag correctly reformats the full data.
    """
    stream = io.StringIO(EXAMPLE_SAHUARO_INPUT_DATA)
    data = parse_input(stream, compat=True)

    # Check for the special 'compat' keys
    assert 'PADEF' in data
    assert 'FRAMEWORK' in data

    # Check PADEF section
    assert len(data['PADEF']) == 7 # Number of defined atoms
    assert 'C3' in data['PADEF']
    assert data['PADEF']['C3']['mass'] == 12.0107
    assert data['PADEF']['C3']['type'] == 'C'

    # Check FRAMEWORK section 
    framework = data['FRAMEWORK']
    assert len(framework['bonds']) == 7
    assert len(framework['bends']) == 7
    assert len(framework['tors']) == 8
    assert len(framework['itors']) == 3

    # Check specific atom connection combinations
    assert framework['bonds'][0]['combo'] == ['Zn1', 'O2']
    assert framework['bends'][0]['combo'] == ['C1', 'C2', 'C3']
    assert framework['itors'][-1]['combo'] == ['C3', 'C3', 'C2', 'H1']

def test_malformed_line_raises_error():
    """
    Tests that a malformed line in a key=value section raises ParserError.
    """
    incorrect_path_input = "PATHS:\nCIF irmof1.cif\n" # Missing the '='
    stream = io.StringIO(incorrect_path_input)

    # pytest.raises checks that the code inside the 'with' block
    # throws the specified exception. The test fails if it doesn't.
    with pytest.raises(ParserError, match="Malformed PATHS line 2"):
        parse_input(stream)


def test_malformed_connectivity_raises_error():
    """
    Tests that a malformed line in a key:value section raises ParserError.
    """
    incorrect_connectivity_input = "CONNECTIVITY_TYPES:\nBOND_STYLE harmonic\n" # Missing the ':'
    stream = io.StringIO(incorrect_connectivity_input)

    with pytest.raises(ParserError, match="Malformed CONNECTIVITY_TYPES line 2"):
        parse_input(stream)