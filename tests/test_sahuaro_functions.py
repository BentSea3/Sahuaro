import pytest
import numpy as np
# Importing the functions from Sahuaro
from sahuaro.Sahuaro import (
    fix_diff,
    dist2,
    calc_bonds,
    classif_angle,
    neighbor_combinations
)

### Learning with basic Sahuaro function tests ###

def test_fix_diff():
    """
    Testing the fix_diff function for PBC.
    """
    cell = np.array([10.0, 10.0, 10.0])

    # Test 1: Nno wrapping needed
    diff = np.array([2.0, 3.0, -4.0])
    np.testing.assert_allclose(fix_diff(diff, cell), diff)

    # Test 2: Wrapping in the 'positive' direction
    diff_pos = np.array([6.0, 7.0, 8.0])
    expected_pos = np.array([-4.0, -3.0, -2.0])
    np.testing.assert_allclose(fix_diff(diff_pos, cell), expected_pos)

    # Test 3: Wrappping in the 'negative' direction
    diff_neg = np.array([-6.0, -7.0, -8.0])
    expected_neg = np.array([4.0, 3.0, 2.0])
    np.testing.assert_allclose(fix_diff(diff_neg, cell), expected_neg)


def test_dist2():
    """
    Tests the dist2 squared distance calculation.
    """
    cell = np.array([10.0, 10.0, 10.0])

    # Simple case without periodic boundaries consideration
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([4.0, 6.0, 8.0])
    # Expected squared distance: (3^2 + 4^2 + 5^2) = 9 + 16 + 25 = 50
    assert dist2(v1, v2, cell) == pytest.approx(50.0)

    # Case with periodic boundaries
    v3 = np.array([1.0, 1.0, 1.0])
    v4 = np.array([9.0, 9.0, 9.0])
    # Difference is [-8, -8, -8], which wraps to [2, 2, 2]
    # Expected squared distance: (2^2 + 2^2 + 2^2) = 12
    assert dist2(v3, v4, cell) == pytest.approx(12.0)


def test_calc_bonds():
    """
    Tests the bond calculation and identification function.
    """
    # Setup the arguments the function actually expects
    pos = np.array([
        [0.0, 0.0, 0.0],  # Atom 0 (type 0, label A)
        [1.5, 0.0, 0.0],  # Atom 1 (type 0, label A)
        [3.5, 0.0, 0.0],  # Atom 2 (type 1, label B)
        [5.5, 0.0, 0.0],  # Atom 3 (type 1, label B)
    ])
    pal = np.array([0, 0, 1, 1]) # Maps atom index to type index
    N = 4
    # valid_bonds[type_index] = {set of valid type_indices}
    valid_bonds = {0: {0, 1}, 1: {0, 1}}
    # A-A threshold = 2.6. B-B = 3.12. A-B = 2.86
    threshold = np.array([
        [2.60, 2.86],
        [2.86, 3.12]
    ])
    box = np.array([10, 10, 10])

    # Expected bonds (squared distances):
    # 0-1: dist^2 = 2.25 (< 2.6^2=6.76) -> bond
    # 1-2: dist^2 = 4.0  (< 2.86^2=8.18) -> bond
    # 2-3: dist^2 = 4.0  (< 3.12^2=9.73) -> bond
    bonds, dists = calc_bonds(pos, pal, N, valid_bonds, threshold, box)

    # Convert list of tuples to a set for easier comparison
    bond_set = {tuple(sorted(b)) for b in bonds}

    assert len(bond_set) == 3
    assert (0, 1) in bond_set
    assert (1, 2) in bond_set
    assert (2, 3) in bond_set


def test_classif_angle():
    """
    Tests the classification of atom types for angle lookup.
    """
    # The function nneeds a triplet of atom LABELS, an angle,
    # and the dictionary it's building.
    triplet_labels = ('C', 'O', 'C')
    angle = 120.0
    trip_classif_angle = {} # empty dictionary

    # Call the function
    index, updated_dict = classif_angle(triplet_labels, angle, trip_classif_angle)

    # Assertions
    assert index == 0 # It should be the first type found for this combo combination
    # The key is the sorted outer atoms + the center atom
    expected_key = ('C', 'C', 'O')
    assert expected_key in updated_dict
    assert updated_dict[expected_key] == [120.0]

    # Test finding an existing angle
    index2, updated_dict2 = classif_angle(triplet_labels, 121.0, updated_dict)
    assert index2 == 0 # Should match the existing angle (within 2.0 degrees)
    assert updated_dict2[expected_key] == [120.0] # Should not add the new angle

    # Test adding a new, different angle
    index3, updated_dict3 = classif_angle(triplet_labels, 130.0, updated_dict2)
    assert index3 == 1 # Should be a new type for this combo
    assert updated_dict3[expected_key] == [120.0, 130.0]


def test_neighbor_combinations():
    """
    Tests the generation of unique neighbor combinations for impropers.
    """
    # Setup: The function needs the full connectivity list (con_list)
    # con_list[i] = list of neighbors for atom i
    con_list = [
        [1],            # Atom 0 is bonded to 1
        [0, 2, 3, 4],   # Atom 1 is bonded to 0, 2, 3, 4
        [1],            # Atom 2 is bonded to 1
        [1],            # Atom 3 is bonded to 1
        [1]             # Atom 4 is bonded to 1
    ]

    # Test with central atom first
    central_first = True
    # For central atom 1, neighbors are 0, 2, 3, 4. Combinations of 3 are:
    # (0,2,3), (0,2,4), (0,3,4), (2,3,4)
    result = neighbor_combinations(1, central_first, con_list)
    # Convert to a set of sorted tuples for comparison
    result_set = {tuple(sorted(r)) for r in result}

    assert len(result_set) == 4
    assert tuple(sorted([1, 0, 2, 3])) in result_set
    assert tuple(sorted([1, 0, 2, 4])) in result_set
    assert tuple(sorted([1, 0, 3, 4])) in result_set
    assert tuple(sorted([1, 2, 3, 4])) in result_set

    # Test with not enough neighbors
    assert neighbor_combinations(0, central_first, con_list) == []
