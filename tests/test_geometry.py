import numpy as np
import pytest

from sahuaro.geometry import dist2, bond_angle


# Check distances for atoms inside the box and across periodic boundaries.
@pytest.mark.parametrize(
    "x1, x2, expected",
    [
        (0.5, 9.5, 1.0),
        (9.5, 0.5, 1.0),
        (1.0, 4.0, 3.0),
    ],
)
def test_periodic_distance(x1, x2, expected):
    box = np.array([10.0, 10.0, 10.0])

    v1 = np.array([x1, 0.0, 0.0])
    v2 = np.array([x2, 0.0, 0.0])

    assert dist2(v1, v2, box) == pytest.approx(expected)


def test_angle_across_periodic_boundaries():
    box = np.array([10.0, 10.0, 10.0])

    # The central atom lies between two atoms pointing in opposite directions.
    pos = np.array([
        [1.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [9.0, 0.0, 0.0],
    ])

    assert bond_angle((0, 1, 2), pos, box) == pytest.approx(180.0)


def test_perpendicular_bonds_across_boundary():
    box = np.array([10.0, 10.0, 10.0])

    # The first bond crosses the x boundary; the second points along y.
    pos = np.array([
        [9.5, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 1.5, 0.0],
    ])

    assert bond_angle((0, 1, 2), pos, box) == pytest.approx(90.0)
