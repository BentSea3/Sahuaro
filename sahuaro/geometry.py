"""
Geometry calculations used to identify connectivity in periodic systems.
"""

import numpy as np


def dist2(v1, v2, box):
    # Apply the minimum-image convention in each box direction.
    diff = v1 - v2
    diff = diff - box * np.rint(diff / box)
    return np.linalg.norm(diff)


def bond_angle(triplet, pos, box):
    # Calculate the angle between two bonds from the central atom.
    a, i, b = triplet

    v_ia = pos[a] - pos[i]
    v_ib = pos[b] - pos[i]

    # Account for bonds that cross periodic boundaries.
    v_ia -= box * np.rint(v_ia / box)
    v_ib -= box * np.rint(v_ib / box)

    cosine_angle = np.dot(v_ia, v_ib) / (
        np.linalg.norm(v_ia) * np.linalg.norm(v_ib)
    )

    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
