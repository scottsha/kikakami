import pyvista as pv
import igl

import numpy as np
import pyvista as pv


def sdf_rotated_rectangle(points, center, half_extents, angle):
    """
    Signed distance to a rotated rectangle in 2D.

    points: (N,2)
    center: (2,)
    half_extents: (2,)  # (half_width, half_height)
    angle: rotation in radians
    """
    # translate
    p = points - center

    # rotate into rectangle frame
    c = np.cos(-angle)
    s = np.sin(-angle)
    R = np.array([[c, -s],
                  [s, c]])
    p = p @ R.T

    # distance to axis-aligned box
    d = np.abs(p) - half_extents
    sdf = d.max(axis=1)
    return sdf


def letter_A_sdf_on_mesh(mesh, thickness=0.1, height=1.0, width=0.8, z_thickness=0.2):
    """
    Compute signed distance of a block letter 'A'
    evaluated at the vertices of a PyVista mesh.

    Returns:
        sdf: (N,) numpy array
    """
    pts = mesh.points
    xy = pts[:, :2]
    z = pts[:, 2]

    # Parameters
    leg_length = height
    leg_width = thickness
    cross_height = height * 0.5
    cross_width = width * 0.5

    # Angles for legs
    angle = np.arctan2(height, width / 2)

    # Left leg
    left_center = np.array([-width / 4, height / 2])
    sdf_left = sdf_rotated_rectangle(
        xy,
        center=left_center,
        half_extents=np.array([leg_width, leg_length / 2]),
        angle=angle
    )

    # Right leg
    right_center = np.array([width / 4, height / 2])
    sdf_right = sdf_rotated_rectangle(
        xy,
        center=right_center,
        half_extents=np.array([leg_width, leg_length / 2]),
        angle=-angle
    )

    # Cross bar
    cross_center = np.array([0.0, cross_height])
    sdf_cross = sdf_rotated_rectangle(
        xy,
        center=cross_center,
        half_extents=np.array([cross_width, leg_width]),
        angle=0.0
    )

    # Union of 2D components
    sdf_2d = np.minimum(np.minimum(sdf_left, sdf_right), sdf_cross)

    # Extrude in Z
    dz = np.abs(z) - z_thickness / 2
    outside_z = np.maximum(dz, 0.0)
    inside_z = np.minimum(dz, 0.0)

    sdf_3d = np.sqrt(np.maximum(sdf_2d, 0) ** 2 + outside_z ** 2) + np.minimum(
        np.maximum(sdf_2d, outside_z), 0
    )

    return sdf_3d

mesh = pv.Disc(inner=0, outer=5, r_res=100, c_res=100)

sdf_values = sdf_rotated_rectangle(mesh.points[:,0:2], center=np.array([0,0]), half_extents=np.array([1,1.3]), angle=0)

mesh["A_sdf"] = sdf_values
mesh.plot(scalars="A_sdf", cmap="coolwarm")