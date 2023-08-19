"""
Description:
    Modeling for kinematic robot manipulators (e.g. no dynamics), covers forward kinematics, Jacobians, inverse kinematics, etc.

Example:
    arm = SerialArm(dh_parameters, joint_types)
    q = np.radians([30, 180, -60])
    T = arm.fk(q, rep='cartesian')
    q = arm.ik(T)
    J = arm.jacob(q, rep='cartesian')

Author:
    John Morrell (Tarnarmour@gmail.com)

Date:
    Created: August 19 2023
    Last Updated: August 19 2023

Version:
    0.0.1

Dependencies:
    - numpy
"""


class SerialArm:
    def __init__(self, dh):
        self.dh = dh
