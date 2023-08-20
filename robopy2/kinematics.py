"""
Description:
    Modeling for kinematic robot manipulators (e.g. no dynamics), covers forward kinematics, Jacobians, inverse kinematics, etc.

Example:
    arm = SerialArm(dh_parameters, joint_types)
    q = np.radians([30, 180, -60])
    T = arm.fk(q, rep='cartesian')
    q = arm.ik(T)
    J = arm.jacob(q, rep='cartesian')
"""
from typing import Iterable, Any, Sized, Sequence, Union
from numbers import Number
from numpy.typing import NDArray
from functools import lru_cache

import numpy as np

from . import transforms


class SerialArm:
    def __init__(self, geometry: Sequence[Any],
                 jt: Sequence[str] = None,
                 qlimits: Sequence[tuple[Number]] = None,
                 base: NDArray = transforms.EYE4,
                 tool: NDArray = transforms.EYE4):
        """
        SerialArm contructor function
        :param geometry: Sequence[Any] defining the geometry of the arm. geometry[i] should either give the dh parameters (using [d, theta, a, alpha] order) or a 4x4 numpy array representing the transform from joint i to i + 1
        :param jt: Sequence[str] = ('r', ... 'r') joint types, 'r' for revolute 'p' for prismatic e.g. ['r', 'p', 'r']
        :param qlimits: Sequence[tuple[Number]] = ((-2 * pi, 2 * pi) ... (-2 * pi, 2 * pi)) joint angle limits in [(low, high), (low, high)] format, ex. [(-pi, pi), (-pi/2, pi)]
        :param base: ndarray[4, 4] = np.eye(4) transform from world frame to joint 1 frame
        :param tool: ndarray[4, 4] = np.eye(4) transform from joint n frame to tool frame
        """
        self.n = len(geometry)
        self.geometry = []
        for x in geometry:
            if isinstance(x, np.ndarray) and x.shape == (4, 4):
                self.geometry.append(np.asarray(x, dtype=float))
            else:
                self.geometry.append(np.array(
                    [[np.cos(x[1]), -np.sin(x[1]) * np.cos(x[3]), np.sin(x[1]) * np.sin(x[3]), x[2] * np.cos(x[1])],
                     [np.sin(x[1]), np.cos(x[1]) * np.cos(x[3]), -np.cos(x[1]) * np.sin(x[3]), x[2] * np.sin(x[1])],
                     [0, np.sin(x[3]), np.cos(x[3]), x[0]],
                     [0, 0, 0, 1]], dtype=float
                ))
        self.geometry = tuple(self.geometry)

        if jt is None:
            self.jt = tuple(['r' for x in range(self.n)])
        else:
            self.jt = tuple([x for x in jt])
        if len(self.jt) != self.n:
            raise ValueError("'jt' not the same length as number of joints!")

        if qlimits is None:
            self.qlimits = tuple([(-2 * np.pi, 2 * np.pi) if jt == 'r' else (-1.0, 1.0) for jt in self.jt])
        else:
            if len(qlimits) != self.n:
                raise ValueError("'qlimits' not the same length as number of joints!")
            for i, qlimit in enumerate(qlimits):
                if qlimit[0] >= 0 or qlimit[1] <= 0:
                    raise ValueError(f"qlimit at index {i} has improper value (lower bound must be <= 0, upper bound must be >= 0")
            self.qlimits = qlimits

        if isinstance(base, np.ndarray) and base.shape == (4, 4):
            self.base = base
        else:
            raise ValueError("Invalid 'base' argument, must be 4 x 4 homogeneous transform")

        if isinstance(tool, np.ndarray) and tool.shape == (4, 4):
            self.tool = tool
        else:
            raise ValueError("Invalid 'tool' argument, must be 4 x 4 homogeneous transform")

    @lru_cache(maxsize=16)
    def _fk(self, q: tuple[Number],
            index: tuple[int, int],
            base: bool,
            tool: bool) -> NDArray:

        if index[0] == 0 and base:
            T = self.base
        else:
            T = transforms.EYE4
        
        for i in range(index[0], index[1]):
            if self.jt[i] == 'r':
                T = T @ transforms.trotz(q[i])
            elif self.jt[i] == 'p':
                T = T @ transforms.translz(q[i])
            else:
                pass
            T = T @ self.geometry[i]
        
        if index[1] == self.n and tool:
            T = T @ self.tool
        
        return T

    def fk(self, q: Sequence[Number],
          index: Union[Sequence[int], int] = None,
          base: bool = False,
          tool: bool = False,
          rep: str = 'se3') -> np.ndarray:

        if len(q) != self.n:
            raise ValueError("Incorrect number of joint angles for fk function!")

        if index is None:
            index = (0, self.n)
        elif isinstance(index, int):
            if index < 0 or index > self.n:
                raise ValueError(f"Invalid index {index} to fk function!")
            index = (0, index)
        else:
            if index[1] < index[0]:
                raise ValueError(f"Invalid index {index} to fk function!")

        T = self._fk(tuple(q), index, base, tool)

        return transforms.T2rep(T, rep)
