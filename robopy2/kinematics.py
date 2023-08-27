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
from functools import lru_cache, wraps
import numba

import numpy as np

from . import transforms


class SerialArm:
    def __init__(self, geometry: Sequence[Any],
                 jt: Sequence[str] = None,
                 qlimits: Sequence[Sequence[Any]] = None,
                 base: NDArray = transforms.EYE4,
                 tool: NDArray = transforms.EYE4):
        """
        SerialArm constructor function
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
                if qlimit[0] >= 0.0 or qlimit[1] <= 0.0:
                    raise ValueError(
                        f"qlimit at index {i} has improper value (lower bound must be <= 0, upper bound must be >= 0")
            self.qlimits = qlimits

        if isinstance(base, np.ndarray) and base.shape == (4, 4):
            self.base = base
        else:
            raise ValueError("Invalid 'base' argument, must be 4 x 4 homogeneous transform")

        if isinstance(tool, np.ndarray) and tool.shape == (4, 4):
            self.tool = tool
        else:
            raise ValueError("Invalid 'tool' argument, must be 4 x 4 homogeneous transform")

        self._fk_atom = lru_cache(maxsize=self.n)(self._fk_atom_raw)
        self._fk = lru_cache(maxsize=min(10 * self.n, self.n ** 2))(self._fk_raw)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_fk_atom']
        del state['_fk']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # reset cache sizes
        self._fk_atom = lru_cache(maxsize=self.n)(self._fk_atom_raw)
        self._fk = lru_cache(maxsize=min(10 * self.n, self.n ** 2))(self._fk_raw)

    @staticmethod
    def _fk(func):
        """
        Defined in __init__ as the lru_cache wrapped version of self._fk_raw
        """
        wraps(func)
        @lru_cache()
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    # @lru_cache(None)
    def _fk_raw(self, q: tuple[float],
            index: tuple[int, int],
            base: bool,
            tool: bool) -> NDArray:

        T = transforms.EYE4
        for i in range(index[0], index[1]):
            T = T @ self._fk_atom(q[i], i)

        if base and index[0] == 0:
            T = self.base @ T
        elif tool and index[1] == self.n:
            T = T @ self.tool

        return T

    @staticmethod
    def _fk_atom(func):
        """
        Defined in __init__ as the lru_cache wrapped version of self._fk_atom_raw
        """
        wraps(func)
        @lru_cache()
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    # @lru_cache(None)
    def _fk_atom_raw(self, q: float, index: int):
        if self.jt[index] == 'r':
            T = transforms.trotz(q) @ self.geometry[index]
        elif self.jt[index] == 'p':
            T = transforms.translz(q) @ self.geometry[index]
        else:
            T = transforms.EYE4

        return T

    def fk(self, q: Sequence[float],
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
            index = tuple(index)
            if index[1] < index[0] or index[1] > self.n or index[0] < 0:
                raise ValueError(f"Invalid index {index} to fk function!")

        T = self._fk(tuple(q), index, base, tool)

        return transforms.T2rep(T, rep)

    # @lru_cache(maxsize=16)
    def _jacob(self, q: tuple[float],
               index: tuple[int, int],
               base: bool, tool: bool) -> NDArray:

        Te = self._fk(q, index, base, tool)
        pe = Te[0:3, 3]
        J = np.zeros((6, index[1] - index[0]))

        for i in range(index[0], index[1]):
            Ti = self._fk(q, (index[0], i), base, tool)
            zaxis = Ti[0:3, 2]
            if self.jt[i] == 'r':
                p = pe - Ti[0:3, 3]
                J[0:3, i - index[0]] = np.cross(zaxis, p)
                J[3:6, i - index[0]] = zaxis
            elif self.jt[i] == 'p':
                J[0:3, i - index[0]] = zaxis
            else:
                pass

        return J

    def jacob(self, q: Sequence[float],
              index: Union[tuple[int, int], int] = None,
              base: bool = False,
              tool: bool = False,
              rep: str = 'full'):

        q = tuple(q)

        if index is None:
            index = (0, self.n)
        elif isinstance(index, int):
            if index < 0 or index > self.n:
                raise ValueError(f"Invalid index {index} to jacob function!")
            index = (0, index)
        else:
            index = tuple(index)
            if index[1] < index[0] or index[1] > self.n or index[0] < 0:
                raise ValueError(f"Invalid index {index} to jacob function!")

        J = self._jacob(q, index, base, tool)

        return SerialArm.J2rep(J, rep)

    @staticmethod
    def J2rep(J: NDArray, rep: str):
        rep = rep.lower()
        if rep in ('full', 'se3', ''):
            return J
        elif rep in ('cart', 'xyz'):
            return J[0:3]
        elif rep in ('planar', 'xyth'):
            return J[[0, 1, 5]]
        else:
            raise ValueError(f"Invalid string for rep type: {rep}")
