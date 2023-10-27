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
import scipy
import numba

import numpy as np

from . import transforms


class SerialArm:
    def __init__(self, geometry: Sequence[Any],
                 jt: Sequence[str] = None,
                 qlimits: Sequence[Sequence[Any]] = None,
                 limit_type: str = 'ignore',
                 base: NDArray = transforms.EYE4,
                 tool: NDArray = transforms.EYE4,
                 q0: Sequence[float] = None):
        """
        SerialArm constructor function
        :param geometry: Sequence[Any] defining the geometry of the arm. geometry[i] should either give the dh parameters (using [d, theta, a, alpha] order) or a 4x4 numpy array representing the transform from joint i to i + 1
        :param jt: Sequence[str] = ('r', ... 'r') joint types, 'r' for revolute 'p' for prismatic e.g. ['r', 'p', 'r']
        :param qlimits: Sequence[tuple[Number]] = ((-2 * pi, 2 * pi) ... (-2 * pi, 2 * pi)) joint angle limits in [(low, high), (low, high)] format, ex. [(-pi, pi), (-pi/2, pi)]
        :param limit_type: str = 'ignore', string for how to treat joint limits. 'ignore', 'clamp', 'exception', 'warn'
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

        if limit_type in ('ignore', 'clamp', 'exception', 'warn'):
            self.limit_type = limit_type
        else:
            raise ValueError(f"Unknown 'limit_type' argument: {limit_type}")
        self._suppress_limit_check = False

        if isinstance(base, np.ndarray) and base.shape == (4, 4):
            self.base = base
        else:
            raise ValueError("Invalid 'base' argument, must be 4 x 4 homogeneous transform")

        if isinstance(tool, np.ndarray) and tool.shape == (4, 4):
            self.tool = tool
        else:
            raise ValueError("Invalid 'tool' argument, must be 4 x 4 homogeneous transform")

        if q0 is None:
            self.q0 = tuple([0.0 for _ in range(self.n)])
        else:
            if len(q0) != self.n:
                raise ValueError(f"Incorrect number of joint angles default joint position q0: {q0}")
            self.q0 = self.check_qlimit(tuple(q0))

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

    def check_qlimit(self, q: tuple[float]) -> tuple[float]:
        if self._suppress_limit_check:
            return q
        if self.limit_type == 'ignore':
            return q
        if self.limit_type == 'clamp':
            return tuple([np.clip(x, lim[0], lim[1]) for x, lim in zip(q, self.qlimits)])
        if self.limit_type == 'exception':
            for x, lim in zip(q, self.qlimits):
                if not lim[0] <= x <= lim[1]:
                    raise ValueError(f'q out of joint limits: {q}')
            return q
        if self.limit_type == 'warn':
            for i, x, lim in enumerate(zip(q, self.qlimits)):
                if not lim[0] <= x <= lim[1]:
                    print(f"Warning: q out of joint limits at index {i}: {q[i]}")
            return q

    def suppress_limit_checks(self, do_suppress):
        self._suppress_limit_check = do_suppress

    @staticmethod
    def _fk(func):
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
            raise ValueError(f"Incorrect number of joint angles for fk function: {q}")

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

        q = self.check_qlimit(tuple(q))

        T = self._fk(q, index, base, tool)

        return transforms.T2rep(T, rep)

    @lru_cache(maxsize=16)
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

        if len(q) != self.n:
            raise ValueError(f"Incorrect number of joint angles for jacob function: {q}")

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

        q = self.check_qlimit(q)
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

    def ik(self, target: Union[np.ndarray, Sequence[float]],
           q0: Sequence[float] = None,
           base: bool = True,
           tool: bool = True,
           eps: float = 1e-3,
           mit: int = 1e3):

        if q0 is None:
            q0 = self.q0
        else:
            if len(q0) != self.n:
                raise ValueError(f"Incorrect number of joint angles for q0 in ik function: {q0}")
            q0 = self.check_qlimit(tuple(q0))

        p = (0., 0., 0.)
        R = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))

        if isinstance(target, np.ndarray):
            if target.shape == (3, 3):  # treat as a rotation matrix
                R = tuple([tuple(r) for r in target])
            elif target.shape == (4, 4):  # treat as a homogeneous transform
                R = tuple([tuple(r) for r in target[0:3, 0:3]])
                p = tuple(target[0:3, 3])
        elif len(target) == 3:  # treat as xyz sequence
            p = tuple(target)
        else:
            raise ValueError(f"Invalid argument 'target' for ik function: {target}")

        return self._ik(p, R, q0, base, tool, eps, mit)

    @lru_cache(maxsize=16)
    def _ik(self, p: tuple[float],
            R: tuple[tuple],
            q0: tuple[float],
            base: bool,
            tool: bool,
            eps: float,
            mit: int):

        if self.limit_type in ('warn', 'ignore'):
            self._suppress_limit_check = True

        q_current = np.asarray(q0)
        p_target = np.asarray(p)
        R_target = np.asarray(R)

        def calculate_error(q):
            T = self._fk(tuple(q), (0, self.n), base, tool)
            p = T[0:3, 3]
            R = T[0:3, 0:3]

            p_error = p_target - p
            w_error = R @ scipy.linalg.logm(R.T @ R_target)[(2, 0, 1), (1, 2, 0)]
            e = np.hstack([p_error, w_error])
            return e

        e = calculate_error(q_current)

        nit = 0

        while np.linalg.norm(e) > eps and nit < mit:
            print(f'Iteration start: {np.linalg.norm(e)}')
            J = self._jacob(tuple(q_current), (0, self.n), base, tool)
            dq = np.linalg.pinv(J) @ e

            q_a = q_current
            e_a = e
            q_b = q_current + dq
            e_b = calculate_error(q_b)
            step_count_forward = 0
            step_count_binary = 0

            print(f'\nStarting foward stepping: {np.linalg.norm(e_b)}')
            while np.linalg.norm(e_b) < np.linalg.norm(e_a) and step_count_forward < 10:
                q_a = q_b
                e_a = e_b
                q_b = q_b + dq
                e_b = calculate_error(q_b)
                dq = dq * 1.0
                step_count_forward += 1
                print(f'{np.linalg.norm(e_b)}')

            if np.linalg.norm(e_b) > np.linalg.norm(e_a):
                q_high = q_b
                e_high = e_b
                q_low = q_a
                e_low = e_a
            else:
                q_low = q_b
                e_low = e_b
                q_high = q_a
                e_high = e_a

            print(f'\nStarting binary search: {np.linalg.norm(e_low), np.linalg.norm(e_high)}')

            while step_count_binary < 10:
                q_prime = 0.5 * q_low + 0.5 * q_high
                e_prime = calculate_error(q_prime)

                if np.linalg.norm(e_prime) < np.linalg.norm(e_low):
                    q_high = q_low
                    e_high = e_low
                    q_low = q_prime
                    e_low = e_prime
                elif np.isclose(np.linalg.norm(e_high), np.linalg.norm(e_low)):
                    break
                else:
                    q_high = q_prime
                    e_high = e_prime
                step_count_binary += 1
                print(f'{np.linalg.norm(e_low), np.linalg.norm(e_high)}')

            q_current = q_low
            e = calculate_error(q_low)
            nit += 1

            print(np.linalg.norm(e), step_count_forward, step_count_binary, e, '\n\n')

        self._suppress_limit_check = False

        return q_current

    def randq(self, shape=None):
        if shape is None:
            qs = np.asarray([np.random.random() * (qlim[1] - qlim[0]) + qlim[0] for qlim in self.qlimits])
        else:
            qs = np.asarray([np.random.random(shape) * (qlim[1] - qlim[0]) + qlim[0] for qlim in self.qlimits])
        return qs