"""
Description:
    Basic helper functions for representing 3D and 2D transforms and Lie groups

Example:
    T = transl([1, 2, 3])
    q = rpy2quat([pi, pi/2, 0])
"""

import numpy as np


EYE2 = np.eye(2, dtype=float)
EYE3 = np.eye(3, dtype=float)
EYE4 = np.eye(4, dtype=float)


# basic convenience constructor functions
def rotx(th: float):
    return np.array(
        [[1.0, 0.0, 0.0],
         [0.0, np.cos(th), -np.sin(th)],
         [0.0, np.sin(th), np.cos(th)]], dtype=float
    )


def roty(th: float):
    return np.array(
        [[np.cos(th), 0.0, np.sin(th)],
         [0.0, 1.0, -np.sin(th)],
         [-np.sin(th), 0.0, np.cos(th)]], dtype=float
    )


def rotz(th: float):
    return np.array(
        [[np.cos(th), -np.sin(th), 0.0],
         [np.sin(th), np.cos(th), 0.0],
         [0.0, 0.0, 1.0]], dtype=float
    )


def trotx(th: float):
    T = np.eye(4, dtype=float)
    T[0:3, 0:3] = rotx(th)
    return T


def troty(th: float):
    T = np.eye(4, dtype=float)
    T[0:3, 0:3] = roty(th)
    return T


def trotz(th: float):
    T = np.eye(4, dtype=float)
    T[0:3, 0:3] = rotz(th)
    return T


def transl(xyz):
    return np.array([[1.0, 0.0, 0.0, xyz[0]],
                     [0.0, 1.0, 0.0, xyz[1]],
                     [0.0, 0.0, 1.0, xyz[2]],
                     [0.0, 0.0, 0.0, 1.0]], dtype=float)


def translx(x):
    return np.array([[1.0, 0.0, 0.0, x],
                     [0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]], dtype=float)


def transly(y):
    return np.array([[1.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, y],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]], dtype=float)


def translz(z):
    return np.array([[1.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, z],
                     [0.0, 0.0, 0.0, 1.0]], dtype=float)


def se3(R=EYE3, p=np.array([0, 0, 0], dtype=float)):
    T = np.copy(EYE4)
    T[0:3, 0:3] = R
    T[0:3, 3] = p
    return T


def T2rep(T: np.ndarray, rep: str) -> np.ndarray:
    rep = rep.lower()
    if rep in ('se3', 'full', ''):
        return T
    elif rep in ('cart', 'xyz', 'pos', 'cartesian', 'position', 'location', 'loc'):
        return T[0:3, 3]
    else:
        raise ValueError(f"Invalid string for rep type: {rep}")

