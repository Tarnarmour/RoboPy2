import pytest

from robopy2.transforms import *


def test_rot_and_trot_correct_values():
    th = np.pi / 6

    Rx = rotx(th)
    Ry = roty(th)
    Rz = rotz(th)

    Tx = trotx(th)
    Ty = troty(th)
    Tz = trotz(th)

    assert(np.all(np.isclose(Rx, np.array(
        [[1.0, 0.0, 0.0],
         [0.0, np.cos(th), -np.sin(th)],
         [0.0, np.sin(th), np.cos(th)]]
    ))))
    assert(np.all(np.isclose(Ry, np.array(
        [[np.cos(th), 0.0, np.sin(th)],
         [0.0, 1.0, -np.sin(th)],
         [-np.sin(th), 0.0, np.cos(th)]]
    ))))
    assert(np.all(np.isclose(Rz, np.array(
        [[np.cos(th), -np.sin(th), 0.0],
         [np.sin(th), np.cos(th), 0.0],
         [0.0, 0.0, 1.0]]
    ))))

    assert (np.all(np.isclose(Tx, np.array(
        [[1.0, 0.0, 0.0, 0.0],
         [0.0, np.cos(th), -np.sin(th), 0.0],
         [0.0, np.sin(th), np.cos(th), 0.0],
         [0.0, 0.0, 0.0, 1.0]]
    ))))
    assert (np.all(np.isclose(Ty, np.array(
        [[np.cos(th), 0.0, np.sin(th), 0.0],
         [0.0, 1.0, -np.sin(th), 0.0],
         [-np.sin(th), 0.0, np.cos(th), 0.0],
         [0.0, 0.0, 0.0, 1.0]]
    ))))
    assert (np.all(np.isclose(Tz, np.array(
        [[np.cos(th), -np.sin(th), 0.0, 0.0],
         [np.sin(th), np.cos(th), 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]]
    ))))


def test_transl_correct_values():
    x = 1.56
    y = 0.42
    z = -0.99

    xyz = (x, y, z)

    Tx = translx(x)
    Ty = transly(y)
    Tz = translz(z)
    T = transl(xyz)

    assert(np.all(np.isclose(
        Tx,
        np.array(
            [[1.0, 0.0, 0.0, x],
             [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]]
        )
    )))

    assert (np.all(np.isclose(
        Ty,
        np.array(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, y],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]]
        )
    )))

    assert (np.all(np.isclose(
        Tz,
        np.array(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, z],
             [0.0, 0.0, 0.0, 1.0]]
        )
    )))

    assert (np.all(np.isclose(
        T,
        np.array(
            [[1.0, 0.0, 0.0, x],
             [0.0, 1.0, 0.0, y],
             [0.0, 0.0, 1.0, z],
             [0.0, 0.0, 0.0, 1.0]]
        )
    )))


def test_se3_correct_values():
    th = np.pi / 6
    x, y, z = 1.2, -234.0, 0.56

    R = np.array([[np.cos(th), 0.0, np.sin(th)],
                  [0.0, 1.0, 0.0],
                  [-np.sin(th), 0.0, np.cos(th)]])

    T = np.array([[np.cos(th), 0.0, np.sin(th), x],
                  [0.0, 1.0, 0.0, y],
                  [-np.sin(th), 0.0, np.cos(th), z],
                  [0.0, 0.0, 0.0, 1.0]])

    assert(np.all(np.isclose(
        se3(R, (x, y, z)), T
    )))

    assert (np.all(np.isclose(
        se3(R, [x, y, z]), T
    )))

    assert (np.all(np.isclose(
        se3(R, np.array([x, y, z])), T
    )))


def test_T2rep_full_case():
    T = np.array(
        [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 16]], dtype=float
    )

    assert(np.all(np.isclose(
        T, T2rep(T, 'full')
    )))

    assert (np.all(np.isclose(
        T, T2rep(T, 'se3')
    )))

    assert (np.all(np.isclose(
        T, T2rep(T, '')
    )))


def test_T2rep_cartesian_case():
    T = np.array(
        [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 16]], dtype=float
    )
    xyz = T[0:3, 3]

    assert(np.all(np.isclose(
        xyz, T2rep(T, 'cart')
    )))

    assert (np.all(np.isclose(
        xyz, T2rep(T, 'xyz')
    )))

    assert (np.all(np.isclose(
        xyz, T2rep(T, 'pos')
    )))

    assert (np.all(np.isclose(
        xyz, T2rep(T, 'cartesian')
    )))

    assert (np.all(np.isclose(
        xyz, T2rep(T, 'position')
    )))

    assert (np.all(np.isclose(
        xyz, T2rep(T, 'location')
    )))

    assert (np.all(np.isclose(
        xyz, T2rep(T, 'loc')
    )))


def test_T2rep_planar_case():
    th = np.pi / 4
    x = 1.234
    y = 9.876
    z = 0.1928
    T = np.array(
        [[np.cos(th), -np.sin(th), 0.0, x],
         [np.sin(th), np.cos(th), 0.0, y],
         [0.0, 0.0, 1.0, z],
         [0.0, 0.0, 0.0, 1.0]], dtype=float
    )
    xyth = np.array([x, y, th])

    assert(np.all(np.isclose(
        xyth, T2rep(T, 'planar')
    )))

    assert (np.all(np.isclose(
        xyth, T2rep(T, 'plane')
    )))

    assert (np.all(np.isclose(
        xyth, T2rep(T, 'xyth')
    )))


def test_T2rep_value_error():
    T = np.eye(4)
    with pytest.raises(ValueError) as exception_info:
        y = T2rep(T, 'thing')
    assert str(exception_info.value) == "Invalid string for rep type: thing"


def test_inv_4x4():
    th = np.pi / 7
    T = np.array(
        [[np.cos(th), -np.sin(th), 0.0, 1],
         [np.sin(th), np.cos(th), 0.0, -3],
         [0.0, 0.0, 1.0, -120],
         [0.0, 0.0, 0.0, 1.0]], dtype=float
    )

    assert(np.all(np.isclose(
        np.eye(4), T @ inv(T)
    )))


    assert (np.all(np.isclose(
        np.eye(4), inv(T) @ T
    )))


def test_inv_3x3():
    th = -np.pi / 7
    R = np.array(
        [[np.cos(th), -np.sin(th), 0.0],
         [np.sin(th), np.cos(th), 0.0],
         [0.0, 0.0, 1.0]], dtype=float
    )

    assert (np.all(np.isclose(
        np.eye(3), R @ inv(R)
    )))

    assert (np.all(np.isclose(
        np.eye(3), inv(R) @ R
    )))
