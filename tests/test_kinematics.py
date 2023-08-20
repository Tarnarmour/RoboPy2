import pytest

from robopy2.kinematics import *


def test_SerialArm_constructor_length_exception():
    # check that the SerialArm constructor catches errors where jt or qlimits don't match number of geometries
    dh = [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]]

    try:
        SerialArm(dh)
    except ValueError:
        assert False, "default constructor failed"

    with pytest.raises(ValueError) as exception_out:
        SerialArm(dh, ['r', 'r'])
    assert(str(exception_out.value) == "'jt' not the same length as number of joints!")

    try:
        SerialArm(dh, ['r', 'r', 'p'])
    except ValueError:
        assert False, "correct number of joints failed"

    with pytest.raises(ValueError) as exception_out:
        SerialArm(dh, qlimits=[(-1, 1)])
    assert(str(exception_out.value) == "'qlimits' not the same length as number of joints!")

    try:
        SerialArm(dh, qlimits=((-1, 1), (-1, 1), (-1, 1)))
    except ValueError:
        assert False, "correct number of qlimits failed"


def test_SerialArm_constructor_qlimits_exception():
    dh = [[0, 0, 1, 0]]

    with pytest.raises(ValueError) as exception_out:
        SerialArm(dh, qlimits=((1, -1),))
    assert str(exception_out.value) == "qlimit at index 0 has improper value (lower bound must be <= 0, upper bound must be >= 0"


def test_SerialArm_constructor_base_and_tool_exception():
    dh = [[0, 0, 1, 0], [0, 0, 1, 0]]

    base = np.eye(4)
    tool = np.eye(4)

    try:
        SerialArm(dh, base=base, tool=tool)
    except ValueError:
        assert False, "proper tool or base argument raised exception"

    with pytest.raises(ValueError) as exception_out:
        SerialArm(dh, base=np.eye(6))
    assert str(exception_out.value) == "Invalid 'base' argument, must be 4 x 4 homogeneous transform"

    with pytest.raises(ValueError) as exception_out:
        SerialArm(dh, tool=np.eye(3))
    assert str(exception_out.value) == "Invalid 'tool' argument, must be 4 x 4 homogeneous transform"

