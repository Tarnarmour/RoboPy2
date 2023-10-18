import pickle
from robopy2.kinematics import SerialArm
import numpy as np
import pytest


@pytest.fixture
def setup_pickled_arm():
    dh = [[0, 0, 1, 0], [0, 0, 1, 0]]
    fresh_arm = SerialArm(dh)
    pickled_arm = pickle.dumps(fresh_arm)
    depickled_arm = pickle.loads(pickled_arm)
    return fresh_arm, pickled_arm, depickled_arm


def test_pickle_serial_arm(setup_pickled_arm):
    arm, pickled_arm, depickled_arm = setup_pickled_arm
    assert(isinstance(depickled_arm, SerialArm))
    assert(np.all([np.isclose(x, y) for x, y in zip(arm.geometry, depickled_arm.geometry)]))

def test_pickle_serial_arm_functions(setup_pickled_arm):
    arm, pickled_arm, depickled_arm = setup_pickled_arm
    q = np.random.random((arm.n,)) * 2 * np.pi
    assert(np.all(np.isclose(arm.fk(q), depickled_arm.fk(q))))
    assert(np.all(np.isclose(arm.jacob(q), depickled_arm.jacob(q))))

