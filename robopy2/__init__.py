# Immediately importable functions:
from .kinematics import SerialArm
from .transforms import *
from .visualizations import *

# Package level functions
def nice_printoptions():
    import numpy as np
    np.set_printoptions(precision=4,
                        suppress=True)