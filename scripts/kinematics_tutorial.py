""" Getting started tutorial """

""" First, import robopy2 and numpy """
import robopy2 as rp
import numpy as np

""" Make transforms print more nicely """
rp.nice_printoptions()

""" Now, we can define a robot arm using DH parameters or a list of 4x4 transforms or a combination of both.
These transforms define the geometry connecting subsequent joints on the robot arm. Each joint actuates around
or along the z axis of it's corresponding frame. robopy2 uses the order [d theta a alpha]"""
dh = [[0, 0, 1, np.pi / 2], [0, 0, 1, 0]]  # a two link arm
jt = ['r', 'r']  # both joints are revolute

""" Now make a robot arm using these parameters and the SerialArm class from the kinematics module """
arm = rp.SerialArm(dh, jt)

""" We can calculate the forward kinematics of this arm using the fk function """
q = np.radians([30, 90])  # define a Sequence of joint angles
T = arm.fk(q)  # calculate the forward kinematics
print(T)

""" By default, fk calculates from frame 0 (te frame located at the first joint) to frame n (located at the end
 of the robot). We can get the transform between any pair of frames by specifying the index: """
T = arm.fk(q, index=(1, 2))
print(T)

""" We can also choose to optionally define and include transforms from the world frame to the base, or from
the end of the arm out to a tool frame:"""
base = rp.transl([1.0, 0.0, 1.0])  # translation of [1, 0, 1]
tool = rp.trotx(np.pi / 2)  # rotation about x axis of 90 degrees
arm = rp.SerialArm(dh, jt, base=base, tool=tool)
T = arm.fk(q, base=True, tool=True)
print(T)

""" We can also choose to get the output in a specific format using the rep argument: """
T = arm.fk(q, rep='cart')  # cartesian position only
print(T)

""" This arm could be equivalently defined using full transforms instead of dh parameters: """
geometry = [rp.translx(1.0) @ rp.trotx(np.pi / 2), rp.translx(1.0)]
arm = rp.SerialArm(geometry, jt)
T = arm.fk(q, rep='cart')
print(T)
