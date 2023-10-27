from importlib import import_module
import numpy as np
import time

from .kinematics import SerialArm

import pyqtgraph.opengl as gl
import PyQt5.QtWidgets as QtWidgets


# PyQt5 = None
# gl = None
# graphics_loaded = False
# QtWidgets = None

def lazy_graphics_import():
    # global graphics_loaded
    # global gl
    # global PyQt5
    # global QtWidgets
    # if not graphics_loaded:
    #     graphics_loaded = True
    #     PyQt5 = import_module('PyQt5')
    #     gl = import_module('pyqtgraph.opengl')
    #     QtWidgets = import_module('PyQt5.QtWidgets')
    pass


class VizScene:

    def __init__(self, grid=gl.GLGridItem()):
        # lazy import of graphics modules
        lazy_graphics_import()

        # setup app
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])

        self.main_window = QtWidgets.QMainWindow()
        self.main_window.setWindowTitle('robopy2')
        self.main_window.resize(960, 1080)

        self.central_widget = QtWidgets.QWidget()
        self.main_window.setCentralWidget(self.central_widget)

        self.main_layout = QtWidgets.QHBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        self.joint = None
        self.setup_pyqtgraph_viewer(grid)
        self.mesh_items = []

        self.main_window.show()

    def update(self, timeout=0.0):
        self.app.processEvents()
        if timeout > 0.0:
            time.sleep(timeout)

    def hold(self):
        while self.main_window.isVisible():
            self.app.processEvents()
            time.sleep(0.016)

    def setup_pyqtgraph_viewer(self, grid):
        self.glview = gl.GLViewWidget()
        self.glview.opts['distance'] = 40
        self.main_layout.addWidget(self.glview)
        if grid:
            self.glview.addItem(grid)

    def add_mesh_object(self, mesh_object):
        self.glview.addItem(mesh_object.get_mesh_item())

    def remove_mesh_object(self, mesh_object):
        self.glview.removeItem(mesh_object.get_mesh_item())
        for i, item in self.mesh_items:
            if item == mesh_object:
                self.mesh_items.pop(i)

    def remove_all_objects(self):
        for i, item in self.mesh_items:
            self.glview.removeItem(item.get_mesh_item())


class MeshObject:
    def __init__(self):
        self.mesh_item = None
        self.mesh_data = None

    def get_mesh_item(self):
        return self.mesh_item

    def update(self, *args):
        raise NotImplemented('`update` function not implemented')

    def _set_mesh_data(self, *args):
        raise NotImplemented('`_set_mesh_data` function not implemented')


class RotaryJoint(MeshObject):
    def __init__(self, scale=1.0, radius=1.0, height=1.0, num_polygon=8, color=(0.1, 0.5, 1.0, 1.0)):
        super(RotaryJoint).__init__()
        self._set_mesh_data(scale, radius, height, num_polygon, color)
        self.mesh_item = gl.GLMeshItem(meshdata=self.mesh_data,
                                       smooth=False,
                                       computeNormals=False,
                                       drawEdges=True,
                                       edgeColor=(0, 0, 0, 1))

    def _set_mesh_data(self, scale, radius, height, num_polygon, color):
        # Make points
        angles = np.linspace(0, 2 * np.pi, num_polygon, endpoint=False)
        points = np.asarray([[radius * np.cos(theta), radius * np.sin(theta), height] for theta in angles] +
                            [[radius * np.cos(theta), radius * np.sin(theta), -height] for theta in angles] +
                            [[0., 0., height], [0, 0, -height]]) * scale

        # Top and bottom
        top_mesh = [[i, (i + 1) % num_polygon, 2 * num_polygon] for i in range(num_polygon)]
        bottom_mesh = [[i + num_polygon, (i + 1) % num_polygon + num_polygon, 2 * num_polygon + 1] for i in range(num_polygon)]

        # Side Meshes
        top_side_mesh = [[i, (i + 1) % num_polygon, i + num_polygon] for i in range(num_polygon)]
        bottom_side_mesh = [[i + num_polygon, (i + 1) % num_polygon + num_polygon, (i + 1) % num_polygon] for i in
                            range(num_polygon)]

        # Combine to form faces
        faces = np.asarray(top_mesh + bottom_mesh + top_side_mesh + bottom_side_mesh)

        # Get mesh colors
        colors = np.full((faces.shape[0], 4), color)
        self.mesh_data = gl.MeshData(vertexes=points, faces=faces, faceColors=colors)

    def update(self, T):
        self.mesh_item.setTransform(T)


class RigidLink(MeshObject):
    def __init__(self, width=0.5, length=1.0, num_polygon=4, color=(0.6, 0.6, 0.6, 1.0)):
        super(RigidLink).__init__()
        self._set_mesh_data(width, length, num_polygon, color)
        self.mesh_item = gl.GLMeshItem(meshdata=self.mesh_data,
                                       smooth=False,
                                       computeNormals=False,
                                       drawEdges=True,
                                       edgeColor=(0, 0, 0, 1))

    def _set_mesh_data(self, width, length, num_polygon, color):
        # Make points
        angles = np.linspace(0, 2 * np.pi, num_polygon, endpoint=False)

        points = np.asarray([[length * 0.5, 0.5 * width * np.cos(theta), 0.5 * width * np.sin(theta)] for theta in angles] +
                            [[-length * 0.5, 0.5 * width * np.cos(theta), 0.5 * width * np.sin(theta)] for theta in angles])

        # Side Meshes
        top_side_mesh = [[i, (i + 1) % num_polygon, i + num_polygon] for i in range(num_polygon)]
        bottom_side_mesh = [[i + num_polygon, (i + 1) % num_polygon + num_polygon, (i + 1) % num_polygon] for i in
                            range(num_polygon)]

        # Combine to form faces
        faces = np.asarray(top_side_mesh + bottom_side_mesh)

        # Get mesh colors
        colors = np.full((faces.shape[0], 4), color)
        self.mesh_data = gl.MeshData(vertexes=points, faces=faces, faceColors=colors)

    def update(self, T):
        self.mesh_item.setTransform(T)


class SerialArmMeshObject(MeshObject):
    def __init__(self, arm: SerialArm, scale=1.0, joint_radius=0.2, joint_height=0.4, joint_colors=(0.1, 0.5, 0.6, 1.0), link_width=0.15, link_colors=(0.6, 0.6, 0.6, 1.0)):
        super(SerialArmMeshObject).__init__()
        self.arm = arm
        self.joints = []
        self.links = []
        for i in range(self.arm.n):
            self.joints.append(RotaryJoint(scale, joint_radius, joint_height, 8, joint_colors))
            if i == 0:
                previous_transform = np.eye(4)
            else:
                previous_transform = arm.geometry[i - 1]
            current_transform = arm.geometry[i]
            length = np.linalg.norm(previous_transform[0:3, 3] - current_transform[0:3, 3])
            self.links.append(RigidLink(length, link_width * scale, 4, link_colors))


