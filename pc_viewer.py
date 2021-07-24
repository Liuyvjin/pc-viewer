import sys
import os.path as osp
from utils.pc_util import draw_pointcloud_rgb, rot_angle_axis, read_ply
import numpy as np
from PyQt5.QtWidgets import ( QStatusBar, QVBoxLayout, QWidget, QHBoxLayout,
                             QLabel, QApplication)
from PyQt5 import QtGui as QG
from PyQt5.QtCore import Qt

BASE_DIR = osp.dirname(osp.abspath(__file__))


class PC_Viewer(QWidget):

    def __init__(   self,
                    pointcloud,
                    rgb         =   None,
                    alpha       =   1,
                    diameter    =   25,
                    bg_color    =   [255, 255, 255] ):

        super().__init__()
        self.pc = pointcloud
        self.alpha = alpha
        self.diameter = diameter
        self.rgb = [99, 184, 255] if rgb is None else rgb
        self.bg_color = bg_color
        self.canva_size = 700
        self.paint_size = 300
        self.mouse_x0 = 0
        self.mouse_y0 = 0
        self.rot_start = np.eye(3)
        self.rot_relative = np.eye(3)
        self.scale = 1
        self.initUI()
        self.update_img()
        self.update_msg = self.status_bar.showMessage


    def initUI(self):
        # background color
        pal = QG.QPalette()
        pal.setColor(self.backgroundRole(), QG.QColor(*self.bg_color))
        self.setPalette(pal)
        # widgets
        self.pc_viewer = QLabel(self)
        self.pc_viewer.setMinimumSize(20, 20)
        self.status_bar = QStatusBar(self)
        self.status_bar.setFixedHeight(20)
        # layout
        v_layout = QVBoxLayout(self)
        v_layout.setContentsMargins(0, 0, 0, 0)
        v_layout.addWidget(self.pc_viewer)
        v_layout.addWidget(self.status_bar)
        self.setLayout(v_layout)

        self.pc_viewer.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(False)

        self.move(300, 100)
        self.setWindowTitle('Pointcloud Viewer')
        self.show()


    def mousePressEvent(self, pos) -> None:
        """ save mouse press position """
        if pos.y() > self.height()-20:
            return
        self.mouse_x0 = pos.x()
        self.mouse_y0 = pos.y()
        self.rot_start = np.dot(self.rot_relative, self.rot_start)


    def mouseMoveEvent(self, pos):
        if pos.y() > self.height()-20:
            return
        self.update_msg('x={:d}, y={:d}'.format(pos.x(),  pos.y()))

        dx = pos.x() - self.mouse_x0
        dy = pos.y() - self.mouse_y0
        axis = np.array([dx, -dy, 0])
        angle = np.sqrt(dx**2 + dy**2) / 200
        self.rot_relative = rot_angle_axis(angle, axis)

        self.update_img()

    def wheelEvent(self, event: QG.QWheelEvent) -> None:
        if event.angleDelta().y()>0:
            self.scale += 0.05
        else:
            self.scale -= 0.05
        self.scale = max(self.scale, 0.3)
        self.update_img()


    def update_img(self):
        diameter = int(self.scale*self.diameter)
        canva_size = int(self.scale*self.canva_size)
        paint_size = int(self.scale*self.paint_size)
        rot = np.dot(self.rot_relative, self.rot_start)
        img = draw_pointcloud_rgb(  pointcloud  =   self.pc,
                                    rgb         =   self.rgb,
                                    alpha       =   self.alpha ,
                                    diameter    =   diameter,
                                    rot         =   rot,
                                    canvasSize  =   canva_size,
                                    space       =   paint_size,
                                    back_color  =   self.bg_color   )
        self.img = QG.QPixmap(QG.QImage(img.data, img.shape[1], img.shape[0],
                            img.shape[1]*3, QG.QImage.Format_RGB888))
        self.pc_viewer.setPixmap(self.img)



if __name__=="__main__":
    app = QApplication(sys.argv)

    # pointcloud = np.random.random((100, 3))
    pointcloud = read_ply('plane.ply')
    exp = PC_Viewer(pointcloud)

    sys.exit(app.exec_())

