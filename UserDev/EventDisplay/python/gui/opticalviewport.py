
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
import math


class opticalviewport(pg.GraphicsLayoutWidget):

  def __init__(self, geometry, plane=-1):
    super(opticalviewport, self).__init__(border=None)

    self._geometry = geometry

    # add a view box, which is a widget that allows an image to be shown
    self._view = self.addViewBox(border=None)
    # add an image item which handles drawing (and refreshing) the image
    # self._item = pg.ImageItem(useOpenGL=True)
    # self._item._setPen((0,0,0))
    # self._view.addItem(self._item)

    self.init_geometry()

    # self.setMaximumHeight(200)

    self.setMinimumHeight(400)

    self.setBackground('w')

    self.setRenderHints(QtGui.QPainter.Antialiasing)


  def init_geometry(self):

    opdets_x, opdets_y, opdets_z = self._geometry.opdetLoc()
    opdets_name = self._geometry.opdetName()
    diameter = self._geometry.opdetRadius() * 2

    self._opdet_circles = []
    for d in range(0, len(opdets_x)):
        # print('Adding opdet', opdets_x[d], opdets_y[d], diameter, diameter)
        self._opdet_circles.append(QtGui.QGraphicsEllipseItem(opdets_z[d], opdets_y[d], diameter, diameter))
        
        if opdets_name[d] == 'pmt':
            self._opdet_circles[d].setPen(pg.mkPen('r'))
        if opdets_name[d] == 'barepmt':
            self._opdet_circles[d].setPen(pg.mkPen('b'))

        if opdets_x[d] < 20 and (opdets_name[d] == 'pmt' or opdets_name[d] == 'barepmt'):
            self._view.addItem(self._opdet_circles[d])

    
