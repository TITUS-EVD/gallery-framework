#!/usr/env/bin python

''' Helper classes to convert between QWidgets and pyqtgraph '''

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg


class RectItem(pg.GraphicsObject):
    def __init__(self, rect, parent=None, lc='w', fc='k'):
        super().__init__(parent)
        self._rect = rect
        self.picture = QtGui.QPicture()
        self._lc = lc
        self._fc = fc
        self._generate_picture()

    @property
    def fc(self):
        return self._fc

    @property
    def lc(self):
        return self._lc

    @property
    def rect(self):
        return self._rect

    def _generate_picture(self):
        painter = QtGui.QPainter(self.picture)
        painter.setPen(pg.mkPen(self.lc))
        painter.setBrush(pg.mkBrush(self.fc))
        painter.drawRect(self.rect)
        painter.end()

    def paint(self, painter, option, widget=None):
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())

