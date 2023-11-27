#/usr/bin/env python3

"""
This module adds the CRT view & associated controls
"""
import pyqtgraph as pg 
from PyQt5 import QtWidgets, QtGui, QtCore

from ..gui.qrangeslider import QRangeSlider
from ..gui.optical_elements import Pmts, Arapucas, _bordercol_

from titus.modules import Module
# from ..gallery_interface import datatypes


class CrtModule(Module):
    def __init__(self, larsoft_module, geom_module):
        super().__init__()
        self._gm = geom_module
        self._lsm = larsoft_module
        self._central_widget = QtWidgets.QWidget()
        self._layout = QtWidgets.QVBoxLayout()
        self._central_widget.setLayout(self._layout)

    def _initialize(self):
        self._gm.geometryChanged.connect(self.init_ui)

    def init_ui(self):
        pass
