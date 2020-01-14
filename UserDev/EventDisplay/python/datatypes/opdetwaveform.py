from datatypes.database import dataBase
from ROOT import evd
import pyqtgraph as pg
import numpy as np


class opdetwaveform(dataBase):

    """docstring for opdetwaveform"""

    def __init__(self, geom):
        super(opdetwaveform, self).__init__()
        self._n_tpc = geom.nTPCs()
        self._n_plane = geom.nPlanes()
        self._gap = geom.cathodeGap()
        self._process = evd.DrawOpDetWaveform(geom.getGeometryCore(), 
                                              geom.getDetectorProperties(),
                                              geom.getDetectorClocks())
        self._process.set_n_frames(geom.nOpticalFrames())
        self._process.set_time_offset(geom.opticalOffset())
        self._process.initialize()
        self._process.setInput(self._producerName)

    def setProducer(self, producer):
        self._producerName = producer
        if self._process is not None:
            self._process.setInput(self._producerName)

    def getData(self):
        return self._process.getArray()
