from titus.drawables import Drawable

from ROOT import evd
import pyqtgraph as pg
import numpy as np


class OpDetWaveform(Drawable):

    """docstring for opdetwaveform"""

    def __init__(self, gallery_interface, geom):
        super().__init__(gallery_interface)
        self._process = evd.DrawOpDetWaveform(geom.getGeometryCore(),
                                              geom.getDetectorProperties(),
                                              geom.getDetectorClocks())
        self._process.set_n_frames(geom.nOpticalFrames())
        self._process.set_time_offset(geom.opticalOffset())
        self._process.initialize()
        self._process.setInput(self._producer_name)

    def set_producer(self, producer):
        """ override to call setInput instead of setProducer """
        self._producer_name = producer
        if self._process is not None:
            self._process.setInput(self._producer_name)

    def getData(self):
        try:
            return self._process.getArray()
        except:
            return np.ones(shape=(312, 1))
