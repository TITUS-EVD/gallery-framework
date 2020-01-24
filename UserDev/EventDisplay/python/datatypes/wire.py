from datatypes.database import dataBase
from ROOT import evd
import pyqtgraph as pg
import numpy as np


class wire(dataBase):

    """docstring for wire"""

    def __init__(self):
        super(wire, self).__init__()
        self._process = None
        self._n_tpc = None
        self._n_plane = None
        self._gap = None
        self._plane_mix = {}
        self._plane_flip = []

    def getPlane(self, plane, cryo=0):
        '''
        Returns the array of values for the selected plane.
        The values are stored in a 2d array, containing the
        wires on axis 0, and the waveform values on axis 1.
        In the special case in which there are 2 TPCs, the waveform 
        are taken from the same plane on both TPCs. The waveform
        on the second TPC is flipped in time, so as to keep the same
        x orientation. The waveforms on the 2 TPCs are then
        concatenated together on a wire by wire basis. 
        A padding with zeros is added between waveforms on different 
        TPCs, to account for a gap between the two cathodes. This gap is
        customizable by changing the geometry value "cathode gap".
        '''
        plane += cryo * self._n_plane * self._n_tpc
        
        if self._n_tpc == 2:
            array_right = self._process.getArrayByPlane(plane)
            for left_plane in self._plane_mix[plane]:
                array_left  = self._process.getArrayByPlane(left_plane)

                if self._plane_flip[left_plane]:
                    array_left = np.flip(array_left, axis=1)

                npad = ((0, 0), (0, int(self._gap)))
                array_right = np.pad(array_right, pad_width=npad, mode='constant', constant_values=0)

                array_right = np.concatenate((array_right, array_left), axis=1)

            # return array
            return array_right

        return self._process.getArrayByPlane(plane)


class recoWire(wire):

    def __init__(self, geom):
        super(recoWire, self).__init__()
        self._n_tpc = geom.nTPCs()
        self._n_plane = geom.nPlanes()
        self._gap = geom.cathodeGap()
        self._plane_mix = geom.planeMix()
        self._plane_flip = geom.planeFlip()
        self._process = evd.DrawWire(geom.getGeometryCore(), geom.getDetectorProperties())
        self._process.initialize()
        self._process.setInput(self._producerName)
        for plane in range(geom.nViews() * geom.nTPCs()):
            self._process.setYDimension(geom.readoutWindowSize(), plane)
            if geom.readoutPadding() != 0:
                self._process.setPadding(geom.readoutPadding(), plane)

    def setProducer(self, producer):
        self._producerName = producer
        if self._process is not None:
            self._process.clearInput()
            if isinstance(producer, list):
                for p in producer:
                    self._process.addInput(p)
            else:
                self._process.setInput(self._producerName)


class rawDigit(wire):

    def __init__(self, geom):
        super(rawDigit, self).__init__()
        self._n_tpc = geom.nTPCs()
        self._n_plane = geom.nPlanes()
        self._gap = geom.cathodeGap()
        self._plane_mix = geom.planeMix()
        self._plane_flip = geom.planeFlip()
        self._process = evd.DrawRawDigit(geom.getGeometryCore(), geom.getDetectorProperties())
        for i in range(len(geom._pedestals)):
            self._process.setPedestal(geom._pedestals[i], i)
        self._process.initialize()
        if "boone" in geom.name():
            self._process.SetCorrectData(False)
        else:
            self._process.SetCorrectData(False)
        for plane in range(geom.nViews() * geom.nTPCs()):
            self._process.setYDimension(geom.readoutWindowSize(), plane)
            if geom.readoutPadding() != 0:
                self._process.setPadding(geom.readoutPadding(), plane)


    def setProducer(self, producer):
        self._producerName = producer
        if self._process is not None:
            self._process.clearInput()
            if isinstance(producer, list):
                for p in producer:
                    self._process.addInput(p)
            else:
                self._process.setInput(self._producerName)
            
    def toggleNoiseFilter(self, filterNoise):
        self._process.SetCorrectData(filterNoise) 

    def setSubtractPedestal(self, subtract_pedestal=True):
        self._process.SetSubtractPedestal(subtract_pedestal)

