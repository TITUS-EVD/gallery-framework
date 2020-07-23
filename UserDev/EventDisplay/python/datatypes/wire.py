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
        self._split_wire = None
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
        print('Requested to draw plane', plane, 'cryo', cryo)
        n_tpc = self._n_tpc * 2 if self._split_wire else self._n_tpc
        plane += cryo * self._n_plane * n_tpc

        if self._n_tpc == 2:

            array = self._concatenatePlanes(plane)

            if self._split_wire:
                array_2 = self._concatenatePlanes(plane, plane_increase=3)

                npad = ((0, int(self._gap)), (0, 0))
                array = np.pad(array, pad_width=npad, mode='constant', constant_values=0)

                array = np.concatenate((array, array_2), axis=0)

            # return array
            return array

        return self._process.getArrayByPlane(plane)

    def _concatenatePlanes(self, plane, plane_increase=0):
        '''
        Concatenates planes across TPCs,
        as specified by geometry's _plane_mix

        arguments:
        - plane: The first plane to start concatenation
        - plane_increase: A number that constanty increases all plane numbers
        '''

        print('\twhich is plane', plane + plane_increase)

        array_right = self._process.getArrayByPlane(plane + plane_increase)
        for left_plane in self._plane_mix[plane]:
            left_plane += plane_increase
            print('\t\twhich is mixed with plane', left_plane)
            array_left  = self._process.getArrayByPlane(left_plane)

            if self._plane_flip[left_plane]:
                array_left = np.flip(array_left, axis=1)

            npad = ((0, 0), (0, int(self._gap)))
            array_right = np.pad(array_right, pad_width=npad, mode='constant', constant_values=0)

            array_right = np.concatenate((array_right, array_left), axis=1)

        return array_right


class recoWire(wire):

    def __init__(self, geom):
        super(recoWire, self).__init__()
        self._n_tpc = geom.nTPCs()
        self._n_plane = geom.nPlanes()
        self._gap = geom.cathodeGap()
        self._plane_mix = geom.planeMix()
        self._plane_flip = geom.planeFlip()
        self._split_wire = geom.splitWire()
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
        self._split_wire = geom.splitWire()
        self._process = evd.DrawRawDigit(geom.getGeometryCore(), geom.getDetectorProperties())
        self._process.setSplitWire(geom.splitWire())
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

