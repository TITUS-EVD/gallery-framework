from .drawable import Drawable
from ROOT import evd
import pyqtgraph as pg
import numpy as np


class Wire(Drawable):
    def __init__(self, gallery_interface):
        super().__init__(gallery_interface)
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
        # print('Requested to draw plane', plane, 'cryo', cryo)
        # n_tpc = self._n_tpc * 2 if self._split_wire else self._n_tpc
        # plane += cryo * self._n_plane * n_tpc

        plane = self._geom.getPlaneID(plane=plane, tpc=0, cryo=cryo)
        other_planes = self._geom.getOtherPlanes(plane_id=plane)

        if len(other_planes) == 0:
            return self._process.getArrayByPlane(plane)

        if len(other_planes) == 1:
            array = self._concatenatePlanes(plane, other_planes[0])

        elif len(other_planes) == 3:
            array_1 = self._concatenatePlanes(plane, other_planes[1])
            array_2 = self._concatenatePlanes(other_planes[0], other_planes[2])

            # Horizontal concatenation
            npad = ((0, int(self._gap)), (0, 0))
            array_1 = np.pad(array_1, pad_width=npad, mode='constant', constant_values=0)
            array = np.concatenate((array_1, array_2), axis=0)

        else:
            raise Exception(f'{len(other_planes)} additional planes are not supported.')

        return array


    def _concatenatePlanes(self, right_plane, left_plane):
        '''
        Concatenates planes across TPCs,
        as specified by geometry's _plane_mix

        arguments:
        - plane: The first plane to start concatenation
        - plane_increase: A number that constanty increases all plane numbers
        '''

        array_right = self._process.getArrayByPlane(right_plane)
        array_left  = self._process.getArrayByPlane(left_plane)

        if self._geom.flipPlane(left_plane):
            array_left = np.flip(array_left, axis=1)

        npad = ((0, 0), (0, int(self._gap)))
        array_right = np.pad(array_right, pad_width=npad, mode='constant', constant_values=0)

        array = np.concatenate((array_right, array_left), axis=1)

        return array



class RecoWire(Wire):

    def __init__(self, gallery_interface, geom):
        super().__init__(gallery_interface)
        self._geom = geom
        self._n_tpc = geom.nTPCs()
        self._n_plane = geom.nPlanes()
        self._gap = geom.cathodeGap()
        self._plane_mix = geom.planeMix()
        self._plane_flip = geom.planeFlip()
        self._split_wire = geom.splitWire()
        self._process = evd.DrawWire(geom.getGeometryCore(), geom.getDetectorProperties())
        self._process.initialize()
        self._process.setInput(self._producer_name)
        for plane in range(geom.nViews() * geom.nTPCs()):
            self._process.setYDimension(geom.readoutWindowSize(), plane)
            if geom.readoutPadding() != 0:
                self._process.setPadding(geom.readoutPadding(), plane)

    def set_producer(self, producer):
        self._producer_name = producer
        if self._process is not None:
            self._process.clearInput()
            if isinstance(producer, list):
                for p in producer:
                    self._process.addInput(p)
            else:
                self._process.setInput(self._producer_name)

class recoChannelROI(Wire):
    def __init__(self, gallery_interface, geom):
        super().__init__(gallery_interface)
        self._geom = geom
        self._n_tpc = geom.nTPCs()
        self._n_plane = geom.nPlanes()
        self._gap = geom.cathodeGap()
        self._plane_mix = geom.planeMix()
        self._plane_flip = geom.planeFlip()
        self._split_wire = geom.splitWire()
        self._process = evd.DrawChannelROI(geom.getGeometryCore(), geom.getDetectorProperties())
        self._process.initialize()
        self._process.setInput(self._producer_name)
        for plane in range(geom.nViews() * geom.nTPCs()):
            self._process.setYDimension(geom.readoutWindowSize(), plane)
            if geom.readoutPadding() != 0:
                self._process.setPadding(geom.readoutPadding(), plane)

    # def setProducer(self, producer):
    #     self._producerName = producer
    #     if self._process is not None:
    #         self._process.clearInput()
    #         if isinstance(producer, list):
    #             for p in producer:
    #                 self._process.addInput(p)
    #         else:
    #             self._process.setInput(self._producer_name)


class RawDigit(Wire):
    def __init__(self, gallery_interface, geom):
        super().__init__(gallery_interface)
        self._geom = geom
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


    def set_producer(self, producer):
        self._producer_name = producer
        if self._process is not None:
            self._process.clearInput()
            if isinstance(producer, list):
                for p in producer:
                    self._process.addInput(p)
            else:
                self._process.setInput(self._producer_name)

    def toggleNoiseFilter(self, filterNoise):
        self._process.SetCorrectData(filterNoise) 

    def setSubtractPedestal(self, subtract_pedestal=True):
        self._process.SetSubtractPedestal(subtract_pedestal)

