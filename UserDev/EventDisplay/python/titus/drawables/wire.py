from .drawable import ReaderDrawable
import numpy as np

from titus.display import ViewLayout, raw_digit_signals, wire_image


class Wire(ReaderDrawable):
    '''
    Waveforms drawn as one image per view (plane number and cryostat). The
    parallel planes of all TPCs of the cryostat are tiled into the image as
    described in titus.display.layout: TPCs drifting the other way are
    flipped in time and placed after the cathode gap, and with split wires
    the higher-z TPC of each drift volume is placed after the first along
    the wire axis.

    The image is geom.tRange() ticks per TPC, as the views and their overlays
    (anode/cathode, TPC division, waveform plot) assume. Longer waveforms
    are cut.
    '''

    def __init__(self, gallery_interface, geom):
        super().__init__(gallery_interface)
        self._geom = geom
        self._signals = []

    def on_data(self):
        self._signals = list(self._data)

    def getPlane(self, plane, cryo=0):
        ''' the image of the view for this plane number: (wires, ticks) '''
        layout = ViewLayout(self._geom, plane=plane, cryostat=cryo)
        image = np.zeros(layout.shape, dtype='f4')
        for signals in self._signals:
            image += wire_image(layout, signals, self._geom.detector.channel_map)
        return image


class RecoWire(Wire):
    KIND = 'wires'


class recoChannelROI(Wire):
    KIND = 'channel_rois'


class RawDigit(Wire):
    KIND = 'raw_digits'

    def __init__(self, gallery_interface, geom):
        super().__init__(gallery_interface, geom)
        self._subtract_pedestal = True

    def on_data(self):
        self._signals = [raw_digit_signals(d, self._subtract_pedestal) for d in self._data]

    def toggleNoiseFilter(self, filterNoise):
        ''' the noise filter was MicroBooNE only '''
        pass

    def setSubtractPedestal(self, subtract_pedestal=True):
        self._subtract_pedestal = subtract_pedestal
