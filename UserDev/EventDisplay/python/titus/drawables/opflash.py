import numpy as np

from titus.drawables.drawable import ReaderDrawable


class Flash:
    ''' one recob::OpFlash, with the accessors the optical views use '''

    def __init__(self, flashes, row, n_channels=None):
        self._flashes, self._row, self._n_channels = flashes, row, n_channels

    def time(self):
        ''' AbsTime: us from the hardware trigger '''
        return float(self._flashes.abs_time[self._row])

    def time_width(self):
        return float(self._flashes.time_width[self._row])

    def total_pe(self):
        return float(self._flashes.total_pe[self._row])

    def y(self):
        return float(self._flashes.y_center[self._row])

    def z(self):
        return float(self._flashes.z_center[self._row])

    def pe_per_opdet(self):
        '''
        PEs by optical channel, for every channel: recob::OpFlash::PEs() ends
        at the highest channel its producer reads (e.g. 305 of SBND's 312)
        '''
        pe = self._flashes.pe_per_channel(self._row)
        if self._n_channels is None or len(pe) >= self._n_channels:
            return pe
        return np.concatenate([pe, np.zeros(self._n_channels - len(pe), dtype=pe.dtype)])


def flash_volumes(flashes, channel_volume):
    '''
    The drift volume (x-ordered, as Geometry.opdetToTPC) of each flash: the
    one whose optical detectors collected most of its PEs. channel_volume:
    drift volume per optical channel.
    '''
    out = np.zeros(len(flashes), dtype=int)
    n_volumes = int(channel_volume.max()) + 1 if len(channel_volume) else 1
    for row in range(len(flashes)):
        pe = flashes.pe_per_channel(row)[:len(channel_volume)]
        out[row] = np.argmax(np.bincount(channel_volume[:len(pe)], weights=pe, minlength=n_volumes))
    return out


class OpFlash(ReaderDrawable):
    '''
    recob::OpFlash, shown by the optical module: each flash colors the
    optical detectors of the drift volume holding most of its PEs, by their
    PEs, and appears in the flash time view
    '''
    KIND = 'opflashes'

    def __init__(self, gallery_interface, geom, opdet_module):
        super().__init__(gallery_interface)
        self._product_name = 'opflash'
        self._geom = geom
        # this product is only drawn when OpDet module is active, and it needs
        # to access the GUI elements in the OpDet module class.
        self._module = opdet_module
        self._channel_volume = None

    def n_volumes(self):
        return self._geom.nTPCs() * self._geom.nCryos()

    def channel_volume(self):
        if self._channel_volume is None:
            self._channel_volume = np.array([self._geom.opdetToTPC(ch)
                                             for ch in range(len(self._geom.detector.opdets))], dtype=int)
        return self._channel_volume

    def flashes_by_volume(self):
        ''' {drift volume: [Flash]} of the current event '''
        out = {}
        for flashes in self._data:
            for row, volume in enumerate(flash_volumes(flashes, self.channel_volume())):
                out.setdefault(int(volume), []).append(Flash(flashes, row, len(self.channel_volume())))
        return out

    def drawObjects(self):
        for volume, flashes in self.flashes_by_volume().items():
            if volume < self.n_volumes():
                self._module.setFlashesForPlane(volume, flashes)

    def clearDrawnObjects(self, obj_list=None):
        """ Override base class """
        for p in range(self.n_volumes()):
            self._module.setFlashesForPlane(p, None)
