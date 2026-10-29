import numpy as np

from titus.display.crt import crt_strip_hits
from titus.drawables.drawable import ReaderDrawable


class FEBData(ReaderDrawable):
    '''
    sbnd::crt::FEBData as CRT strip hits for the CRT view (getData(), see
    titus.display.crt). The event trigger time stamp comes from the file's
    SPEC-TDC product (TDC_LABEL), when there is one (data, not MC).
    '''
    KIND = 'crt_febdata'
    TDC_LABEL = 'tdcdecoder'

    def __init__(self, gallery_interface, geom):
        super().__init__(gallery_interface)
        self._product_name = 'febdata'
        self._geom = geom
        self._hits = np.zeros((0, 4))

    def tdc_timestamps(self):
        tags = [tag for kind, tag in self.file_products() if kind == 'daq_timestamps'
                and tag.split(':')[0] == self.TDC_LABEL]
        return self.read_optional('daq_timestamps', tags[0]) if tags else None

    def on_data(self):
        if not self._data:
            self._hits = np.zeros((0, 4))
            return
        timestamps = self.tdc_timestamps()
        self._hits = np.concatenate([crt_strip_hits(febs, timestamps) for febs in self._data])

    def getData(self):
        return self._hits
