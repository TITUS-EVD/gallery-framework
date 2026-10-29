from titus.display.optical import opdet_traces
from titus.drawables.drawable import ReaderDrawable


class OpDetWaveform(ReaderDrawable):
    '''
    raw::OpDetWaveform as one trace per optical channel of the detector
    description (titus.display.optical), for the optical module's waveform
    view and optical detector colors. compression: keep every n-th sample.
    '''
    KIND = 'opdet_waveforms'
    DEFAULT_COMPRESSION = 10

    def __init__(self, gallery_interface, geom):
        super().__init__(gallery_interface)
        self._product_name = 'opdetwaveform'
        self._geom = geom
        self._compression = self.DEFAULT_COMPRESSION
        self._traces = None

    def set_compression_factor(self, x: int):
        self._compression = max(int(x), 1)
        self._traces = None

    def on_data(self):
        self._traces = None

    def getData(self):
        ''' {channel: {'time': us, 'adc': ...}} of the current event '''
        if self._traces is None:
            self._traces = {}
            for waveforms in self._data:
                for channel, trace in opdet_traces(
                        waveforms, len(self._geom.detector.opdets), self._geom.detector.timing.optical_tick_period,
                        self._compression, self._geom.opticalOffset()).items():
                    self._traces.setdefault(channel, trace)
        return self._traces
