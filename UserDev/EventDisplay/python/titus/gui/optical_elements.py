import pyqtgraph as pg
import numpy as np

_bordercol_ = {
    'pmt_coated'    : (255,255,255,255),
    'pmt_uncoated'  : (0,0,255,255),
    'arapuca_vuv'  : (34,139,34),
    'arapuca_vis'  : (34,139,34),
    'xarapuca_vuv'  : (34,139,34),
    'xarapuca_vis'  : (34,139,34),
}

class OpticalElements(pg.ScatterPlotItem):
    '''
    This class handles the drawing of optical elements
      (PTMs, arapucas, ...) as a scatter plot
    '''

    def __init__(self, geom, tpc=0, pmtscale=None):
        super().__init__()

        self._geom = geom
        self._tpc = tpc
        self._pmtscale = pmtscale

        # self._names = ['pmt_coated', 'pmt_uncoated']
        # self._size = 10
        # self._symbol = 'o'
        # self._line_width = 2

        self._data = None
        self._selected_ch = None
        self._time_range_wf = None # Time window for raw waveforms
        self._time_range_flash = [0, 10] # Time window for flashes
        self._exclude_uncoated = False

        self._min_range = 0
        self._max_range = 0

        self._flashes = None # The flashes to be displayed

        self._opdet_circles = self.get_opdet_circles()
        self._n_objects = len(self._opdet_circles)

        self.setAcceptHoverEvents(True)
        self.addPoints(self._opdet_circles)


    def get_opdet_circles(self, pe=None, max_pe=None):

        self._opdet_circles = []

        opdets_x, opdets_y, opdets_z = self._geom.opdetLoc()
        opdets_name = self._geom.opdetName()
        diameter = self._geom.opdetRadius() * 2

        brush = (0,0,0,0)

        loop_max = len(opdets_x)
        if pe is not None:
            loop_max = len(pe)

        if max_pe == 0:
            max_pe = 1

        for d in range(0, loop_max):
            if opdets_name[d] not in self._names:
                continue
            if self._geom.opdetToTPC(d) != self._tpc:
                continue

            if pe is not None:
                brush = self._pmtscale.colorMap().map(pe[d]/max_pe)

            self._opdet_circles.append(
                {
                    'pos'    : (opdets_z[d], opdets_y[d]),
                    'size'   : self._size,
                    'pen'    : {'color': _bordercol_[opdets_name[d]],
                                'width': self._line_width},
                    'brush'  : brush,
                    'symbol' : self._symbol,
                    'data'   : {'id': d, 'highlight': False}
                }
            )
        self._opdets_name = opdets_name
        self._opdets_x = opdets_x
        self._opdets_y = opdets_y
        self._opdets_z = opdets_z

        return self._opdet_circles

    def select_opdet(self, selected_ch=None):
        for p in self.points():
            ch = p.data()['id']
            pen_data = {
                'color': _bordercol_[self._geom.opdetName()[ch]],
                'width': self._line_width
            }
            if ch == selected_ch:
                pen_data['color'] = 'g'
                pen_data['width'] = 3
            p.setPen(**pen_data)


    def connectStatusBar(self, statusBar):
        self._statusBar = statusBar


    def onMove(self, pos):
        act_pos = self.mapFromScene(pos)
        p1 = self.pointsAt(act_pos)
        # print ('onMove, act_pos', act_pos, 'p1', p1)
        if len(p1) != 0:

            opdet_id = p1[0].data()['id']
            opdet_name = self._opdets_name[opdet_id]

            if (pg.Qt.QT_LIB == 'PyQt4'):
                message = QtCore.QString()
            else:
                message = str()

            if type(message) != str:
                message.append("OpDetName: ")
                message.append(opdet_name)
                message.append(";   X: ")
                message.append("{0:.1f}".format(self._opdets_x[opdet_id]))
                message.append(";   Y: ")
                message.append("{0:.1f}".format(self._opdets_y[opdet_id]))
                message.append(";   Z: ")
                message.append("{0:.1f}".format(self._opdets_z[opdet_id]))
                message.append(";   OpCh: ")
                message.append("{0:.0f}".format(opdet_id))
            else:
                message += "OpDetName: "
                message += opdet_name
                message += ";   X: "
                message += "{0:.1f}".format(self._opdets_x[opdet_id])
                message += ";   Y: "
                message += "{0:.1f}".format(self._opdets_y[opdet_id])
                message += ";   Z: "
                message += "{0:.1f}".format(self._opdets_z[opdet_id])
                message += ";   OpCh: "
                message += "{0:.0f}".format(opdet_id)

            self._statusBar.showMessage(message)


    def set_time_range(self, time_range):
        self._time_range_flash = time_range

    def set_wf_time_range(self, time_range):
        self._time_range_wf = time_range

    def exclude_uncoated(self, do_exclude=False):
        self.show_raw_data(self._data, self._selected_ch)

    def get_min_scale(self):
        return self._min_range

    def get_max_scale(self):
        return self._max_range

    def set_scale(self, min_range, max_range):
        self._min_range = min_range
        self._max_range = max_range
        self._pmtscale


    def drawFlashes(self, flashes):
        if flashes is None:
            return

        if len(flashes) == 0:
            return

        self._flashes = flashes

        n_drawn_flashes = 0

        total_pes_per_opdet = [0] * len(flashes[0].pe_per_opdet())

        for f in flashes:
            if (f.time() > self._time_range_flash[0] and f.time() < self._time_range_flash[1]):
                total_pes_per_opdet = np.add(total_pes_per_opdet, f.pe_per_opdet())
                n_drawn_flashes += 1

        max_pe = np.max(total_pes_per_opdet)
        self._opdet_circles = self.get_opdet_circles(total_pes_per_opdet, max_pe)
        self.clear()
        self.addPoints(self._opdet_circles)

        # print ('Displaying', n_drawn_flashes, 'flashes.')


    def set_raw_data(self, data):
        '''
        Sets raw waveform data and evaluates max and min of
        based on waveform amplitudes
        '''
        self._data = data

        self._pe_per_opdet = [0] * self._geom.getGeometryCore().NOpDets()
        for element in self._opdet_circles:
            ch = element['data']['id']
            data_y = self._data[ch,:]

            if len(data_y) == 0:
                continue

            if data_y[0] == self._geom.opdetDefaultValue():
                continue

            if 'uncoated' in self._opdets_name[ch] and self._exclude_uncoated:
                continue

            if self._time_range_wf:
                tick_min = int(self._time_range_wf[0])
                tick_max = int(self._time_range_wf[1])

                tick_min = np.maximum(tick_min, 0)
                tick_max = np.minimum(tick_max, len(data_y))

                data_y_sel = data_y[tick_min:tick_max]
            else:
                data_y_sel = data_y

            amplitude = data_y_sel.max() - data_y_sel.min()

            self._pe_per_opdet[ch] = np.abs(amplitude)

        max_pe = np.max(self._pe_per_opdet)
        min_pe = 0

        return min_pe, max_pe


    def show_raw_data(self, min_scale, max_scale, selected_ch=None):
        '''
        Displays the data
        '''

        self._opdet_circles = self.get_opdet_circles(self._pe_per_opdet, max_scale)
        self.clear()
        self.addPoints(self._opdet_circles)
        self.select_opdet(selected_ch)


class Pmts(OpticalElements):
    '''
    This class handles the drawing of the
      PMTs as a scatter plot
    '''
    def __init__(self, geom, tpc, pmtscale=None):
        self._names = ['pmt_coated', 'pmt_uncoated']
        self._size = 10
        self._symbol = 'o'
        self._line_width = 2
        super().__init__(geom, tpc, pmtscale)


class Arapucas(OpticalElements):
    '''
    This class handles the drawing of the
      Arapucas as a scatter plot
    '''
    def __init__(self, geom, tpc=0, pmtscale=None):
        self._names = ['xarapuca_vuv', 'xarapuca_vis', 'arapuca_vuv', 'arapuca_vis']
        self._size = 6
        self._symbol = 's'
        self._line_width = 1
        super(Arapucas, self).__init__(geom, tpc, pmtscale)
