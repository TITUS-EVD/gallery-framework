#/usr/bin/env python3

"""
This module adds the optical view & associated controls
"""
import numpy as np
import pyqtgraph as pg 
from PyQt5 import QtWidgets, QtGui, QtCore

from titus.gui.qrangeslider import QRangeSlider
from titus.gui.optical_elements import Pmts, Arapucas, _bordercol_

from titus.modules import Module
import titus.drawables as drawables

# place any drawables associated with optical view here. For now, just flashes
_RAW_OPDETWAVEFORM = 'raw::OpDetWaveform'
_DRAWABLE_LIST = {
    'OpFlash': [drawables.OpFlash, "recob::OpFlash"],
    'OpDetWaveform': [drawables.OpDetWaveform, _RAW_OPDETWAVEFORM],
}


class OpDetModule(Module):
    def __init__(self, larsoft_module, geom_module):
        super().__init__()
        self._gm = geom_module
        self._lsm = larsoft_module
        self._central_widget = QtWidgets.QWidget()
        self._layout = QtWidgets.QVBoxLayout()
        self._central_widget.setLayout(self._layout)

        self._dock =  QtWidgets.QDockWidget('OpDet Controls', self._gui)
        self._dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self._dock_widgets = [self._dock]

        self._opdet_views = []
        self._last_clicked_pmts = []
        self._last_clicked_arapucas = []
        self._flashes = {}
        self._flash_drawers = {}
        self._opdet_wf_drawer = None

    def _initialize(self):
        self._gui.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._dock)

        # geometry should be selected before creating buttons
        self._gm.geometryChanged.connect(self.init_ui)

    def init_ui(self):
        self._wf_view = waveform_view(self._gm.current_geom)
        self._layout.addWidget(self._wf_view)

        for tpc in range(self._gm.current_geom.nTPCs() * self._gm.current_geom.nCryos()):
            opdet_view = pg.GraphicsLayoutWidget()
            self._layout.addWidget(opdet_view)
            self._opdet_views.append(opdet_view)

        self.init_opdet_ui()
        self.connectStatusBar(self._gui.statusBar())

        self._flash_time_view = flash_time_view(self._gm.current_geom)
        self._layout.addWidget(self._flash_time_view)
        self._time_window = pg.LinearRegionItem(values=[0,10], orientation=pg.LinearRegionItem.Vertical)
        self._time_window.sigRegionChangeFinished.connect(self.time_range_worker)
        self._flash_time_view.connectTimeWindow(self._time_window)
        for p in self._pmts:
            p.set_time_range(self._time_window.getRegion())

        self.add_button_layout()

        self._wf_view.setMaximumHeight(200)
        self._wf_view.setMinimumHeight(200)

        self._flash_time_view.setMaximumHeight(150)
        self._flash_time_view.setMinimumHeight(150)

        for view in self._opdet_views:
            view.setMaximumHeight(500)
            view.setMinimumHeight(150)

        self._layout.setAlignment(QtCore.Qt.AlignTop)


    def add_button_layout(self):
        frame = QtWidgets.QWidget(self._dock)
        main_layout = QtWidgets.QVBoxLayout()
        frame.setLayout(main_layout)
        self._dock.setWidget(frame)

        view_group_box = QtWidgets.QGroupBox("Views")
        _bg1 = QtWidgets.QButtonGroup(self)
        self._tpc_all_button = QtWidgets.QRadioButton("All Cryos and TPCs")
        self._tpc_all_button.setToolTip("Shows all TPCs.")
        self._tpc_all_button.setChecked(True)
        self._tpc_all_button.clicked.connect(self.viewSelectionWorker)
        _bg1.addButton(self._tpc_all_button)

        self._tpc_buttons = []
        for cryo in range(self._gm.current_geom.nCryos()):
            for tpc in range(self._gm.current_geom.nTPCs()):
                tpc_button = QtWidgets.QRadioButton("Cryo "+str(cryo)+", TPC "+str(tpc))
                tpc_button.setToolTip("Shows only Cryo "+str(cryo)+", TPC "+str(tpc))
                tpc_button.clicked.connect(self.viewSelectionWorker)
                _bg1.addButton(tpc_button)
                self._tpc_buttons.append(tpc_button)

        button_layout = QtWidgets.QVBoxLayout()


        button_layout.addWidget(self._tpc_all_button)
        for item in self._tpc_buttons:
            button_layout.addWidget(item)
        view_group_box.setLayout(button_layout)

        draw_group_box = QtWidgets.QGroupBox("Products")
        _bg2 = QtWidgets.QButtonGroup(self)
        self._show_raw_btn = QtWidgets.QRadioButton("Raw Data")
        self._show_raw_btn.clicked.connect(self._raw_flash_switch_worker)
        self._show_raw_btn.setChecked(True)
        self._show_raw = True
        _bg2.addButton(self._show_raw_btn)
        self._show_flash_btn = QtWidgets.QRadioButton("Flashes")
        self._show_flash_btn.clicked.connect(self._raw_flash_switch_worker)
        _bg2.addButton(self._show_flash_btn)

        raw_flash_btn_layout = QtWidgets.QVBoxLayout()
        raw_flash_btn_layout.addWidget(self._show_raw_btn)
        raw_flash_btn_layout.addWidget(self._show_flash_btn)
        draw_group_box.setLayout(raw_flash_btn_layout)

        main_layout.addWidget(view_group_box)
        main_layout.addWidget(draw_group_box)
        main_layout.addStretch()


    def viewSelectionWorker(self):
        self._wf_view.setVisible(False)
        for view in self._opdet_views:
            view.setVisible(False)

        if self.sender() == self._tpc_all_button:
            self._wf_view.setVisible(True)
            for view in self._opdet_views:
                view.setVisible(True)

        for i in range(0, len(self._tpc_buttons)):
            if self.sender() == self._tpc_buttons[i]:
                self._wf_view.setVisible(True)
                self._opdet_views[i].setVisible(True)

    def _raw_flash_switch_worker(self):
        if self._flash_drawers:
            for _, drawer in self._flash_drawers.items():
                drawer.clearDrawnObjects()
        
        if self.sender() == self._show_raw_btn:
            self._show_raw = True
            return


        # otherwise, draw flashes
        self._show_raw = False
        if not self._flash_drawers:
            products = self._gi.get_products(_DRAWABLE_LIST['OpFlash'][1],
                                        self._lsm.current_stage)
            if products is None:
                return

            for p in products:
                drawer = self.register_drawable(
                    _DRAWABLE_LIST['OpFlash'][0](self._gi, self._gm.current_geom, self)
                )
                drawer.set_producer(p.full_name())
                self._flash_drawers[p.full_name()] = drawer
        
        for _, drawer in self._flash_drawers.items():
            drawer.drawObjects()

    def time_range_worker(self):
        for p in self._pmts:
            p.set_time_range(self._time_window.getRegion())

    def init_opdet_ui(self):
        self._opdet_plots = []
        self._pmts = []
        self._arapucas = []
        self._opdetscales = []

        for tpc in range(self._gm.current_geom.nTPCs() * self._gm.current_geom.nCryos()):
            opdet_plot = self._opdet_views[tpc].addPlot()
            opdet_plot.setLabel(axis='left', text='Y [cm]')
            opdet_plot.setLabel(axis='bottom', text='Z [cm]')

            this_scale = pg.GradientEditorItem(orientation='right')
            self._opdet_views[tpc].addItem(this_scale, 0, 1)

            these_pmts = Pmts(self._gm.current_geom, tpc=tpc, pmtscale=this_scale)
            opdet_plot.addItem(these_pmts)
            these_pmts.sigClicked.connect(self.pmtClickWorker)
            these_pmts.scene().sigMouseMoved.connect(these_pmts.onMove)

            these_arapucas = Arapucas(self._gm.current_geom, tpc=tpc, pmtscale=this_scale)
            opdet_plot.addItem(these_arapucas)
            these_arapucas.sigClicked.connect(self.arapucaClickWorker)
            these_arapucas.scene().sigMouseMoved.connect(these_arapucas.onMove)

            self._opdet_plots.append(opdet_plot)
            self._pmts.append(these_pmts)
            self._arapucas.append(these_arapucas)
            self._opdetscales.append(this_scale)

    def update(self):
        all_producers = self._gi.get_producers(_RAW_OPDETWAVEFORM, self._lsm.current_stage)
        if all_producers is None:
            if self._opdet_wf_drawer is not None:
                self.remove_drawable(self._opdet_wf_drawer)
                self._opdet_wf_drawer = None
            return
       
        # set up waveform drawer
        if self._opdet_wf_drawer is None:
            products = self._gi.get_products(_DRAWABLE_LIST['OpDetWaveform'][1],
                                             self._lsm.current_stage)
            self._opdet_wf_drawer = self.register_drawable(
                _DRAWABLE_LIST['OpDetWaveform'][0](self._gi, self._gm.current_geom)
            )
            self._opdet_wf_drawer.set_producer(products[1].full_name())

        self.drawOpDetWvf(self._opdet_wf_drawer.getData())

        

    def drawOpDetWvf(self, data):
        self._wf_view.drawOpDetWvf(data)

        if self._show_raw:
            for p, a in zip(self._pmts, self._arapucas):
                p.show_raw_data(data)
                a.show_raw_data(data)


    def setFlashesForPlane(self, p, flashes):

        if len(flashes) == 0:
            return

        times = []
        for f in flashes:
            times.append(f.time())

        time_min = np.min(times) - 10
        time_max = np.max(times) + 10

        # self._time_range.setMin(int(time_min))
        # self._time_range.setMax(int(time_max))
        # self._time_range.setVisible(False)
        # self._time_range.setVisible(True)
        self._flashes[p] = flashes
        self._pmts[p].drawFlashes(flashes)

        self._flash_time_view.drawOpFlashTimes(flashes)


    def getWidget(self):
        return self, self._layout


    def connectStatusBar(self, statusBar):
        self._statusBar = statusBar
        for t in range(0, len(self._pmts)):
            self._pmts[t].connectStatusBar(self._statusBar)
            self._pmts[t].connectStatusBar(self._statusBar)
            self._arapucas[t].connectStatusBar(self._statusBar)
            self._arapucas[t].connectStatusBar(self._statusBar)

    def connectMessageBar(self, messageBar):
        self._messageBar = messageBar

    def pmtClickWorker(self, plot, points):
        for p in self._last_clicked_pmts:
            p.setPen('w', width=2)
        for p in points:
            p.setPen('g', width=3)
            self._selected_ch = p.data()['id']

        self._last_clicked_pmts = points

        self._wf_view.drawWf(self._selected_ch)


    def arapucaClickWorker(self, plot, points):
        for p in self._last_clicked_arapucas:
            p.setPen(_bordercol_['arapuca_vuv'], width=1)
        for p in points:
            p.setPen('g', width=2)
            self._selected_ch = p.data()['id']

        self._last_clicked_arapucas = points

        self._wf_view.drawWf(self._selected_ch)

    def restoreDefaults(self):
        self._time_window.setBounds([0, 10])
        self.time_range_worker()


class waveform_view(pg.GraphicsLayoutWidget):

    def __init__(self, geometry, plane=-1):
        super(waveform_view, self).__init__(border=None)

        self._geometry = geometry

        self._data = None

        self._wf_plot = pg.PlotItem(name="OpDetWaveform")
        self._wf_plot.setLabel(axis='left', text='ADC')
        self._wf_plot.setLabel(axis='bottom', text='Ticks')
        # self._wf_linear_region = pg.LinearRegionItem(values=[0,30], orientation=pg.LinearRegionItem.Vertical)
        # self._wf_plot.addItem(self._wf_linear_region)
        self.addItem(self._wf_plot)


    def getWidget(self):
        return self._widget, self._layout

      # def init_geometry(self):

      #   opdets_x, opdets_y, opdets_z = self._geometry.opdetLoc()
      #   opdets_name = self._geometry.opdetName()
      #   diameter = self._geometry.opdetRadius() * 2

      #   self._opdet_circles = []
      #   for d in range(0, len(opdets_x)):
      #       # print('Adding opdet', opdets_x[d], opdets_y[d], diameter, diameter)
      #       self._opdet_circles.append(QtWidgets.QGraphicsEllipseItem(opdets_z[d], opdets_y[d], diameter, diameter))

      #       if opdets_name[d] == 'pmt':
      #           self._opdet_circles[d].setPen(pg.mkPen('r'))
      #       if opdets_name[d] == 'barepmt':
      #           self._opdet_circles[d].setPen(pg.mkPen('b'))

      #       if opdets_x[d] < 20 and (opdets_name[d] == 'pmt' or opdets_name[d] == 'barepmt'):
      #           self._view.addItem(self._opdet_circles[d])


    def drawOpDetWvf(self, data, offset=100):

        self._data = data

        # opdets_name = self._geometry.opdetName()

        # data_x = np.linspace(-1250, 2500, len(data[0]))

        # counter = 0
        # for ch in range(0, len(opdets_name)):
        #     name = opdets_name[ch]

        #     if name == 'pmt':
        #         data_y = data[ch]
        #         data_y = data_y + counter * offset
        #         # self._wf_plot.plot(x=data_x, y=data_y)
        #         counter += 1
        #         if counter > 7:
        #             break

        self._wf_plot.autoRange()

    def drawWf(self, ch):

        if self._data is None:
            print ('OpDetWaveform data not loaded. No waveform to display.')
            return

        self._wf_plot.clear()

        # n_time_ticks = self._geometry.getDetectorClocks().OpticalClock().FrameTicks() * self._geometry.nOpticalFrames()
        data_x = np.arange(len(self._data[ch]))
        # data_x = np.linspace(-1250, 2500, len(self._data[0]))
        data_y = self._data[ch]

        # Remove the dafault values from the entries to be plotted
        default_value_indexes = np.where(data_y == self._geometry.opdetDefaultValue())
        data_x = np.delete(data_x, default_value_indexes)
        data_y = np.delete(data_y, default_value_indexes)

        # self._wf_plot.plot(x=data_x, y=data_y, connect=False, symbol='o')
        self._wf_plot.plot(x=data_x, y=data_y)

        self._wf_plot.autoRange()




class flash_time_view(pg.GraphicsLayoutWidget):

    def __init__(self, geometry, plane=-1):
        super(flash_time_view, self).__init__(border=None)

        self._geometry = geometry

        self._data = None

        self._time_plot = pg.PlotItem(name="OpFlash Times")
        self._time_plot.setLabel(axis='left', text='Flashes')
        self._time_plot.setLabel(axis='bottom', text='Time [us]')

        self.addItem(self._time_plot)

    def connectTimeWindow(self, tw):
        self._time_window = tw


    def getWidget(self):
        return self._widget, self._layout


        self._wf_plot.autoRange()

    def drawOpFlashTimes(self, flashes):

        if flashes is None:
            return

        if len(flashes) == 0:
            return

        self._time_plot.clear()

        times = []
        for f in flashes:
            times.append(f.time())

        t_min = np.min(times)
        t_max = np.max(times)
        n_bins = int(t_max - t_min)

        if len(flashes) == 1:
            t_min -= 100
            t_max += 100
            n_bins = 200

        data_y, data_x = np.histogram(times, bins=np.linspace(t_min, t_max, n_bins))

        self._time_plot.plot(x=data_x, y=data_y, stepMode=True, fillLevel=0, brush=(0,0,255,150))
        self._time_plot.addItem(self._time_window)

        self._time_plot.autoRange()


