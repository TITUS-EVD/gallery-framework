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
from titus.gui.widgets import MultiSelectionBox, recoBox

# place any drawables associated with optical view here. For now, just flashes
_RAW_OPDETWAVEFORM = 'raw::OpDetWaveform'
_RECOB_OPFLASH = 'recob::OpFlash'
_DRAWABLE_LIST = {
    'OpFlash': [drawables.OpFlash, _RECOB_OPFLASH],
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

        self._view_dock =  QtWidgets.QDockWidget('OpDet View Controls', self._gui)
        self._view_dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self._dock =  QtWidgets.QDockWidget('OpDet Controls', self._gui)
        self._dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self._dock_widgets = set([self._view_dock, self._dock])

        self._opdet_views = []
        self._last_clicked_opdet = None
        self._selected_ch = None
        self._flashes = {}
        self._flash_drawers = {}
        self._opdet_wf_drawer = None

    def _initialize(self):
        self._gui.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self._view_dock)
        self._gui.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._dock)

        # geometry should be selected before creating buttons
        self._gm.geometryChanged.connect(self.init_ui)

    def init_ui(self):
        self._wf_view = waveform_view(self._gm.current_geom)
        self._layout.addWidget(self._wf_view)

        for tpc in range(self._gm.current_geom.nTPCs() * self._gm.current_geom.nCryos()):
            opdet_view = pg.GraphicsLayoutWidget()
            opdet_view.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
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

        self.init_opdet_controls()
        self.add_button_layout()

        self._wf_view.setMaximumHeight(200)
        self._wf_view.setMinimumHeight(200)

        self._flash_time_view.setMaximumHeight(150)
        self._flash_time_view.setMinimumHeight(150)

        for view in self._opdet_views:
            view.setMaximumHeight(500)
            view.setMinimumHeight(150)

        self._layout.setAlignment(QtCore.Qt.AlignTop)

    def init_opdet_controls(self):
        frame = QtWidgets.QWidget(self._dock)
        main_layout = QtWidgets.QVBoxLayout()
        frame.setLayout(main_layout)
        self._view_dock.setWidget(frame)

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
        main_layout.addWidget(view_group_box)
        main_layout.addStretch()


    def add_button_layout(self):
        frame = QtWidgets.QWidget(self._dock)
        main_layout = QtWidgets.QVBoxLayout()
        frame.setLayout(main_layout)
        self._dock.setWidget(frame)

        draw_group_box = QtWidgets.QGroupBox("Products")
        _bg2 = QtWidgets.QButtonGroup(self)
        self._show_none_btn = QtWidgets.QRadioButton("None")
        self._show_raw_btn = QtWidgets.QRadioButton("Raw Data")
        self._show_flash_btn = QtWidgets.QRadioButton("Flashes")
        self._show_none_btn.clicked.connect(self._raw_flash_switch_worker)
        self._show_raw_btn.clicked.connect(self._raw_flash_switch_worker)
        self._show_flash_btn.clicked.connect(self._raw_flash_switch_worker)
        self._show_none_btn.setChecked(True)
        _bg2.addButton(self._show_none_btn)
        _bg2.addButton(self._show_raw_btn)
        _bg2.addButton(self._show_flash_btn)

        products = self._gi.get_products(_RAW_OPDETWAVEFORM)
        default_products = self._gi.get_default_products(_RAW_OPDETWAVEFORM)
        self._wfm_choice = MultiSelectionBox(self, _RAW_OPDETWAVEFORM, products, default_products)
        self._wfm_choice.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self._wfm_choice.activated.connect(self._raw_flash_switch_worker)
        
        products = self._gi.get_products(_RECOB_OPFLASH)
        default_products = self._gi.get_default_products(_RECOB_OPFLASH)
        self._flash_choice = MultiSelectionBox(self, _RECOB_OPFLASH, products, default_products, mutually_exclusive=False)
        self._flash_choice.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self._flash_choice.activated.connect(self._raw_flash_switch_worker)

        raw_flash_btn_layout = QtWidgets.QGridLayout()
        raw_flash_btn_layout.addWidget(self._show_none_btn, 0, 0, 1, 1)
        raw_flash_btn_layout.addWidget(self._show_raw_btn, 1, 0, 1, 1)
        raw_flash_btn_layout.addWidget(self._wfm_choice, 1, 1, 1, 1)

        raw_flash_btn_layout.addWidget(self._show_flash_btn, 2, 0, 1, 1)
        raw_flash_btn_layout.addWidget(self._flash_choice, 2, 1, 1, 1)

        draw_group_box.setLayout(raw_flash_btn_layout)
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
        if self._show_none_btn.isChecked():
            self.toggle_opdets(None)
        elif self._show_raw_btn.isChecked():
            self.toggle_opdets(_RAW_OPDETWAVEFORM, stage=self._lsm.current_stage, producers=None)
        elif self._show_flash_btn.isChecked():
            self.toggle_opdets(_RECOB_OPFLASH, stage=self._lsm.current_stage, producers=None)

        self._gi.process_event(True)
        self.update()

    def toggle_opdets(self, product, stage=None, producers=None):
        if self._flash_drawers:
            for _, drawer in self._flash_drawers.items():
                drawer.clearDrawnObjects()

        if product is None:
            self.remove_drawable(self._opdet_wf_drawer)
            self._opdet_wf_drawer = None
            return

        all_producers = self._gi.get_producers(product, stage)
        if all_producers is None:
            return

        if product == _RAW_OPDETWAVEFORM:
            # set up waveform drawer
            if self._opdet_wf_drawer is None:
                products = self._gi.get_products(_DRAWABLE_LIST['OpDetWaveform'][1],
                                                 self._lsm.current_stage)
                self._opdet_wf_drawer = self.register_drawable(
                    _DRAWABLE_LIST['OpDetWaveform'][0](self._gi, self._gm.current_geom)
                )
            producer = self._wfm_choice.selected_products()[0]
            self._opdet_wf_drawer.set_producer(producer)

        elif product == _RECOB_OPFLASH:
            producers = self._flash_choice.selected_products()
            if producers is None:
                for _, drawer in self._flash_drawers:
                    self.remove_drawable(drawer)
                self._flash_drawers = {}

            for p in producers:
                if p in self._flash_drawers:
                    continue
                # create new flash drawer 
                products = self._gi.get_products(_DRAWABLE_LIST['OpFlash'][1],
                                            self._lsm.current_stage)
                if products is None:
                    return

                drawer = self.register_drawable(
                    _DRAWABLE_LIST['OpFlash'][0](self._gi, self._gm.current_geom, self)
                )
                self._flash_drawers[p] = drawer


            # check if we un-selected any previously selected producers
            new_flash_drawers = {}
            for producer, drawer in self._flash_drawers.items():
                if producer not in producers:
                    self.remove_drawable(drawer)
                    continue
                new_flash_drawers[producer] = drawer
               
            self._flash_drawers = new_flash_drawers
            print(self._flash_drawers)
            for producer, drawer in self._flash_drawers.items():
                print('set producer', producer)
                drawer.set_producer(producer)
            

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
            these_pmts.sigClicked.connect(self.opdetClickWorker)
            these_pmts.scene().sigMouseMoved.connect(these_pmts.onMove)

            these_arapucas = Arapucas(self._gm.current_geom, tpc=tpc, pmtscale=this_scale)
            opdet_plot.addItem(these_arapucas)
            these_arapucas.sigClicked.connect(self.opdetClickWorker)
            these_arapucas.scene().sigMouseMoved.connect(these_arapucas.onMove)

            self._opdet_plots.append(opdet_plot)
            self._pmts.append(these_pmts)
            self._arapucas.append(these_arapucas)
            self._opdetscales.append(this_scale)

    def update(self):
        if self._opdet_wf_drawer is not None:
            self.drawOpDetWvf(self._opdet_wf_drawer.getData())
            self._wf_view.drawWf(self._selected_ch)
        
        for _, drawer in self._flash_drawers.items():
            drawer.drawObjects()
        
        # self._flash_time_view.clear()
        self._flash_time_view.drawOpFlashTimes(self._flashes)
        
    def drawOpDetWvf(self, data):
        self._wf_view.drawOpDetWvf(data)
        if self._show_raw_btn.isChecked():
            for p, a in zip(self._pmts, self._arapucas):
                p.show_raw_data(data, self._selected_ch)
                a.show_raw_data(data, self._selected_ch)

    def setFlashesForPlane(self, p, flashes):
        if flashes is None:
            self._flashes[p] = None
            self._pmts[p].clear()
            return

        if len(flashes) == 0:
            return

        # self._time_range.setMin(int(time_min))
        # self._time_range.setMax(int(time_max))
        # self._time_range.setVisible(False)
        # self._time_range.setVisible(True)
        self._flashes[p] = flashes
        self._pmts[p].drawFlashes(flashes)


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

    def opdetClickWorker(self, plot, points):
        self._selected_ch = points[0].data()['id']
        all_opdets = self._pmts + self._arapucas
        for opdet_collection in all_opdets:
            opdet_collection.select_opdet(self._selected_ch)
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


        self._wf_plot.autoRange()

    def drawWf(self, ch):
        if ch is None:
            return

        if self._data is None:
            print ('OpDetWaveform data not loaded. No waveform to display.')
            return

        self._wf_plot.clear()

        # n_time_ticks = self._geometry.getDetectorClocks().OpticalClock().FrameTicks() * self._geometry.nOpticalFrames()
        data_y = self._data[ch,:] 
        ticks = len(data_y)
        data_x = np.linspace(0, ticks - 1, ticks)
        if data_y[0] == self._geometry.opdetDefaultValue():
            return

        # Remove the dafault values from the entries to be plotted

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


    def clear(self):
        self._time_plot.clear()

    def drawOpFlashTimes(self, flashes_by_plane):
        if not flashes_by_plane:
            return

        t_min = 0
        t_max = 0
        times = np.array([])
        for p, flashes in flashes_by_plane.items():
            if not flashes:
                continue
            
            times = np.hstack([times, [f.time() for f in flashes]])
            t_min = min(t_min, np.min(times))
            t_max = max(t_max, np.max(times))

        n_bins = int(t_max - t_min)
        if len(times) == 1:
            t_min -= 100
            t_max += 100
            n_bins = 200

        data_y, data_x = np.histogram(times, bins=np.linspace(t_min, t_max, n_bins))
        self._time_plot.plot(
            x=data_x, y=data_y, stepMode=True,
            fillLevel=0, brush=(0,0,255,150),
            clear=True
        )
        
        self._time_plot.addItem(self._time_window)
        self._time_plot.autoRange()
