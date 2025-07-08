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
from titus.gui.widgets import MultiSelectionBox, recoBox, VerticalLabel

# place any drawables associated with optical view here. For now, just flashes
_RAW_OPDETWAVEFORM = 'raw::OpDetWaveform'
_RECOB_OPFLASH = 'recob::OpFlash'
_DRAWABLE_LIST = {
    'OpFlash': [drawables.OpFlash, _RECOB_OPFLASH],
    'OpDetWaveform': [drawables.OpDetWaveform, _RAW_OPDETWAVEFORM],
}


# Optical view setting strings
_SET_WAVEFORM_COMPRESSION = 'Optical/Waveform compression factor'


class OpDetModule(Module):
    def __init__(self, larsoft_module, geom_module):
        super().__init__()
        self._name = 'Optical'
        self._gm = geom_module
        self._lsm = larsoft_module
        self._central_widget = QtWidgets.QWidget()
        self._layout = QtWidgets.QVBoxLayout()
        self._central_widget.setLayout(self._layout)

        self._view_dock =  QtWidgets.QDockWidget('OpDet View Controls', self._gui, objectName='opdet_dock_view')
        self._view_dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self._dock =  QtWidgets.QDockWidget('OpDet Controls', self._gui, objectName='opdet_dock')
        self._dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self._dock_widgets = set([self._view_dock, self._dock])

        self._opdet_views = []
        self._last_clicked_opdet = None
        self._selected_ch = None
        self._flashes = {}
        self._flash_drawers = {}
        self._opdet_wf_drawer = None

        # user settings page
        self._settings_defaults = {
            _SET_WAVEFORM_COMPRESSION: 10,
        }
        self._init_settings_page()

    def _init_settings_page(self):
        self._settings_layout = QtWidgets.QGridLayout()

        label = QtWidgets.QLabel(_SET_WAVEFORM_COMPRESSION.split('/')[1])
        self._settings_layout.addWidget(label, 0, 0, 1, 1)
        self._waveform_compression_box = QtWidgets.QSpinBox()
        self._waveform_compression_box.setRange(1, 100)
        self._waveform_compression_box.setSingleStep(5)
        self._waveform_compression_box.setValue(self._settings_defaults[_SET_WAVEFORM_COMPRESSION])
        self._waveform_compression_box.valueChanged.connect(
            lambda x: self._update_waveform_compression(x)
        )
        self._settings_layout.addWidget(self._waveform_compression_box, 0, 1, 1, -1)

        restore_btn = QtWidgets.QPushButton("Restore Defaults")
        restore_btn.clicked.connect(self.restore_settings_defaults)
        self._settings_layout.addWidget(restore_btn, 1, 1, 1, -1)

        self._settings_layout.setRowStretch(self._settings_layout.rowCount(), 1)

    def restore_settings_defaults(self):
        self._settings.clear()
        self.restore_from_settings()

    def restore_from_settings(self):
        ''' emitting forces the connected signal even if the value is the same '''
        x = int(self._settings.value(_SET_WAVEFORM_COMPRESSION,
                                     self._settings_defaults[_SET_WAVEFORM_COMPRESSION]))
        self._waveform_compression_box.setValue(x)
        self._waveform_compression_box.valueChanged.emit(x)

    def _initialize(self):
        self._gui.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self._view_dock)
        self._gui.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._dock)

        # geometry should be selected before creating buttons
        self._gm.geometryChanged.connect(self.init_ui)
        self._gm.geometryChanged.connect(self.restore_from_settings)

    def init_ui(self):
        # Waveform View
        wfv_layout = QtWidgets.QHBoxLayout()
        self._wf_view = waveform_view(self._gm.current_geom, self._lsm.det_clock_service)
        self._wf_view.timeWindowChanged.connect(self.time_range_wf_worker)

        name = VerticalLabel('Waveform Viewer')
        name.setStyleSheet('color: rgb(169,169,169);')
        name.setMaximumWidth(25)
        wfv_layout.addWidget(name)
        wfv_layout.addWidget(self._wf_view)

        self._layout.addLayout(wfv_layout)

        # TPC OpDets Views
        n_tpcs = self._gm.current_geom.nTPCs() * self._gm.current_geom.nCryos()
        self._opdet_views = [None] * n_tpcs
        for tpc in range(n_tpcs-1, -1, -1):
        # for tpc in range(self._gm.current_geom.nTPCs() * self._gm.current_geom.nCryos()):
            opdet_view = pg.GraphicsLayoutWidget()
            opdet_view.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            # self._layout.addWidget(opdet_view)
            self._opdet_views.append(opdet_view)
            self._opdet_views[tpc] = opdet_view

            name = VerticalLabel('TPC East' if tpc == 0 else 'TPC West')
            name.setToolTip(f'TPC {tpc}')
            name.setStyleSheet('color: rgb(169,169,169);')
            name.setMaximumWidth(25)

            tpc_layout = QtWidgets.QHBoxLayout()
            tpc_layout.addWidget(name)
            tpc_layout.addWidget(opdet_view)

            self._layout.addLayout(tpc_layout)

        self.init_opdet_ui()
        self.connectStatusBar(self._gui.statusBar())

        # Flash Time View
        flash_layout = QtWidgets.QHBoxLayout()
        self._flash_time_view = flash_time_view(self._gm.current_geom)
        # self._layout.addWidget(self._flash_time_view)
        self._time_window = pg.LinearRegionItem(values=[0,10], orientation=pg.LinearRegionItem.Vertical)
        self._time_window.sigRegionChangeFinished.connect(self.time_range_worker)
        self._flash_time_view.connectTimeWindow(self._time_window)
        for p in self._pmts:
            p.set_time_range(self._time_window.getRegion())

        name = VerticalLabel('Flash Time Viewer')
        name.setStyleSheet('color: rgb(169,169,169);')
        name.setMaximumWidth(25)
        flash_layout.addWidget(name)
        flash_layout.addWidget(self._flash_time_view)

        self._layout.addLayout(flash_layout)

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

        self.time_range_wf_worker(self._wf_view._time_range)

    def init_opdet_controls(self):
        frame = QtWidgets.QWidget(self._dock)
        main_layout = QtWidgets.QVBoxLayout()
        frame.setLayout(main_layout)
        self._view_dock.setWidget(frame)

        view_group_box = QtWidgets.QGroupBox("Views")
        _bg1 = QtWidgets.QButtonGroup(self)
        self._tpc_all_button = QtWidgets.QRadioButton("All TPCs")
        self._tpc_all_button.setToolTip("Shows all TPCs.")
        self._tpc_all_button.setChecked(True)
        self._tpc_all_button.clicked.connect(self.viewSelectionWorker)
        _bg1.addButton(self._tpc_all_button)

        self._tpc_buttons = []
        for tpc in range(self._gm.current_geom.nTPCs()):
            tpc_button = QtWidgets.QRadioButton('TPC East' if tpc == 0 else 'TPC West')
            tpc_button.setToolTip('Show only TPC 0' if tpc == 0 else 'Show only TPC 1')
            tpc_button.clicked.connect(self.viewSelectionWorker)
            _bg1.addButton(tpc_button)
            self._tpc_buttons.append(tpc_button)
        
        button_layout = QtWidgets.QVBoxLayout()

        button_layout.addWidget(self._tpc_all_button)
        for item in self._tpc_buttons:
            button_layout.addWidget(item)
        view_group_box.setLayout(button_layout)

        fine_selection_layout = QtWidgets.QVBoxLayout()
        self._no_uncoated_btn = QtWidgets.QCheckBox("Exclude Uncoated")
        self._no_uncoated_btn.stateChanged.connect(self.exclude_uncoated)
        fine_selection_layout.addWidget(self._no_uncoated_btn)


        main_layout.addWidget(view_group_box)
        main_layout.addLayout(fine_selection_layout)
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

    def update_reco_boxes(self):
        ''' update all the product boxes when the file or larsoft stage changes '''
        current_wfm_choice = self._wfm_choice.selected_products()
        products = self._gi.get_products(_RAW_OPDETWAVEFORM, self._lsm.current_stage)
        self._wfm_choice.set_products(products)
        for c in current_wfm_choice:
            self._wfm_choice.select(c)
        if not products and self._show_raw_btn.isChecked():
            self._show_none_btn.toggle()
        
        current_opflash_choice = self._flash_choice.selected_products()
        products = self._gi.get_products(_RECOB_OPFLASH)
        self._flash_choice.set_products(products)
        for c in current_opflash_choice:
            self._flash_choice.select(c)
        if not products and self._show_flash_btn.isChecked():
            self._show_none_btn.toggle()

    def _update_waveform_compression(self, x: int):
        self._settings.setValue(_SET_WAVEFORM_COMPRESSION, x)
        if self._opdet_wf_drawer is None:
            return
        self._opdet_wf_drawer.set_compression_factor(x)
        self._gi.process_event(True)

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
        self.remove_drawable(self._opdet_wf_drawer)
        self._opdet_wf_drawer = None
        for _, drawer in self._flash_drawers.items():
            self.remove_drawable(drawer)
            drawer.clearDrawnObjects()
        self._flash_drawers = {}
        
        if self._show_none_btn.isChecked():
            self.toggle_opdets(None)
        elif self._show_raw_btn.isChecked():
            self.toggle_opdets(_RAW_OPDETWAVEFORM, stage=self._lsm.current_stage, producers=None)
        elif self._show_flash_btn.isChecked():
            self.toggle_opdets(_RECOB_OPFLASH, stage=self._lsm.current_stage, producers=None)

        self._gi.process_event(True)

    def toggle_opdets(self, product, stage=None, producers=None):
        def clear_flash_drawers():
            for _, drawer in self._flash_drawers.items():
                drawer.clearDrawnObjects()
                self.remove_drawable(drawer)

        def clear_opdet_wf_drawer():
            self.remove_drawable(self._opdet_wf_drawer)
            self._opdet_wf_drawer = None

        if product is None:
            clear_flash_drawers()
            clear_opdet_wf_drawer()
            return

        all_producers = self._gi.get_producers(product, stage)

        if product == _RAW_OPDETWAVEFORM:
            if all_producers is None:
                clear_opdet_wf_drawer()
                return
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
            if all_producers is None:
                clear_flash_drawers()
                return
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
            for producer, drawer in self._flash_drawers.items():
                drawer.set_producer(producer)

    def exclude_uncoated(self):
        for p in self._pmts:
            p.exclude_uncoated(self._no_uncoated_btn.isChecked())
            
    def time_range_wf_worker(self, t_range):
        '''
        Sets a time window for raw waveforms. The waveform in the
        time window is used to color OpDets.
        '''
        for p in self._pmts:
            p.set_wf_time_range(t_range)

        self.update()

    def time_range_worker(self):
        '''
        Sets a time window for flashes. The flashes in the
        time window is used to color OpDets.
        '''
        for p in self._pmts:
            p.set_time_range(self._time_window.getRegion())

        self.update()

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

    def on_file_change(self):
        ''' override of base class method to update available products first '''
        self.update_reco_boxes()
        self._raw_flash_switch_worker()

    def update(self):
        if self._opdet_wf_drawer is not None:
            self.drawOpDetWvf(self._opdet_wf_drawer.getData())
            self._wf_view.drawWf(self._selected_ch)
        
        for _, drawer in self._flash_drawers.items():
            drawer.drawObjects()
        
        # self._flash_time_view.clear()
        self._flash_time_view.drawOpFlashTimes(self._flashes)
        
    def drawOpDetWvf(self, data):
        '''
        OpDet display shows raw waveform data
        '''
        if not self._show_raw_btn.isChecked():
            return
        
        result = parse_opdetwaveforms(data, self._gm.current_geom, self._lsm.det_clock_service)

        self._wf_view.drawOpDetWvf(result)

        max_scale = -1e12
        min_scale = 1e12

        for p in self._pmts:
            min_sc, max_sc = p.set_raw_data(result)
            max_scale = max(max_scale, max_sc)
            min_scale = min(min_scale, min_sc)

        for p in self._pmts:
            p.show_raw_data(min_scale, max_scale, self._selected_ch)

        # TODO: do the same for arapucas


    def setFlashesForPlane(self, p, flashes):
        if flashes is None:
            self._flashes[p] = None
            self._pmts[p].clear()
            return

        if len(flashes) == 0:
            return

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

    timeWindowChanged = QtCore.pyqtSignal(tuple)
    
    def __init__(self, geometry, clock_service, plane=-1):
        super(waveform_view, self).__init__(border=None)

        self._geometry = geometry
        self._clock_service = clock_service

        self._data = None

        self._time_range = [-1500, 1500]

        self._wf_plot = pg.PlotItem(name="OpDetWaveform")
        self._wf_plot.setLabel(axis='left', text='ADC')
        self._wf_plot.setLabel(axis='bottom', text='Time (us)')
        self.addItem(self._wf_plot)


        self._time_window = pg.LinearRegionItem(values=self._time_range, orientation=pg.LinearRegionItem.Vertical)
        self._time_window.sigRegionChangeFinished.connect(lambda: self.timeWindowChanged.emit(self._time_window.getRegion()))
        self._wf_plot.addItem(self._time_window)


    def getWidget(self):
        return self._widget, self._layout


    def drawOpDetWvf(self, data):
        """Update member variables needed to properly draw waveforms."""
        self._data = data

        self._wf_plot.autoRange()

    def drawWf(self, ch):
        if ch is None:
            return

        if self._data is None:
            print ('OpDetWaveform data not loaded. No waveform to display.')
            return

        self._wf_plot.clear()
        self._wf_plot.plot(x=self._data[ch]['time'], y=self._data[ch]['adc'])
        self._wf_plot.addItem(self._time_window)
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


def parse_opdetwaveforms(data, geometry, clock_service):
    """Get waveforms & times from raw data returned by DrawOpDetWaveform."""
    # unpack variable number of waveforms
    n_waveforms = int(data[0])
    compression_factor = int(data[1])

    # waveform data follows first two elements. each waveform separated by
    # a NaN. remove the last element (empty) after the split and then the
    # leading NaN from each waveform
    data = data[2:]
    if len(data) == 0:
        return {}

    wvfm_breaks = np.where(np.isnan(data))[0]
    wvfms = np.split(data, wvfm_breaks)[:-1]
    for i in range(len(wvfms)):
        wvfms[i] = wvfms[i][~np.isnan(wvfms[i])]

    # result is a dict like
    # channel: {'time': [t1, t2, ...], 'adc': [a1, a2, a3, ...]}
    tick_period = clock_service.DataForJob().OpticalClock().TickPeriod()
    result = {}
    for w in wvfms:
        ch = int(w[0])
        offset = w[1]
        # remove trailing zeros
        wvfm = np.trim_zeros(w[2:], trim='b')

        if wvfm[0] == geometry.opdetDefaultValue():
            continue

        ticks = len(wvfm)
        # extra element for trailing NaN. The NaN prevents pyqtgraph from
        # connecting waveform lines
        xs = np.arange(0, ticks + 1, dtype=float)
        xs *= compression_factor

        # tick to timestamp
        xs *= tick_period
        xs += offset 

        # TODO: Lots of appends. These create copies so would be better to
        # pre-allocate somehow
        xs[-1] = np.nan
        wvfm = np.append(wvfm, np.nan)
        try:
            result[ch]['time'] = np.append(result[ch]['time'], xs)
            result[ch]['adc'] = np.append(result[ch]['adc'], wvfm)
        except KeyError:
            result[ch] = {'time': xs, 'adc': wvfm}

    return result
