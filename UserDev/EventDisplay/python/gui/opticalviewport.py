
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
import math

from .qrangeslider import QRangeSlider
from .optical_elements import Pmts, Arapucas, _bordercol_

class opticalviewport(QtGui.QWidget):
  def __init__(self, geometry, plane=-1):
    super(opticalviewport, self).__init__()

    self._geometry = geometry

    self._last_clicked_pmts = []
    self._last_clicked_arapucas = []

    self._flashes = {}

    self._totalLayout = QtGui.QVBoxLayout()
    self.setLayout(self._totalLayout)

    self._wf_view = optical_waveform_view(self._geometry)
    self._totalLayout.addWidget(self._wf_view)

    self._opdet_views = []
    for tpc in range(self._geometry.nTPCs() * self._geometry.nCryos()):
        opdet_view = pg.GraphicsLayoutWidget()
        self._totalLayout.addWidget(opdet_view)
        self._opdet_views.append(opdet_view)

    self.init_opdet_ui()

    self._flash_time_view = flash_time_view(self._geometry)
    self._totalLayout.addWidget(self._flash_time_view)
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

    self._totalLayout.setAlignment(QtCore.Qt.AlignTop)


  def add_button_layout(self):

    self._bg1 = QtGui.QButtonGroup(self)
    self._tpc_all_button = QtGui.QRadioButton("All Cryos and TPCs")
    self._tpc_all_button.setToolTip("Shows all TPCs.")
    self._tpc_all_button.clicked.connect(self.viewSelectionWorker)
    self._bg1.addButton(self._tpc_all_button)

    self._tpc_buttons = []
    for cryo in range(self._geometry.nCryos()):
        for tpc in range(self._geometry.nTPCs()):
            tpc_button = QtGui.QRadioButton("Cryo "+str(cryo)+", TPC "+str(tpc))
            tpc_button.setToolTip("Shows only Cryo "+str(cryo)+", TPC "+str(tpc))
            tpc_button.clicked.connect(self.viewSelectionWorker)
            self._bg1.addButton(tpc_button)
            self._tpc_buttons.append(tpc_button)

    self._buttonLayout = QtGui.QHBoxLayout()

    self._buttonLayout.addWidget(self._tpc_all_button)
    for item in self._tpc_buttons:
        self._buttonLayout.addWidget(item)


    self._bg2 = QtGui.QButtonGroup(self)
    txt = QtGui.QLabel("Show:")
    self._show_raw_btn = QtGui.QRadioButton("Raw Data")
    self._show_raw_btn.clicked.connect(self._raw_flash_switch_worker)
    self._show_raw_btn.setChecked(True)
    self._show_raw = True
    self._bg2.addButton(self._show_raw_btn)
    self._show_flash_btn = QtGui.QRadioButton("Flashes")
    self._show_flash_btn.clicked.connect(self._raw_flash_switch_worker)
    self._bg2.addButton(self._show_flash_btn)

    self._raw_flash_btn_layout = QtGui.QHBoxLayout()
    self._raw_flash_btn_layout.addWidget(txt)
    self._raw_flash_btn_layout.addWidget(self._show_raw_btn)
    self._raw_flash_btn_layout.addWidget(self._show_flash_btn)


    # self._time_range_layout = QtGui.QHBoxLayout()

    # self._time_range_title = QtGui.QLabel("Time range [us]:")
    # self._time_range_layout.addWidget(self._time_range_title)

    # self._time_range = QRangeSlider()
    # self._time_range.setMin(-10)
    # self._time_range.setMax(100)
    # self._time_range.setRange(0, 10)
    # self._time_range.endValueChanged.connect(self.time_range_worker)
    # self._time_range.maxValueChanged.connect(self.time_range_worker)
    # self._time_range.minValueChanged.connect(self.time_range_worker)
    # self._time_range.startValueChanged.connect(self.time_range_worker)
    # self._time_range.minValueChanged.connect(self.time_range_worker)
    # self._time_range.maxValueChanged.connect(self.time_range_worker)
    # self._time_range.startValueChanged.connect(self.time_range_worker)
    # self._time_range.endValueChanged.connect(self.time_range_worker)
    # self._time_range_layout.addWidget(self._time_range)

    # for p in self._pmts:
    #     p.set_time_range(self._time_range.getRange())

    self._totalLayout.addLayout(self._buttonLayout)
    self._totalLayout.addLayout(self._raw_flash_btn_layout)
    # self._totalLayout.addLayout(self._time_range_layout)
    # self._totalLayout.addWidget(self._time_range_title)
    # self._totalLayout.addWidget(self._time_range)


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
    if self.sender() == self._show_raw_btn:
        self._show_raw = True
    else:
        self._show_raw = False


  def time_range_worker(self):
    # for p in self._pmts:
    #     p.set_time_range(self._time_range.getRange())
    for p in self._pmts:
        p.set_time_range(self._time_window.getRegion())
    return


  def init_opdet_ui(self):

    self._opdet_plots = []
    self._pmts = []
    self._arapucas = []
    self._opdetscales = []

    for tpc in range(self._geometry.nTPCs() * self._geometry.nCryos()):
        opdet_plot = self._opdet_views[tpc].addPlot()
        opdet_plot.setLabel(axis='left', text='Y [cm]')
        opdet_plot.setLabel(axis='bottom', text='Z [cm]')

        this_scale = pg.GradientEditorItem(orientation='right')
        self._opdet_views[tpc].addItem(this_scale, 0, 1)

        these_pmts = Pmts(self._geometry, tpc=tpc, pmtscale=this_scale)
        opdet_plot.addItem(these_pmts)
        these_pmts.sigClicked.connect(self.pmtClickWorker)
        these_pmts.scene().sigMouseMoved.connect(these_pmts.onMove)

        these_arapucas = Arapucas(self._geometry, tpc=tpc, pmtscale=this_scale)
        opdet_plot.addItem(these_arapucas)
        these_arapucas.sigClicked.connect(self.arapucaClickWorker)
        these_arapucas.scene().sigMouseMoved.connect(these_arapucas.onMove)

        self._opdet_plots.append(opdet_plot)
        self._pmts.append(these_pmts)
        self._arapucas.append(these_arapucas)
        self._opdetscales.append(this_scale)


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
    return self, self._totalLayout


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





class optical_waveform_view(pg.GraphicsLayoutWidget):

  def __init__(self, geometry, plane=-1):
    super(optical_waveform_view, self).__init__(border=None)

    self._geometry = geometry

    self._data = None

    self._wf_plot = pg.PlotItem(name="OpDetWaveform")
    self._wf_plot.setLabel(axis='left', text='ADC')
    self._wf_plot.setLabel(axis='bottom', text='Ticks')
    # self._wf_linear_region = pg.LinearRegionItem(values=[0,30], orientation=pg.LinearRegionItem.Vertical)
    # self._wf_plot.addItem(self._wf_linear_region)
    self.addItem(self._wf_plot)


  def getWidget(self):
    return self._widget, self._totalLayout

  # def init_geometry(self):

  #   opdets_x, opdets_y, opdets_z = self._geometry.opdetLoc()
  #   opdets_name = self._geometry.opdetName()
  #   diameter = self._geometry.opdetRadius() * 2

  #   self._opdet_circles = []
  #   for d in range(0, len(opdets_x)):
  #       # print('Adding opdet', opdets_x[d], opdets_y[d], diameter, diameter)
  #       self._opdet_circles.append(QtGui.QGraphicsEllipseItem(opdets_z[d], opdets_y[d], diameter, diameter))

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
    return self._widget, self._totalLayout


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


