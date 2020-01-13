
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
import math

class opticalviewport(QtGui.QWidget):
  def __init__(self, geometry, plane=-1):
    super(opticalviewport, self).__init__()

    self._geometry = geometry

    self._last_clicked_pmts = []
    self._last_clicked_arapucas = []


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

    self.add_button_layout()
    
    self._wf_view.setMaximumHeight(200)
    self._wf_view.setMinimumHeight(200)

    for view in self._opdet_views:
        view.setMaximumHeight(500)
        view.setMinimumHeight(50)

    self._totalLayout.setAlignment(QtCore.Qt.AlignTop)


  def add_button_layout(self):

    self._tpc_all_button = QtGui.QRadioButton("All Cryos and TPCs")
    self._tpc_all_button.setToolTip("Shows all TPCs.")
    self._tpc_all_button.clicked.connect(self.viewSelectionWorker) 

    self._tpc_buttons = []
    for cryo in range(self._geometry.nCryos()):
        for tpc in range(self._geometry.nTPCs()):
            tpc_button = QtGui.QRadioButton("Cryo "+str(cryo)+", TPC "+str(tpc))
            tpc_button.setToolTip("Shows only Cryo "+str(cryo)+", TPC "+str(tpc))
            tpc_button.clicked.connect(self.viewSelectionWorker) 
            self._tpc_buttons.append(tpc_button)
    
    self._buttonLayout = QtGui.QHBoxLayout()

    # self.increasebutton = QtGui.QPushButton("Increase Amplitude")
    # self.decreasebutton = QtGui.QPushButton("Decrease Amplitude")

    self._buttonLayout.addWidget(self._tpc_all_button)
    for item in self._tpc_buttons:
        self._buttonLayout.addWidget(item)

    # self._buttonLayout.addWidget(self.increasebutton)
    # self._buttonLayout.addWidget(self.decreasebutton)

    self._totalLayout.addLayout(self._buttonLayout)

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

  def init_opdet_ui(self):

    self._opdet_plots = []
    self._pmts = []
    self._arapucas = []

    for tpc in range(self._geometry.nTPCs() * self._geometry.nCryos()):
        opdet_plot = self._opdet_views[tpc].addPlot()
    
        these_pmts = pmts(self._geometry, tpc=tpc)
        opdet_plot.addItem(these_pmts)
        these_pmts.sigClicked.connect(self.pmtClickWorker)
        these_pmts.scene().sigMouseMoved.connect(these_pmts.onMove)


        these_arapucas = arapucas(self._geometry, tpc=tpc)
        opdet_plot.addItem(these_arapucas)
        these_arapucas.sigClicked.connect(self.arapucaClickWorker)
        these_arapucas.scene().sigMouseMoved.connect(these_arapucas.onMove)

        self._opdet_plots.append(opdet_plot)
        self._pmts.append(these_pmts)
        self._arapucas.append(these_arapucas)

  def drawOpDetWvf(self, data):
    self._wf_view.drawOpDetWvf(data)


  def getWidget(self):
    return self, self._totalLayout

  def connectStatusBar(self, statusBar):
    self._statusBar = statusBar
    for t in range(0, len(self._pmts)):
        self._pmts[t].connectStatusBar(self._statusBar)
        self._pmts[t].connectStatusBar(self._statusBar)
        self._arapucas[t].connectStatusBar(self._statusBar)
        self._arapucas[t].connectStatusBar(self._statusBar)

  def pmtClickWorker(self, plot, points):
    for p in self._last_clicked_pmts:
      p.resetPen()
    for p in points:
      p.setPen('r', width=4)
      self._selected_ch = p.data()['id']

    self._last_clicked_pmts = points

    self._wf_view.drawWf(self._selected_ch)

  def arapucaClickWorker(self, plot, points):
    for p in self._last_clicked_arapucas:
      p.resetPen()
    for p in points:
      p.setPen('r', width=4)
      self._selected_ch = p.data()['id']

    self._last_clicked_arapucas = points

    self._wf_view.drawWf(self._selected_ch)




_bordercol_ = {
    'pmt'        : (255,255,255,255),
    'barepmt'    : (0,0,255,255),
    'arapucaT1'  : (34,139,34), 
    'arapucaT2'  : (34,139,34), 
    'xarapuca'   : (34,139,34), 
    'xarapucaT1' : (34,139,34), 
    'xarapucaT2' : (34,139,34)
}

class pmts(pg.ScatterPlotItem):
  '''
  This class handles the drawing of the
  PMTs as a scatter plot
  '''
  def __init__(self, geom, tpc=0):
    super(pmts, self).__init__()

    opdets_x, opdets_y, opdets_z = geom.opdetLoc()
    opdets_name = geom.opdetName()
    diameter = geom.opdetRadius() * 2

    names = ['pmt', 'barepmt']

    brush = (0,0,0,0)
    
    self._opdet_circles = []

    for d in range(0, len(opdets_x)):
        if opdets_name[d] in names:
            if ((opdets_x[d] < -100 and tpc == 0) or
               (opdets_x[d] > -100 and opdets_x[d] < 0 and tpc == 1) or
               (opdets_x[d] > 0 and opdets_x[d] < 100 and tpc == 2) or
               (opdets_x[d] > 100 and tpc == 3)):
                self._opdet_circles.append({'pos'    : (opdets_z[d], opdets_y[d]), 
                                            'size'   : diameter, 
                                            'pen'    : {'color': _bordercol_[opdets_name[d]], 'width': 2}, 
                                            'brush'  : brush, 
                                            'symbol' : 'o', 
                                            'data'   : {'id': d, 'highlight': False}})

    self.setAcceptHoverEvents(True)
    self.addPoints(self._opdet_circles)

    self._opdets_name = opdets_name
    self._opdets_x = opdets_x
    self._opdets_y = opdets_y
    self._opdets_z = opdets_z

  def connectStatusBar(self, statusBar):
    self._statusBar = statusBar

  # def hoverEnterEvent(self, e):
  #     print ('hoverEnterEvent')

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
        else:
            message += "OpDetName: "
            message += opdet_name
            message += ";   X: "
            message += "{0:.1f}".format(self._opdets_x[opdet_id])
            message += ";   Y: "
            message += "{0:.1f}".format(self._opdets_y[opdet_id])
            message += ";   Z: "
            message += "{0:.1f}".format(self._opdets_x[opdet_id])
        self._statusBar.showMessage(message)
    


class arapucas(pg.ScatterPlotItem):
  '''
  This class handles the drawing of the
  arapucas as a scatter plot
  '''
  def __init__(self, geom, tpc=0):
    super(arapucas, self).__init__()

    names = ['arapucaT1', 'arapucaT2', 'xarapuca', 'xarapucaT1', 'xarapucaT2']

    opdets_x, opdets_y, opdets_z = geom.opdetLoc()
    opdets_name = geom.opdetName()
    size = 6

    brush = (0,0,0,0)
    
    self._opdet_circles = []

    for d in range(0, len(opdets_x)):
        if opdets_name[d] in names:
            if (opdets_x[d] < 0 and tpc == 0) or (opdets_x[d] > 0 and tpc == 1):
                self._opdet_circles.append({'pos'    : (opdets_z[d], opdets_y[d]), 
                                            'size'   : size, 
                                            'pen'    : {'color': _bordercol_[opdets_name[d]], 'width': 2}, 
                                            'brush'  : brush, 
                                            'symbol' : 's', 
                                            'data'   : {'id': d, 'highlight': False}})

    self.setAcceptHoverEvents(True)
    self.addPoints(self._opdet_circles)

    self._opdets_name = opdets_name
    self._opdets_x = opdets_x
    self._opdets_y = opdets_y
    self._opdets_z = opdets_z


  def connectStatusBar(self, statusBar):
    self._statusBar = statusBar

  # def hoverEnterEvent(self, e):
  #     print ('hoverEnterEvent')

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
        else:
            message += "OpDetName: "
            message += opdet_name
            message += ";   X: "
            message += "{0:.1f}".format(self._opdets_x[opdet_id])
            message += ";   Y: "
            message += "{0:.1f}".format(self._opdets_y[opdet_id])
            message += ";   Z: "
            message += "{0:.1f}".format(self._opdets_x[opdet_id])
        self._statusBar.showMessage(message)




class optical_waveform_view(pg.GraphicsLayoutWidget):

  def __init__(self, geometry, plane=-1):
    super(optical_waveform_view, self).__init__(border=None)

    self._geometry = geometry

    self._data = None

    self._wf_plot = pg.PlotItem(name="OpDetWaveform")
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

    opdets_name = self._geometry.opdetName()

    data_x = np.linspace(-1250, 2500, len(data[0]))

    counter = 0
    for ch in range(0, len(opdets_name)):
        name = opdets_name[ch]

        if name == 'pmt':
            data_y = data[ch]
            data_y = data_y + counter * offset
            # self._wf_plot.plot(x=data_x, y=data_y)
            counter += 1
            if counter > 7:
                break

    self._wf_plot.autoRange()

  def drawWf(self, ch):

    if self._data is None:
        print ('OpDetWaveform data not loaded. No waveform to display.')
        return

    self._wf_plot.clear()

    data_x = np.linspace(-1250, 2500, len(self._data[0]))
    data_y = self._data[ch]

    # Remove the dafault values from the entries to be plotted
    default_value_indexes = np.where(data_y == self._geometry.opdetDefaultValue())
    data_x = np.delete(data_x, default_value_indexes)
    data_y = np.delete(data_y, default_value_indexes)

    # self._wf_plot.plot(x=data_x, y=data_y, connect=False, symbol='o')
    self._wf_plot.plot(x=data_x, y=data_y)

    self._wf_plot.autoRange()

   
    
